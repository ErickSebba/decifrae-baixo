"""
MÓDULO DE TESTE POLIFÔNICO — Validação do pYIN por Corda
=========================================================
Testa especificamente os problemas que o pYIN monofônico não resolve:
    1. Detecção de notas simultâneas (duas cordas soando ao mesmo tempo)
    2. Correção de erro de oitava (2° harmônico detectado em vez da fundamental)
    3. Rejeição de notas fantasma em silêncios e ataques percussivos
    4. Isolamento por corda (o filtro passa-banda está funcionando?)
    5. Degradação progressiva com ruído e mix realista

Estrutura (espelhada no modulo_teste_2.py):
    1. GERADORES DE SINAL         — reutilizados do módulo original
    2. GERADORES POLIFÔNICOS      — novos: constroem acordes e intervalos simultâneos
    3. MÉTRICAS POLIFÔNICAS       — novas: precision/recall por nota, detecção de pares
    4. CASOS DE TESTE             — 5 suites específicas para polifonia
    5. RUNNER E RELATÓRIO         — imprime tabela e plota painéis comparativos

Nenhum arquivo de áudio externo é necessário.

Execução:
    python modulo_teste_polifonico.py
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Módulo sendo testado
from modulo_pyin_polifonico import (
    analisar_baixo_polifonico,
    bridge_pyin_para_notas_poli,
    _filtro_passa_banda,
    CORDAS_5, CORDAS_4,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

SR               = 22050
TOLERANCIA_CENTS = 50    # ±50 cents = ±meio semitom
TOLERANCIA_MIDI  = 0.5
HOP_LENGTH       = 1024


# ══════════════════════════════════════════════════════════════════════════════
# 1. GERADORES DE SINAL  (mesmos do modulo_teste_2.py)
# ══════════════════════════════════════════════════════════════════════════════

def sinal_puro(freq_hz: float, duracao: float, sr: int = SR,
               amplitude: float = 0.8) -> np.ndarray:
    t = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def sinal_harmonicos(freq_hz: float, duracao: float, sr: int = SR,
                     amplitude: float = 0.8, perfil: str = "baixo") -> np.ndarray:
    """
    Fundamental + harmônicos realistas de baixo elétrico.
    perfil "baixo"    → expoente 1.8, 6 harmônicos  (nota dedilhada)
    perfil "baixo_sl" → expoente 1.2, 9 harmônicos  (slap — harmônicos ricos)
    perfil "puro"     → só fundamental
    """
    perfis = {
        "baixo":    {"expoente": 1.8, "n_harm": 6},
        "baixo_sl": {"expoente": 1.2, "n_harm": 9},
        "puro":     {"expoente": 99,  "n_harm": 1},
    }
    cfg = perfis.get(perfil, perfis["baixo"])
    t   = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    y   = np.zeros_like(t)
    for h in range(1, cfg["n_harm"] + 1):
        y += (amplitude / h ** cfg["expoente"]) * np.sin(2 * np.pi * freq_hz * h * t)
    pico = np.max(np.abs(y))
    return (y / pico * amplitude) if pico > 0 else y


def sinal_harmonicos_decay(freq_hz: float, duracao: float, sr: int = SR,
                            amplitude: float = 0.8, tau: float = 0.5,
                            perfil: str = "baixo") -> np.ndarray:
    """Harmônicos + envelope exponencial decrescente (caso mais realista)."""
    t        = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    envelope = np.exp(-t / tau)
    base     = sinal_harmonicos(freq_hz, duracao, sr, amplitude=1.0, perfil=perfil)
    y        = base * envelope * amplitude
    pico = np.max(np.abs(y))
    return (y / pico * amplitude) if pico > 0 else y


def ruido_branco(n_amostras: int, sr: int = SR,
                 amplitude: float = 0.05) -> np.ndarray:
    """Ruído gaussiano branco — simula hum de amplificador."""
    return np.random.normal(0, amplitude, n_amostras).astype(np.float32)


def percussao_sintetica(duracao: float, sr: int = SR,
                         bpm: float = 120.0,
                         amplitude: float = 0.3) -> np.ndarray:
    """
    Simula um bumbo/caixa sintético em pulsos periódicos.
    Importante para testar se o voiced_flag rejeita ataques percussivos
    que poderiam ser confundidos com notas de baixo.
    """
    n = int(sr * duracao)
    y = np.zeros(n)
    periodo = int(sr * 60 / bpm)    # amostras por beat
    for i in range(0, n, periodo):
        # Pulso de ruído com decaimento rápido
        comprimento = min(int(sr * 0.05), n - i)   # 50 ms de decaimento
        if comprimento <= 0:
            break
        t_loc  = np.arange(comprimento) / sr
        pulso  = amplitude * np.exp(-t_loc / 0.01) * np.random.normal(0, 1, comprimento)
        y[i:i + comprimento] += pulso
    return y.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GERADORES POLIFÔNICOS  (novos)
# ══════════════════════════════════════════════════════════════════════════════

def construir_nota_simples(freq_hz: float, duracao: float,
                            sr: int = SR, **kwargs) -> tuple[np.ndarray, list[dict]]:
    """Gera áudio de uma nota única com gabarito."""
    audio = sinal_harmonicos_decay(freq_hz, duracao, sr, **kwargs)
    gab   = [{"freq_hz": freq_hz,
              "nota_nome": librosa.hz_to_note(freq_hz, octave=True),
              "inicio": 0.0, "fim": duracao}]
    return audio.astype(np.float32), gab


def construir_intervalo_simultaneo(
    freq1_hz: float, freq2_hz: float,
    duracao: float,
    amp1: float = 0.7, amp2: float = 0.6,
    sr: int = SR,
    perfil: str = "baixo",
    tau: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """
    Gera dois sinais soando AO MESMO TEMPO e os soma.
    Esse é o caso-chave que o pYIN monofônico falha.

    Retorna
    -------
    (audio_somado, gabarito)
    gabarito: ambas as notas com mesmo inicio/fim
    """
    y1   = sinal_harmonicos_decay(freq1_hz, duracao, sr, amplitude=amp1,
                                   perfil=perfil, tau=tau)
    y2   = sinal_harmonicos_decay(freq2_hz, duracao, sr, amplitude=amp2,
                                   perfil=perfil, tau=tau)
    soma = y1 + y2
    # Normaliza para não clipar
    pico = np.max(np.abs(soma))
    if pico > 0:
        soma = soma / pico * 0.8
    gab = [
        {"freq_hz": freq1_hz, "nota_nome": librosa.hz_to_note(freq1_hz, octave=True),
         "inicio": 0.0, "fim": duracao},
        {"freq_hz": freq2_hz, "nota_nome": librosa.hz_to_note(freq2_hz, octave=True),
         "inicio": 0.0, "fim": duracao},
    ]
    return soma.astype(np.float32), gab


def construir_sequencia_com_overlap(
    eventos: list[dict],
    sr: int = SR,
) -> tuple[np.ndarray, list[dict]]:
    """
    Constrói áudio onde eventos podem se sobrepor no tempo.
    Cada evento: {"freq_hz", "inicio", "duracao", "amplitude", "perfil", "tau"}

    Permite montar cenários como:
        - nota na corda A soando, enquanto nota na corda E começa
        - pedal de baixo sustentado enquanto melodia sobe

    Retorna (audio, gabarito) com gabarito listando cada evento individualmente.
    """
    # Descobre duração total
    dur_total = max(e["inicio"] + e["duracao"] for e in eventos)
    n_total   = int(sr * dur_total)
    y         = np.zeros(n_total, dtype=np.float64)
    gabarito  = []

    for ev in eventos:
        freq  = ev["freq_hz"]
        amp   = ev.get("amplitude", 0.7)
        perfil = ev.get("perfil", "baixo")
        tau   = ev.get("tau", 0.5)
        dur   = ev["duracao"]
        ini_s = ev["inicio"]

        segmento = sinal_harmonicos_decay(freq, dur, sr,
                                           amplitude=amp, perfil=perfil, tau=tau)
        ini_idx  = int(ini_s * sr)
        fim_idx  = min(ini_idx + len(segmento), n_total)
        y[ini_idx:fim_idx] += segmento[:fim_idx - ini_idx]

        gabarito.append({
            "freq_hz":   freq,
            "nota_nome": librosa.hz_to_note(freq, octave=True),
            "inicio":    ini_s,
            "fim":       ini_s + dur,
        })

    # Normaliza
    pico = np.max(np.abs(y))
    if pico > 0:
        y = y / pico * 0.8
    return y.astype(np.float32), gabarito


# ══════════════════════════════════════════════════════════════════════════════
# 3. MÉTRICAS POLIFÔNICAS  (novas)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResultadoPolifonico:
    """Resultado de um caso de teste para o pYIN polifônico."""
    nome:             str
    notas_esperadas:  list[str]          # nomes das notas do gabarito
    notas_detectadas: list[str]          # nomes únicas detectadas

    # Métricas por nota esperada
    por_nota: dict = field(default_factory=dict)
    # {"A1": {"detectada": True, "melhor_erro_cents": 0.0, "corda_correta": True}}

    # Métricas globais
    precision:        float = 0.0  # das detectadas, quantas estão no gabarito
    recall:           float = 0.0  # das esperadas, quantas foram detectadas
    f1:               float = 0.0
    notas_fantasma:   int   = 0    # detectadas que NÃO estão no gabarito
    oitavas_erradas:  int   = 0    # detectadas 1 oitava acima/abaixo da esperada

    diagnostico: list[str] = field(default_factory=list)


def _midi_de_nota(nome: str) -> float:
    return float(librosa.note_to_midi(nome))


def calcular_metricas_polifonicas(
    segmentos: list[dict],
    gabarito: list[dict],
    nome_teste: str,
    tolerancia_midi: float = TOLERANCIA_MIDI,
) -> ResultadoPolifonico:
    """
    Compara segmentos detectados pelo pYIN polifônico com o gabarito.

    Lógica de matching:
        - Para cada nota do gabarito, busca o segmento detectado com menor
          erro de pitch que coincida no tempo (sobreposição > 30% da duração).
        - Se encontrar dentro de ±tolerancia_midi semitons → TP (verdadeiro positivo)
        - Se encontrar com erro de exatamente 12 semitons → erro de oitava
        - Segmentos sem correspondência → nota fantasma

    Parâmetros
    ----------
    segmentos    : saída de analisar_baixo_polifonico()
    gabarito     : list[dict] com "freq_hz", "nota_nome", "inicio", "fim"
    nome_teste   : rótulo do teste (para o relatório)
    """
    notas_esp_nomes = [g["nota_nome"] for g in gabarito]

    # Converte gabarito para MIDI
    gab_midi = [(g, librosa.hz_to_midi(g["freq_hz"])) for g in gabarito]

    # Converte segmentos para lista de (seg, midi_detectado)
    segs_midi = [(s, float(s["nota_midi"])) for s in segmentos]

    # ── Matching: gabarito → segmento detectado ───────────────────────────────
    por_nota   = {}
    matched_segs = set()   # índices de segmentos já usados num match

    for gab_entry, gab_m in gab_midi:
        gab_ini = gab_entry["inicio"]
        gab_fim = gab_entry["fim"]
        gab_dur = gab_fim - gab_ini

        melhor_erro = float("inf")
        melhor_idx  = None
        melhor_seg  = None

        for idx, (seg, seg_m) in enumerate(segs_midi):
            if idx in matched_segs:
                continue
            # Verifica sobreposição temporal (> 30% da nota esperada)
            overlap = max(0, min(seg["fim"], gab_fim) - max(seg["inicio"], gab_ini))
            if gab_dur > 0 and (overlap / gab_dur) < 0.30:
                continue
            erro = abs(seg_m - gab_m)
            if erro < melhor_erro:
                melhor_erro = erro
                melhor_idx  = idx
                melhor_seg  = seg

        detectada      = melhor_erro <= tolerancia_midi
        oitava_errada  = (not detectada) and abs(melhor_erro - 12) < 1.5
        corda_correta  = False
        if melhor_seg and "corda" in melhor_seg:
            # Verifica se a corda detectada faz sentido para a frequência
            freq_esp = gab_entry["freq_hz"]
            for c in CORDAS_5:
                if c["fmin"] <= freq_esp <= c["fmax"]:
                    corda_correta = (melhor_seg["corda"] == c["nome"])
                    break

        por_nota[gab_entry["nota_nome"]] = {
            "detectada":         detectada,
            "oitava_errada":     oitava_errada,
            "melhor_erro_cents": melhor_erro * 100 if melhor_erro < float("inf") else None,
            "corda_correta":     corda_correta,
            "segmento_match":    melhor_seg,
        }
        if detectada and melhor_idx is not None:
            matched_segs.add(melhor_idx)

    # ── Notas fantasma ────────────────────────────────────────────────────────
    fantasmas = len(segs_midi) - len(matched_segs)
    fantasmas = max(0, fantasmas)  # pode ser negativo se match foi conservador

    # ── Erros de oitava globais ───────────────────────────────────────────────
    n_oitava = sum(1 for v in por_nota.values() if v["oitava_errada"])

    # ── Precision / Recall / F1 ───────────────────────────────────────────────
    tp = sum(1 for v in por_nota.values() if v["detectada"])
    fn = len(por_nota) - tp
    fp = fantasmas

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    notas_det_nomes = [librosa.midi_to_note(int(s["nota_midi"])) for s in segmentos]

    # ── Diagnóstico ───────────────────────────────────────────────────────────
    diag = []
    if recall < 0.5:
        diag.append(f"⚠ RECALL BAIXO ({recall*100:.0f}%): muitas notas esperadas não detectadas")
    if precision < 0.5:
        diag.append(f"⚠ PRECISÃO BAIXA ({precision*100:.0f}%): muitas notas fantasma")
    if n_oitava > 0:
        diag.append(f"⚠ ERRO DE OITAVA: {n_oitava} nota(s) com 1 oitava de erro")
    if fantasmas > len(gabarito):
        diag.append(f"⚠ EXCESSO DE FANTASMAS: {fantasmas} notas extras (gabarito={len(gabarito)})")
    cordas_ok = sum(1 for v in por_nota.values() if v["corda_correta"])
    if tp > 0 and cordas_ok < tp:
        diag.append(f"⚠ CORDA ERRADA: {tp - cordas_ok} nota(s) detectada na corda incorreta")
    if not diag:
        diag.append("✓ Sem problemas críticos")

    return ResultadoPolifonico(
        nome             = nome_teste,
        notas_esperadas  = notas_esp_nomes,
        notas_detectadas = notas_det_nomes,
        por_nota         = por_nota,
        precision        = precision,
        recall           = recall,
        f1               = f1,
        notas_fantasma   = fantasmas,
        oitavas_erradas  = n_oitava,
        diagnostico      = diag,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. CASES DE TESTE — 5 suites
# ══════════════════════════════════════════════════════════════════════════════

# Notas de referência organizadas por corda
NOTAS_POR_CORDA = {
    "B": [("B0", librosa.note_to_hz("B0")),  ("C1",  librosa.note_to_hz("C1")),
          ("D1", librosa.note_to_hz("D1")),  ("E1",  librosa.note_to_hz("E1"))],
    "E": [("E1", librosa.note_to_hz("E1")),  ("F1",  librosa.note_to_hz("F1")),
          ("G1", librosa.note_to_hz("G1")),  ("A1",  librosa.note_to_hz("A1"))],
    "A": [("A1", librosa.note_to_hz("A1")),  ("B1",  librosa.note_to_hz("B1")),
          ("C2", librosa.note_to_hz("C2")),  ("D2",  librosa.note_to_hz("D2"))],
    "D": [("D2", librosa.note_to_hz("D2")),  ("E2",  librosa.note_to_hz("E2")),
          ("F2", librosa.note_to_hz("F2")),  ("G2",  librosa.note_to_hz("G2"))],
    "G": [("G2", librosa.note_to_hz("G2")),  ("A2",  librosa.note_to_hz("A2")),
          ("B2", librosa.note_to_hz("B2")),  ("C3",  librosa.note_to_hz("C3"))],
}

# Intervalos musicalmente comuns no baixo (pares de cordas distintas)
INTERVALOS_TESTE = [
    # (nome, freq1, freq2, descrição musical)
    ("E1+A1",  librosa.note_to_hz("E1"),  librosa.note_to_hz("A1"),  "Quinta (cordas vizinhas E+A)"),
    ("A1+D2",  librosa.note_to_hz("A1"),  librosa.note_to_hz("D2"),  "Quinta (cordas vizinhas A+D)"),
    ("E1+D2",  librosa.note_to_hz("E1"),  librosa.note_to_hz("D2"),  "Sétima menor (cordas distantes E+D)"),
    ("B0+E1",  librosa.note_to_hz("B0"),  librosa.note_to_hz("E1"),  "Quinta (cordas graves B+E)"),
    ("A1+G2",  librosa.note_to_hz("A1"),  librosa.note_to_hz("G2"),  "Sétima menor (A+G — mais agudo)"),
    ("E1+G2",  librosa.note_to_hz("E1"),  librosa.note_to_hz("G2"),  "Décima (E+G — cordas extremas)"),
]


# ── Suite 1: Nota única por corda (baseline do isolamento de filtro) ──────────
def suite_nota_unica_por_corda() -> list[tuple]:
    """
    Testa se o pYIN detecta corretamente UMA nota por vez em cada corda.
    Caso de referência: se isto falhar, os casos polifônicos certamente falharão.
    """
    casos = []
    for nome_corda, notas in NOTAS_POR_CORDA.items():
        for nota_nome, freq_hz in notas:
            audio, gab = construir_nota_simples(freq_hz, duracao=1.0,
                                                 amplitude=0.7, tau=0.5)
            casos.append((audio, gab,
                          f"[CORDA {nome_corda}] {nota_nome} | nota única"))
    return casos


# ── Suite 2: Intervalos simultâneos (o caso crítico) ─────────────────────────
def suite_intervalos_simultaneos() -> list[tuple]:
    """
    Dois sinais somados soando ao mesmo tempo — o teste central do módulo.
    Gabarito: AMBAS as notas devem ser detectadas no mesmo intervalo de tempo.
    """
    casos = []
    for nome_int, f1, f2, desc in INTERVALOS_TESTE:
        # Caso ideal: amplitudes iguais, sinal limpo
        audio, gab = construir_intervalo_simultaneo(f1, f2, duracao=1.2)
        casos.append((audio, gab, f"[POLI] {nome_int} | {desc} | limpo"))

        # Com ruído leve (SNR ≈ 25 dB)
        ruidoso = audio + ruido_branco(len(audio), amplitude=0.03)
        casos.append((ruidoso, gab, f"[POLI] {nome_int} | {desc} | + ruído leve"))

        # Amplitudes desbalanceadas (uma nota dominante, outra fraca)
        audio_des, gab_des = construir_intervalo_simultaneo(
            f1, f2, duracao=1.2, amp1=0.75, amp2=0.25
        )
        casos.append((audio_des, gab_des,
                      f"[POLI] {nome_int} | {desc} | amp desbalanceada (3:1)"))
    return casos


# ── Suite 3: Sobreposição temporal parcial (legato polifônico) ────────────────
def suite_sobreposicao_parcial() -> list[tuple]:
    """
    Nota 1 começa, nota 2 entra enquanto nota 1 ainda está soando, depois nota 1 para.
    Simula hammer-on, legato ou simplesmente duas cordas com timings diferentes.
    """
    casos = []
    cenarios = [
        {
            "nome": "E1+A1 parcial (A1 entra no meio de E1)",
            "eventos": [
                {"freq_hz": librosa.note_to_hz("E1"), "inicio": 0.0,  "duracao": 1.0,
                 "amplitude": 0.7, "tau": 0.6},
                {"freq_hz": librosa.note_to_hz("A1"), "inicio": 0.5,  "duracao": 0.8,
                 "amplitude": 0.6, "tau": 0.5},
            ]
        },
        {
            "nome": "A1+D2 parcial (overlap de 400ms)",
            "eventos": [
                {"freq_hz": librosa.note_to_hz("A1"), "inicio": 0.0,  "duracao": 0.8,
                 "amplitude": 0.7, "tau": 0.5},
                {"freq_hz": librosa.note_to_hz("D2"), "inicio": 0.4,  "duracao": 0.8,
                 "amplitude": 0.65, "tau": 0.5},
            ]
        },
        {
            "nome": "B0+E1+A1 (três notas, entradas escalonadas)",
            "eventos": [
                {"freq_hz": librosa.note_to_hz("B0"), "inicio": 0.0,  "duracao": 1.4,
                 "amplitude": 0.6, "tau": 0.7},
                {"freq_hz": librosa.note_to_hz("E1"), "inicio": 0.3,  "duracao": 1.0,
                 "amplitude": 0.6, "tau": 0.5},
                {"freq_hz": librosa.note_to_hz("A1"), "inicio": 0.7,  "duracao": 0.7,
                 "amplitude": 0.55, "tau": 0.4},
            ]
        },
    ]
    for c in cenarios:
        audio, gab = construir_sequencia_com_overlap(c["eventos"])
        casos.append((audio, gab, f"[OVERLAP] {c['nome']}"))
    return casos


# ── Suite 4: Stress test de erro de oitava em notas críticas ─────────────────
def suite_erro_oitava_critico() -> list[tuple]:
    """
    Notas onde o pYIN costuma errar 1 oitava, com perfis de harmônicos que
    agravam o problema (harmônicos proeminentes = 2° harmônico mais forte).
    Testa se _corrigir_oitava() resolve.
    """
    notas_criticas = [
        ("B0",  librosa.note_to_hz("B0"),  "mais grave — 2° harm = B1"),
        ("E1",  librosa.note_to_hz("E1"),  "corda solta E — 2° harm = E2"),
        ("A1",  librosa.note_to_hz("A1"),  "corda solta A — 2° harm = A2"),
        ("D2",  librosa.note_to_hz("D2"),  "corda solta D — 2° harm = D3"),
        ("G2",  librosa.note_to_hz("G2"),  "corda solta G — 2° harm = G3"),
    ]
    casos = []
    for nome, freq, desc in notas_criticas:
        # Perfil "baixo_sl" → harmônicos muito ricos → caso mais difícil para oitava
        audio = sinal_harmonicos_decay(freq, duracao=1.5, amplitude=0.8,
                                        perfil="baixo_sl", tau=0.4)
        gab   = [{"freq_hz": freq, "nota_nome": nome, "inicio": 0.0, "fim": 1.5}]
        casos.append((audio.astype(np.float32), gab,
                      f"[OITAVA] {nome} | {desc} | slap (harm. rico)"))

        # Mesmo sinal + percussão para dificultar mais
        perc  = percussao_sintetica(1.5, amplitude=0.25)
        mix   = (audio + perc[:len(audio)]).astype(np.float32)
        casos.append((mix, gab,
                      f"[OITAVA] {nome} | {desc} | slap + percussão"))
    return casos


# ── Suite 5: Silêncios e notas fantasma ──────────────────────────────────────
def suite_fantasmas_e_silencio() -> list[tuple]:
    """
    Testa se o módulo gera notas onde não deveria haver nenhuma.
    Gabarito tem silêncios longos; qualquer detecção neles é falso positivo.
    """
    casos = []

    # Caso 1: silêncio puro — nenhuma nota deve ser detectada
    silencio = np.zeros(int(SR * 2.0), dtype=np.float32)
    gab_vazio = []  # gabarito vazio = nenhuma nota esperada
    casos.append((silencio, gab_vazio, "[FANTASMA] Silêncio puro (0 notas esperadas)"))

    # Caso 2: percussão sem notas de baixo — não deve gerar notas
    perc = percussao_sintetica(2.0, amplitude=0.5).astype(np.float32)
    casos.append((perc, gab_vazio, "[FANTASMA] Percussão pura (0 notas esperadas)"))

    # Caso 3: nota curta + silêncios longos nas bordas
    freq = librosa.note_to_hz("A1")
    nota = sinal_harmonicos_decay(freq, duracao=0.5, amplitude=0.7, tau=0.4)
    bordas = np.concatenate([
        np.zeros(int(SR * 0.5)),
        nota,
        np.zeros(int(SR * 0.5)),
    ]).astype(np.float32)
    gab_bordas = [{"freq_hz": freq, "nota_nome": "A1", "inicio": 0.5, "fim": 1.0}]
    casos.append((bordas, gab_bordas, "[FANTASMA] A1 curta entre silêncios longos"))

    # Caso 4: ruído branco moderado — não deve gerar notas
    ruido = ruido_branco(int(SR * 2.0), amplitude=0.3).astype(np.float32)
    casos.append((ruido, gab_vazio, "[FANTASMA] Ruído branco puro (0 notas esperadas)"))

    # Caso 5: percussão + nota real — distingue nota do ruído percussivo
    freq2 = librosa.note_to_hz("E1")
    nota2 = sinal_harmonicos_decay(freq2, duracao=0.8, amplitude=0.7, tau=0.5)
    perc2 = percussao_sintetica(0.8, amplitude=0.3)
    mix   = (nota2 + perc2[:len(nota2)]).astype(np.float32)
    gab_mix = [{"freq_hz": freq2, "nota_nome": "E1", "inicio": 0.0, "fim": 0.8}]
    casos.append((mix, gab_mix, "[FANTASMA] E1 + percussão simultânea"))

    return casos


# ══════════════════════════════════════════════════════════════════════════════
# 5. RUNNER — executa o módulo em array numpy (sem salvar arquivo)
# ══════════════════════════════════════════════════════════════════════════════

def rodar_polifonico_em_array(
    audio: np.ndarray,
    num_cordas: int = 5,
    sr: int = SR,
) -> list[dict]:
    """
    Executa analisar_baixo_polifonico() diretamente em um array numpy,
    sem precisar salvar para disco.

    Internamente usa um objeto file-like que o librosa.load() aceita
    via soundfile, mas como já temos o array, bypassa o load e chama
    as funções internas diretamente.
    """
    import librosa.effects

    # Replica o pré-processamento de analisar_baixo_polifonico()
    y_harmonic, _ = librosa.effects.hpss(audio.astype(np.float32), margin=3.0)

    from modulo_pyin_polifonico import _pyin_uma_corda, _mesclar_segmentos
    cordas = CORDAS_5 if num_cordas == 5 else CORDAS_4

    todos = []
    for corda in cordas:
        segs = _pyin_uma_corda(y_harmonic, sr, corda,
                                frame_length=4096, hop_length=HOP_LENGTH)
        todos.extend(segs)

    return _mesclar_segmentos(todos, tolerancia_overlap=0.03, duracao_minima=0.10)


# ══════════════════════════════════════════════════════════════════════════════
# 6. RELATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resultado(r: ResultadoPolifonico) -> None:
    larg = 72
    print(f"\n{'─'*larg}")
    print(f"  {r.nome}")
    print(f"{'─'*larg}")
    print(f"  Esperadas : {', '.join(r.notas_esperadas) or '(nenhuma)'}")
    print(f"  Detectadas: {', '.join(sorted(set(r.notas_detectadas))) or '(nenhuma)'}")
    print(f"  Precision : {r.precision*100:5.1f}%   "
          f"Recall: {r.recall*100:5.1f}%   "
          f"F1: {r.f1*100:5.1f}%")
    print(f"  Fantasmas : {r.notas_fantasma}   "
          f"Erros oitava: {r.oitavas_erradas}")

    if r.por_nota:
        print(f"\n  Detalhe por nota:")
        for nome_nota, info in r.por_nota.items():
            status = "✓" if info["detectada"] else ("≈8va" if info["oitava_errada"] else "✗")
            erro_s = (f"{info['melhor_erro_cents']:.0f}¢"
                      if info["melhor_erro_cents"] is not None else "—")
            corda_s = ("corda ok" if info["corda_correta"] else "corda?")
            print(f"    {status} {nome_nota:<5}  erro={erro_s:<8} {corda_s}")

    print(f"\n  Diagnóstico:")
    for d in r.diagnostico:
        print(f"    {d}")


def plotar_suite_polifonica(
    nome_suite: str,
    casos: list[tuple],
    resultados: list[ResultadoPolifonico],
    audios: list[np.ndarray],
    segmentos_lista: list[list[dict]],
) -> None:
    """
    Painel de visualização para uma suite polifônica.

    Linha 1: Barras de Precision / Recall / F1 para todos os casos.
    Linhas seguintes: waveform + piano roll comparativo (gabarito vs detectado)
                      para os primeiros N casos.
    """
    n = len(resultados)
    if n == 0:
        return

    # ── Painel de métricas (barras horizontais) ───────────────────────────────
    fig_m, axes_m = plt.subplots(1, 4, figsize=(18, max(3, n * 0.45 + 2)))
    fig_m.suptitle(f"Métricas Polifônicas — {nome_suite}", fontsize=13, fontweight="bold")

    nomes_curtos = [r.nome.split("|")[-1].strip()[:40] for r in resultados]
    prec  = [r.precision * 100 for r in resultados]
    rec   = [r.recall    * 100 for r in resultados]
    f1s   = [r.f1        * 100 for r in resultados]
    fant  = [r.notas_fantasma  for r in resultados]

    def _cores(vals, bom=80, med=50, inv=False):
        resultado = []
        for v in vals:
            if inv:
                c = "#2ecc71" if v == 0 else "#e67e22" if v <= 2 else "#e74c3c"
            else:
                c = "#2ecc71" if v >= bom else "#e67e22" if v >= med else "#e74c3c"
            resultado.append(c)
        return resultado

    y_pos = range(n)
    for ax, vals, cores_fn, titulo, fmt, inv in [
        (axes_m[0], prec, lambda v: _cores(v),         "Precision (%)",    "{:.0f}%", False),
        (axes_m[1], rec,  lambda v: _cores(v),         "Recall (%)",       "{:.0f}%", False),
        (axes_m[2], f1s,  lambda v: _cores(v),         "F1 (%)",           "{:.0f}%", False),
        (axes_m[3], fant, lambda v: _cores(v, inv=True),"Notas Fantasma",  "{:.0f}",  True),
    ]:
        bars = ax.barh(y_pos, vals, color=cores_fn(vals), edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nomes_curtos, fontsize=7)
        ax.set_title(titulo, fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=7)
        if titulo != "Notas Fantasma":
            ax.set_xlim(0, 110)

    plt.tight_layout()
    plt.show()

    # ── Detalhe: waveform + piano roll (máx 6 por suite) ─────────────────────
    n_plot = min(n, 6)
    fig_d  = plt.figure(figsize=(16, n_plot * 3.5))
    fig_d.suptitle(f"Piano Roll: gabarito vs detectado — {nome_suite}",
                   fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(n_plot, 2, figure=fig_d, hspace=0.6, wspace=0.3)

    cores_gabarito  = "#2ecc71"
    cores_detectado = "#e74c3c"

    for idx in range(n_plot):
        ax_w  = fig_d.add_subplot(gs[idx, 0])
        ax_pr = fig_d.add_subplot(gs[idx, 1])

        audio    = audios[idx]
        segs     = segmentos_lista[idx]
        res      = resultados[idx]
        _, gab   = casos[idx][0], casos[idx][1]   # áudio e gabarito

        # Waveform
        t_wave = np.linspace(0, len(audio) / SR, len(audio))
        ax_w.plot(t_wave, audio, color="#3498db", linewidth=0.4, alpha=0.8)
        ax_w.set_title(res.nome.split("|")[-1].strip()[:45], fontsize=8)
        ax_w.set_ylabel("Amplitude", fontsize=7)
        ax_w.tick_params(labelsize=6)
        ax_w.set_facecolor("#f8f9fa")

        # Piano Roll
        midi_vals = set()

        # Gabarito (verde)
        for g in gab:
            m = librosa.hz_to_midi(g["freq_hz"])
            midi_vals.add(round(m))
            dur = g["fim"] - g["inicio"]
            rect = mpatches.Rectangle(
                (g["inicio"], m - 0.4), dur, 0.8,
                color=cores_gabarito, alpha=0.5, ec="darkgreen", lw=1.5,
                label="Gabarito",
            )
            ax_pr.add_patch(rect)
            ax_pr.text(g["inicio"] + dur / 2, m + 0.6, g["nota_nome"],
                       ha="center", va="bottom", fontsize=6,
                       color="darkgreen", fontweight="bold")

        # Detectado (vermelho)
        for seg in segs:
            m   = float(seg["nota_midi"])
            midi_vals.add(round(m))
            dur = seg["fim"] - seg["inicio"]
            rect = mpatches.Rectangle(
                (seg["inicio"], m - 0.38), dur, 0.76,
                color=cores_detectado, alpha=0.4, ec="darkred", lw=1.0,
                linestyle="--", label="Detectado",
            )
            ax_pr.add_patch(rect)
            nome_seg = librosa.midi_to_note(int(m))
            ax_pr.text(seg["inicio"] + dur / 2, m - 0.65, nome_seg,
                       ha="center", va="top", fontsize=6,
                       color="darkred")

        if midi_vals:
            mn, mx = min(midi_vals) - 2, max(midi_vals) + 2
            ax_pr.set_ylim(mn, mx)
            yticks = [v for v in range(int(mn), int(mx) + 1)]
            ax_pr.set_yticks(yticks)
            ax_pr.set_yticklabels([librosa.midi_to_note(v) for v in yticks], fontsize=6)
        ax_pr.set_xlim(0, len(audio) / SR)
        ax_pr.grid(axis="x", linestyle="--", alpha=0.3)
        ax_pr.grid(axis="y", linestyle=":",  alpha=0.2)
        ax_pr.set_title(f"P={res.precision*100:.0f}% R={res.recall*100:.0f}% F1={res.f1*100:.0f}%",
                        fontsize=8)
        ax_pr.set_xlabel("Tempo (s)", fontsize=7)
        ax_pr.tick_params(labelsize=6)

        # Legenda só no primeiro
        if idx == 0:
            patch_g = mpatches.Patch(color=cores_gabarito, alpha=0.6, label="Gabarito")
            patch_d = mpatches.Patch(color=cores_detectado, alpha=0.6, label="Detectado")
            ax_pr.legend(handles=[patch_g, patch_d], fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show()


def rodar_suite(
    casos: list[tuple],
    nome_suite: str,
    num_cordas: int = 5,
    plotar: bool = True,
) -> list[ResultadoPolifonico]:
    """
    Executa uma suite completa e imprime o relatório.
    """
    print(f"\n{'═'*72}")
    print(f"  SUITE: {nome_suite}")
    print(f"  {len(casos)} caso(s) de teste  |  {num_cordas} cordas")
    print(f"{'═'*72}")

    resultados      = []
    audios_plot     = []
    segmentos_plot  = []

    for audio, gab, nome in casos:
        print(f"\n  ▶ {nome} ...", end=" ", flush=True)
        segs = rodar_polifonico_em_array(audio, num_cordas=num_cordas)
        res  = calcular_metricas_polifonicas(segs, gab, nome)
        print(f"P={res.precision*100:.0f}%  R={res.recall*100:.0f}%  "
              f"F1={res.f1*100:.0f}%  fantasmas={res.notas_fantasma}")
        imprimir_resultado(res)
        resultados.append(res)
        audios_plot.append(audio)
        segmentos_plot.append(segs)

    # Resumo da suite
    print(f"\n{'─'*72}")
    print(f"  RESUMO — {nome_suite}")
    print(f"{'─'*72}")
    print(f"  Precision média : {np.mean([r.precision for r in resultados])*100:.1f}%")
    print(f"  Recall médio    : {np.mean([r.recall    for r in resultados])*100:.1f}%")
    print(f"  F1 médio        : {np.mean([r.f1        for r in resultados])*100:.1f}%")
    total_fant = sum(r.notas_fantasma for r in resultados)
    total_8va  = sum(r.oitavas_erradas for r in resultados)
    print(f"  Total fantasmas : {total_fant}")
    print(f"  Total 8va erros : {total_8va}")

    if plotar:
        plotar_suite_polifonica(
            nome_suite, casos, resultados, audios_plot, segmentos_plot
        )

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("\n" + "█"*72)
    print("  TESTES DO pYIN POLIFÔNICO")
    print("  Todos os sinais são sintéticos — gabarito 100% conhecido")
    print("█"*72)

    PLOTAR     = True
    NUM_CORDAS = 5

    # Suite 1 — nota única por corda (baseline do filtro)
    r1 = rodar_suite(
        suite_nota_unica_por_corda(),
        "NOTA ÚNICA POR CORDA — baseline do isolamento de filtro",
        num_cordas=NUM_CORDAS, plotar=PLOTAR,
    )

    # Suite 2 — intervalos simultâneos (caso central)
    r2 = rodar_suite(
        suite_intervalos_simultaneos(),
        "INTERVALOS SIMULTÂNEOS — o caso que o monofônico falha",
        num_cordas=NUM_CORDAS, plotar=PLOTAR,
    )

    # Suite 3 — sobreposição parcial (legato/overlap)
    r3 = rodar_suite(
        suite_sobreposicao_parcial(),
        "SOBREPOSIÇÃO PARCIAL — notas com timings desalinhados",
        num_cordas=NUM_CORDAS, plotar=PLOTAR,
    )

    # Suite 4 — erro de oitava com harmônicos ricos
    r4 = rodar_suite(
        suite_erro_oitava_critico(),
        "ERRO DE OITAVA — cordas soltas com perfil slap (caso mais difícil)",
        num_cordas=NUM_CORDAS, plotar=PLOTAR,
    )

    # Suite 5 — silêncios e notas fantasma
    r5 = rodar_suite(
        suite_fantasmas_e_silencio(),
        "FANTASMAS E SILÊNCIO — rejeição de falsos positivos",
        num_cordas=NUM_CORDAS, plotar=PLOTAR,
    )

    # ── Resumo geral ──────────────────────────────────────────────────────────
    print("\n" + "█"*72)
    print("  RESUMO GERAL — todas as suites")
    print("█"*72)
    todos = r1 + r2 + r3 + r4 + r5
    print(f"  Casos totais    : {len(todos)}")
    print(f"  Precision média : {np.mean([r.precision for r in todos])*100:.1f}%")
    print(f"  Recall médio    : {np.mean([r.recall    for r in todos])*100:.1f}%")
    print(f"  F1 médio        : {np.mean([r.f1        for r in todos])*100:.1f}%")
    print(f"  Total fantasmas : {sum(r.notas_fantasma for r in todos)}")
    print(f"  Total 8va erros : {sum(r.oitavas_erradas for r in todos)}")
    print("█"*72 + "\n")
