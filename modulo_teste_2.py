"""
MÓDULO DE TESTE — Ambiente Controlado para Validação do Pipeline
================================================================
Execução standalone (não faz parte do pipeline principal):

    python modulo_teste.py

Estrutura:
    1. GERADORES DE SINAL     — constroem áudio sintético com gabarito conhecido
    2. PERTURBADORES          — aplicam ruído/deformações controláveis
    3. CASOS DE TESTE         — cenários predefinidos (nota única, sequência, stress)
    4. MÉTRICAS               — comparam saída do pYIN com o gabarito
    5. RELATÓRIO              — imprime e plota os resultados

Nenhum arquivo de áudio externo é necessário.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Ruídos de Lee & Shen (2011) adaptados para áudio musical
from modulo_ruidos import suite_ruidos_musicais, listar_ruidos, RuidoMusical


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

SR         = 22050       # sample rate padrão (mesmo do librosa.load)
TOLERANCIA_CENTS = 50    # ±50 cents = ±meio semitom → aceitável como "correto"
TOLERANCIA_MIDI  = 0.5   # equivalente em MIDI


# ══════════════════════════════════════════════════════════════════════════════
# 1. GERADORES DE SINAL
# ══════════════════════════════════════════════════════════════════════════════

def sinal_puro(freq_hz: float, duracao: float, sr: int = SR,
               amplitude: float = 0.8) -> np.ndarray:
    """
    Onda senoidal pura — o caso mais favorável possível para o pYIN.
    Representa uma nota sustentada ideal, sem decaimento ou harmônicos.

    Parâmetros
    ----------
    freq_hz   : frequência fundamental em Hz
    duracao   : duração em segundos
    amplitude : amplitude de pico (0.0 – 1.0)
    """
    t = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)

def soma_sinais(freq_hz_1: float, freq_hz_2: float, duracao: float, sr: int = SR,
                amplitude: float = 0.8) -> np.ndarray:
    """
    Soma de duas ondas senoidais — simula a situação de duas notas soando
    simultaneamente (interferência harmônica).

    Útil para testar a robustez do pYIN em separar fontes.
    """
    t = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    sinal1 = amplitude * np.sin(2 * np.pi * freq_hz_1 * t)
    sinal2 = amplitude * np.sin(2 * np.pi * freq_hz_2 * t)
    return (sinal1 + sinal2) / 2  # normaliza para evitar clipping

def sinal_decay(freq_hz: float, duracao: float, sr: int = SR,
                amplitude: float = 0.8, tau: float = 0.4) -> np.ndarray:
    """
    Senoidal com envelope exponencial decrescente — simula corda dedilhada.
    O parâmetro tau controla a velocidade do decaimento:
        tau pequeno (~0.2) → nota seca, estilo pizzicato
        tau grande  (~1.5) → nota longa, sustentada
    """
    t       = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    envelope = np.exp(-t / tau)
    return amplitude * envelope * np.sin(2 * np.pi * freq_hz * t)


def sinal_harmonicos(freq_hz: float, duracao: float, sr: int = SR,
                     amplitude: float = 0.8,
                     perfil: str = "baixo") -> np.ndarray:
    """
    Fundamental + harmônicos com amplitudes relativas realistas.
    Essa é a forma de onda mais próxima de um instrumento real.

    Perfis disponíveis:
        "baixo"    → forte no fundamental, queda rápida nos harmônicos
        "baixo_sl" → slap bass: harmônicos mais proeminentes
        "puro"     → apenas fundamental (equivale a sinal_puro)

    A amplitude de cada harmônico h é: A / h^expoente
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
        amp_h = amplitude / (h ** cfg["expoente"])
        y    += amp_h * np.sin(2 * np.pi * freq_hz * h * t)
    # Normaliza para o pico não ultrapassar amplitude
    pico = np.max(np.abs(y))
    if pico > 0:
        y = y / pico * amplitude
    return y


def sinal_harmonicos_decay(freq_hz: float, duracao: float, sr: int = SR,
                            amplitude: float = 0.8, tau: float = 0.5,
                            perfil: str = "baixo") -> np.ndarray:
    """
    Combinação: harmônicos reais + envelope de decaimento.
    O caso mais próximo de um baixo real tocando uma nota solta.
    """
    t        = np.linspace(0, duracao, int(sr * duracao), endpoint=False)
    envelope = np.exp(-t / tau)
    base     = sinal_harmonicos(freq_hz, duracao, sr, amplitude=1.0, perfil=perfil)
    y        = base * envelope * amplitude
    pico = np.max(np.abs(y))
    if pico > 0:
        y = y / pico * amplitude
    return y


# ══════════════════════════════════════════════════════════════════════════════
# 2. PERTURBADORES
# ══════════════════════════════════════════════════════════════════════════════

def adicionar_ruido_branco(sinal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adiciona ruído gaussiano branco com SNR alvo em dB.
    SNR = 20*log10(P_sinal / P_ruido)

    snr_db > 30  → ruído quase imperceptível
    snr_db = 20  → ruído leve
    snr_db = 10  → ruído moderado (começa a prejudicar detecção)
    snr_db =  0  → sinal e ruído de mesma potência (detecção muito difícil)
    snr_db < 0   → ruído domina o sinal
    """
    potencia_sinal = np.mean(sinal ** 2)
    if potencia_sinal == 0:
        return sinal.copy()
    potencia_ruido_alvo = potencia_sinal / (10 ** (snr_db / 10))
    ruido = np.random.normal(0, np.sqrt(potencia_ruido_alvo), len(sinal))
    return sinal + ruido


def adicionar_interferencia_harmonica(sinal: np.ndarray,
                                       freq_interferente: float,
                                       amplitude_rel: float = 0.3,
                                       sr: int = SR) -> np.ndarray:
    """
    Adiciona um segundo instrumento (guitarra, piano) soando simultaneamente.
    Simula a situação real de separação de fonte deficiente.

    amplitude_rel : amplitude da interferência relativa ao pico do sinal original
                    0.1 → interferência fraca
                    0.5 → interferência forte (caso crítico)
                    1.0 → interferência igual ao sinal
    """
    duracao       = len(sinal) / sr
    interferencia = sinal_harmonicos(freq_interferente, duracao, sr,
                                     amplitude=amplitude_rel, perfil="baixo_sl")
    return sinal + interferencia[:len(sinal)]


def adicionar_silencio(sinal: np.ndarray, posicao: str = "inicio",
                        duracao_s: float = 0.2, sr: int = SR) -> np.ndarray:
    """
    Insere silêncio no início, fim ou ambos — testa robustez do voiced_flag.
    """
    silencio = np.zeros(int(sr * duracao_s))
    if posicao == "inicio":
        return np.concatenate([silencio, sinal])
    elif posicao == "fim":
        return np.concatenate([sinal, silencio])
    else:  # ambos
        return np.concatenate([silencio, sinal, silencio])


def construir_sequencia(notas_hz: list[float], duracao_nota: float = 0.5,
                         silencio_entre: float = 0.05,
                         tipo: str = "harmonicos_decay",
                         sr: int = SR, **kwargs) -> tuple[np.ndarray, list[dict]]:
    """
    Constrói um áudio de múltiplas notas em sequência com gabarito temporal.

    Retorna
    -------
    (audio, gabarito)
    gabarito: list[dict] com "freq_hz", "nota_nome", "inicio", "fim"
    """
    geradores = {
        "puro":              sinal_puro,
        "decay":             sinal_decay,
        "harmonicos":        sinal_harmonicos,
        "harmonicos_decay":  sinal_harmonicos_decay,
    }
    fn      = geradores.get(tipo, sinal_harmonicos_decay)
    audio   = np.array([], dtype=np.float64)
    gabarito = []
    t_atual  = 0.0

    for freq in notas_hz:
        segmento = fn(freq, duracao_nota, sr, **kwargs)
        if silencio_entre > 0:
            gap = np.zeros(int(sr * silencio_entre))
        else:
            gap = np.array([])

        gabarito.append({
            "freq_hz":   freq,
            "nota_nome": librosa.hz_to_note(freq, octave=True),
            "inicio":    t_atual,
            "fim":       t_atual + duracao_nota,
        })

        audio   = np.concatenate([audio, segmento, gap])
        t_atual += duracao_nota + silencio_entre

    return audio.astype(np.float32), gabarito


# ══════════════════════════════════════════════════════════════════════════════
# 3. RUNNER DO pYIN (interface limpa para os testes)
# ══════════════════════════════════════════════════════════════════════════════

def rodar_pyin(audio: np.ndarray, sr: int = SR,
               fmin_nota: str = "B0", fmax_nota: str = "G4") -> dict:
    """
    Roda o pYIN diretamente em um array numpy (sem arquivo).

    Retorna dict com:
        "f0"           → array bruto de frequências
        "f0_clean"     → f0 com NaN nos frames não-vozeados
        "voiced_flag"  → array booleano
        "voiced_probs" → probabilidades
        "times"        → timestamps de cada frame
        "midi_raw"     → f0_clean em MIDI (com NaN)
        "midi_quant"   → MIDI arredondado (nota mais próxima)
    """
    fmin = librosa.note_to_hz(fmin_nota)
    fmax = librosa.note_to_hz(fmax_nota)

    y_harm, _ = librosa.effects.hpss(audio, margin=2.0)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harm,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=4096,
        hop_length=1024,
    )

    f0_clean       = f0.copy()
    f0_clean[~voiced_flag] = np.nan
    times          = librosa.times_like(f0, sr=sr, hop_length=1024)
    midi_raw       = librosa.hz_to_midi(f0_clean)
    midi_quant     = np.round(midi_raw)

    return {
        "f0":           f0,
        "f0_clean":     f0_clean,
        "voiced_flag":  voiced_flag,
        "voiced_probs": voiced_probs,
        "times":        times,
        "midi_raw":     midi_raw,
        "midi_quant":   midi_quant,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResultadoTeste:
    nome:              str
    nota_esperada_hz:  float
    nota_esperada_nome: str
    # Métricas de detecção
    taxa_deteccao:     float   # % frames voiced quando deveria haver nota
    taxa_silencio_ok:  float   # % frames silenciosos corretamente ignorados
    # Métricas de acurácia
    acuracia_nota:     float   # % frames voiced com nota correta (±50 cents)
    erro_oitava_pct:   float   # % frames com erro exatamente de 1 oitava
    mediana_cents:     float   # erro mediano em cents
    max_cents:         float   # pior erro em cents
    # Diagnóstico textual
    diagnostico:       list[str] = field(default_factory=list)


def calcular_metricas(resultado_pyin: dict,
                       gabarito: list[dict],
                       nome_teste: str,
                       sr: int = SR) -> ResultadoTeste:
    """
    Compara a saída do pYIN com o gabarito de uma sequência.

    Para gabarito de nota única, passe uma lista com um elemento.
    """
    times      = resultado_pyin["times"]
    midi_quant = resultado_pyin["midi_quant"]
    voiced     = resultado_pyin["voiced_flag"]

    # ── Agrega notas esperadas por frame ─────────────────────────────────────
    midi_esperado = np.full(len(times), np.nan)  # NaN = silêncio esperado

    for seg in gabarito:
        midi_alvo = librosa.hz_to_midi(seg["freq_hz"])
        for i, t in enumerate(times):
            if seg["inicio"] <= t <= seg["fim"]:
                midi_esperado[i] = midi_alvo

    frames_nota     = ~np.isnan(midi_esperado)  # onde deveria haver nota
    frames_silencio = np.isnan(midi_esperado)   # onde deveria haver silêncio

    # ── Taxa de detecção ─────────────────────────────────────────────────────
    if np.sum(frames_nota) > 0:
        taxa_deteccao = np.mean(voiced[frames_nota]) * 100
    else:
        taxa_deteccao = 0.0

    # ── Taxa de silêncio correto ──────────────────────────────────────────────
    if np.sum(frames_silencio) > 0:
        taxa_silencio_ok = np.mean(~voiced[frames_silencio]) * 100
    else:
        taxa_silencio_ok = 100.0

    # ── Acurácia de pitch nos frames voiced com nota esperada ─────────────────
    frames_validos = frames_nota & voiced & ~np.isnan(midi_quant)
    erros_cents    = np.array([])
    erro_oitava    = np.array([], dtype=bool)

    if np.sum(frames_validos) > 0:
        detectado = midi_quant[frames_validos]
        esperado  = midi_esperado[frames_validos]
        # Converte erro MIDI → cents (1 semitom = 100 cents)
        erros_midi  = detectado - esperado
        erros_cents = erros_midi * 100
        # Erro de oitava: erro próximo de ±1200 cents (±12 semitons)
        erro_oitava = np.abs(np.abs(erros_midi) - 12) < 1.5
        acuracia_nota = np.mean(np.abs(erros_cents) <= TOLERANCIA_CENTS) * 100
    else:
        acuracia_nota = 0.0

    mediana_cents = float(np.median(np.abs(erros_cents))) if len(erros_cents) > 0 else 0.0
    max_cents     = float(np.max(np.abs(erros_cents)))    if len(erros_cents) > 0 else 0.0
    erro_oitava_pct = float(np.mean(erro_oitava) * 100)  if len(erro_oitava) > 0 else 0.0

    # ── Diagnóstico automático ────────────────────────────────────────────────
    nota_ref_hz   = gabarito[0]["freq_hz"]
    nota_ref_nome = gabarito[0]["nota_nome"]
    diagnosticos  = []

    if taxa_deteccao < 50:
        diagnosticos.append("⚠ DETECÇÃO BAIXA: pYIN não encontrou nota na maioria dos frames")
    if erro_oitava_pct > 20:
        diagnosticos.append(f"⚠ ERRO DE OITAVA: {erro_oitava_pct:.0f}% dos frames com oitava errada")
    if mediana_cents > TOLERANCIA_CENTS:
        diagnosticos.append(f"⚠ PITCH IMPRECISO: erro mediano {mediana_cents:.0f} cents")
    if taxa_silencio_ok < 80:
        diagnosticos.append("⚠ FALSO POSITIVO: detectando notas em silêncio")
    if not diagnosticos:
        diagnosticos.append("✓ Sem problemas críticos detectados")

    return ResultadoTeste(
        nome               = nome_teste,
        nota_esperada_hz   = nota_ref_hz,
        nota_esperada_nome = nota_ref_nome,
        taxa_deteccao      = taxa_deteccao,
        taxa_silencio_ok   = taxa_silencio_ok,
        acuracia_nota      = acuracia_nota,
        erro_oitava_pct    = erro_oitava_pct,
        mediana_cents      = mediana_cents,
        max_cents          = max_cents,
        diagnostico        = diagnosticos,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. CASOS DE TESTE
# ══════════════════════════════════════════════════════════════════════════════

# Notas de referência do baixo (usadas nos testes)
NOTAS_REFERENCIA = {
    "B0":  librosa.note_to_hz("B0"),    # corda solta mais grave
    "E1":  librosa.note_to_hz("E1"),    # corda solta
    "A1":  librosa.note_to_hz("A1"),    # corda solta
    "D2":  librosa.note_to_hz("D2"),    # corda solta
    "G2":  librosa.note_to_hz("G2"),    # corda solta mais aguda
    "A2":  librosa.note_to_hz("A2"),    # candidata a erro de oitava
    "B2":  librosa.note_to_hz("B2"),    # candidata a erro de oitava
    "F#2": librosa.note_to_hz("F#2"),
}

# Riff simplificado de Money (B1–D2–F#2–A2 transposto para as cordas soltas)
RIFF_MONEY_HZ = [
    librosa.note_to_hz(n) for n in ["A1", "A1", "D2", "A1", "D2", "F#2", "A2"]
]

# Riff pentatônico genérico — bom caso de teste de transição
RIFF_PENTATONICO_HZ = [
    librosa.note_to_hz(n) for n in ["A1", "C2", "D2", "E2", "G2"]
]


def suite_nota_unica() -> list[tuple[np.ndarray, list[dict], str]]:
    """
    Gera casos de teste para nota única sob diferentes condições.
    Retorna lista de (audio, gabarito, nome_do_teste).
    """
    casos = []
    freq  = NOTAS_REFERENCIA["A2"]   # A2: candidata a erro de oitava
    dur   = 1.5

    # ── Caso 1: Onda pura — baseline ideal ───────────────────────────────────
    audio, gab = construir_sequencia([freq], duracao_nota=dur, tipo="puro")
    casos.append((audio, gab, "A2 | Onda pura (baseline ideal)"))

    # ── Caso 2: Decay natural ─────────────────────────────────────────────────
    audio, gab = construir_sequencia([freq], duracao_nota=dur, tipo="decay", tau=0.5)
    casos.append((audio, gab, "A2 | Decay natural (tau=0.5s)"))

    # ── Caso 3: Decay rápido (pizzicato seco) ─────────────────────────────────
    audio, gab = construir_sequencia([freq], duracao_nota=dur, tipo="decay", tau=0.15)
    casos.append((audio, gab, "A2 | Decay rápido (tau=0.15s — pizzicato)"))

    # ── Caso 4: Harmônicos reais de baixo ────────────────────────────────────
    audio, gab = construir_sequencia([freq], duracao_nota=dur,
                                      tipo="harmonicos", perfil="baixo")
    casos.append((audio, gab, "A2 | Harmônicos de baixo (sem decay)"))

    # ── Caso 5: Harmônicos + decay (mais realista) ────────────────────────────
    audio, gab = construir_sequencia([freq], duracao_nota=dur,
                                      tipo="harmonicos_decay", perfil="baixo", tau=0.5)
    casos.append((audio, gab, "A2 | Harmônicos + decay (caso mais realista)"))

    # ── Casos 6–10: SNR progressivo com sinal realista ───────────────────────
    base, gab = construir_sequencia([freq], duracao_nota=dur,
                                     tipo="harmonicos_decay", perfil="baixo", tau=0.5)
    for snr in [40, 20, 10, 5, 0]:
        ruidoso = adicionar_ruido_branco(base, snr_db=snr)
        casos.append((ruidoso, gab, f"A2 | Harmônicos + decay + ruído branco SNR={snr}dB"))

    # ── Caso 11: Interferência de instrumento externo (guitarra em E2) ────────
    base, gab = construir_sequencia([freq], duracao_nota=dur,
                                     tipo="harmonicos_decay", perfil="baixo", tau=0.5)
    com_interf = adicionar_interferencia_harmonica(
        base, librosa.note_to_hz("E4"), amplitude_rel=0.4
    )
    casos.append((com_interf, gab, "A2 | + interferência guitarra (E4, amp=0.4)"))

    # ── Caso 12: Silêncio antes e depois ─────────────────────────────────────
    base, gab = construir_sequencia([freq], duracao_nota=dur,
                                     tipo="harmonicos_decay", perfil="baixo", tau=0.5)
    com_sil = adicionar_silencio(base, posicao="ambos", duracao_s=0.3)
    # Ajusta gabarito para o offset de silêncio
    gab_adj = [{"freq_hz": g["freq_hz"], "nota_nome": g["nota_nome"],
                 "inicio": g["inicio"] + 0.3, "fim": g["fim"] + 0.3} for g in gab]
    casos.append((com_sil, gab_adj, "A2 | Com 300ms de silêncio nas bordas"))

    return casos


def suite_erro_oitava() -> list[tuple[np.ndarray, list[dict], str]]:
    """
    Testa notas conhecidas por causar erro de oitava no pYIN.
    Inclui notas em cada oitava do range do baixo.
    """
    casos = []
    notas_teste = ["B0", "E1", "A1", "D2", "G2", "A2", "B2"]

    for nota_nome in notas_teste:
        freq  = librosa.note_to_hz(nota_nome)
        audio, gab = construir_sequencia(
            [freq], duracao_nota=1.0,
            tipo="harmonicos_decay", perfil="baixo", tau=0.5
        )
        casos.append((audio, gab, f"{nota_nome} | Teste de oitava (harmônicos+decay)"))

    return casos


def suite_ruidos_lee_shen(nota_hz: float = 110.0,
                           nota_nome: str = "A2",
                           duracao: float = 1.5) -> list[tuple]:
    """
    Suite de testes com todos os 16 modelos de perturbação de Lee & Shen (2011),
    adaptados para o domínio musical.

    Cada caso usa os parâmetros gerados aleatoriamente (seed fixo para
    reprodutibilidade). O gabarito é sempre a nota fundamental sem perturbação,
    permitindo medir exatamente o quanto cada perturbação degrada a detecção.
    """
    return suite_ruidos_musicais(
        nota_hz=nota_hz, nota_nome=nota_nome, duracao=duracao, seed=42
    )


def suite_sequencia_riff() -> list[tuple[np.ndarray, list[dict], str]]:
    """
    Testa a detecção em sequências de notas (riffs).
    Verifica se o pYIN rastreia corretamente as transições.
    """
    casos = []

    # Riff Money — notas do transcrito original
    audio, gab = construir_sequencia(
        RIFF_MONEY_HZ, duracao_nota=0.4, silencio_entre=0.05,
        tipo="harmonicos_decay", perfil="baixo", tau=0.4
    )
    casos.append((audio, gab, "Riff Money | harmonicos_decay"))

    # Mesmo riff com ruído moderado
    audio_r = adicionar_ruido_branco(audio, snr_db=15)
    casos.append((audio_r, gab, "Riff Money | + ruído branco SNR=15dB"))

    # Riff pentatônico (notas mais agudas)
    audio, gab = construir_sequencia(
        RIFF_PENTATONICO_HZ, duracao_nota=0.4, silencio_entre=0.05,
        tipo="harmonicos_decay", perfil="baixo", tau=0.4
    )
    casos.append((audio, gab, "Riff Pentatônico | harmonicos_decay"))

    # Slap bass (decay rápido + perfil com mais harmônicos)
    audio, gab = construir_sequencia(
        RIFF_PENTATONICO_HZ, duracao_nota=0.3, silencio_entre=0.08,
        tipo="harmonicos_decay", perfil="baixo_sl", tau=0.12
    )
    casos.append((audio, gab, "Riff Pentatônico | slap (decay rápido + harmônicos ricos)"))

    return casos


# ══════════════════════════════════════════════════════════════════════════════
# 6. RELATÓRIO E VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_resultado(r: ResultadoTeste) -> None:
    largura = 70
    print(f"\n{'─'*largura}")
    print(f"  {r.nome}")
    print(f"{'─'*largura}")
    print(f"  Nota alvo        : {r.nota_esperada_nome}  ({r.nota_esperada_hz:.2f} Hz)")
    print(f"  Taxa de detecção : {r.taxa_deteccao:.1f}%   (frames voiced quando deveria ter nota)")
    print(f"  Silêncio correto : {r.taxa_silencio_ok:.1f}%   (frames silenciosos corretamente ignorados)")
    print(f"  Acurácia nota    : {r.acuracia_nota:.1f}%   (dentro de ±{TOLERANCIA_CENTS} cents)")
    print(f"  Erro de oitava   : {r.erro_oitava_pct:.1f}%   (frames com erro de exatamente ±1 oitava)")
    print(f"  Erro mediano     : {r.mediana_cents:.0f} cents")
    print(f"  Pior erro        : {r.max_cents:.0f} cents")
    print(f"\n  Diagnóstico:")
    for d in r.diagnostico:
        print(f"    {d}")


def plotar_suite(resultados: list[ResultadoTeste],
                  titulo_suite: str,
                  audios: list[np.ndarray],
                  pyin_outs: list[dict],
                  gabaritos: list[list[dict]]) -> None:
    """
    Plota painel de resultados para uma suite de testes.
    Linha superior: gráfico de barras das métricas.
    Linhas inferiores: waveform + f0 detectado vs esperado para cada caso.
    """
    n = len(resultados)
    if n == 0:
        return

    # ── Painel de métricas ────────────────────────────────────────────────────
    fig_m, axes_m = plt.subplots(1, 4, figsize=(16, max(3, n * 0.5 + 2)))
    fig_m.suptitle(f"Métricas — {titulo_suite}", fontsize=13, fontweight="bold")

    nomes        = [r.nome.split("|")[-1].strip() for r in resultados]
    det          = [r.taxa_deteccao    for r in resultados]
    acc          = [r.acuracia_nota    for r in resultados]
    oct_err      = [r.erro_oitava_pct  for r in resultados]
    med_cents    = [r.mediana_cents    for r in resultados]

    cores_det = ["#2ecc71" if v >= 80 else "#e67e22" if v >= 50 else "#e74c3c" for v in det]
    cores_acc = ["#2ecc71" if v >= 80 else "#e67e22" if v >= 50 else "#e74c3c" for v in acc]
    cores_oct = ["#2ecc71" if v <= 10 else "#e67e22" if v <= 30 else "#e74c3c" for v in oct_err]
    cores_med = ["#2ecc71" if v <= 50 else "#e67e22" if v <= 150 else "#e74c3c" for v in med_cents]

    y_pos = range(n)

    for ax, vals, cores, titulo, fmt, lim in [
        (axes_m[0], det,       cores_det, "Taxa Detecção (%)",   "{:.0f}%",  [0, 100]),
        (axes_m[1], acc,       cores_acc, "Acurácia Nota (%)",   "{:.0f}%",  [0, 100]),
        (axes_m[2], oct_err,   cores_oct, "Erro de Oitava (%)",  "{:.0f}%",  [0, 100]),
        (axes_m[3], med_cents, cores_med, "Erro Mediano (cents)", "{:.0f}c", [0, None]),
    ]:
        bars = ax.barh(y_pos, vals, color=cores, edgecolor="white", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nomes, fontsize=8)
        ax.set_title(titulo, fontsize=9)
        if lim[1]:
            ax.set_xlim(lim)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", fontsize=7)

    plt.tight_layout()
    plt.show()

    # ── Detalhe: waveform + pitch para cada caso (máx 6 para não poluir) ─────
    n_plot = min(n, 6)
    fig_d, axes_d = plt.subplots(n_plot, 2, figsize=(14, n_plot * 2.5))
    fig_d.suptitle(f"Detalhe Pitch — {titulo_suite}", fontsize=12, fontweight="bold")

    if n_plot == 1:
        axes_d = [axes_d]   # garante iterabilidade

    for idx in range(n_plot):
        ax_w, ax_p = axes_d[idx]
        audio   = audios[idx]
        out     = pyin_outs[idx]
        gab     = gabaritos[idx]
        res     = resultados[idx]

        t_wave = np.linspace(0, len(audio) / SR, len(audio))
        ax_w.plot(t_wave, audio, color="#3498db", linewidth=0.5, alpha=0.8)
        ax_w.set_ylabel("Amplitude", fontsize=7)
        ax_w.set_title(res.nome.split("|")[-1].strip(), fontsize=8)
        ax_w.tick_params(labelsize=7)

        # Pitch detectado
        ax_p.plot(out["times"], out["f0_clean"], color="#e74c3c",
                  linewidth=2, label="pYIN detectado", zorder=3)

        # Pitch esperado (gabarito)
        for seg in gab:
            ax_p.hlines(seg["freq_hz"], seg["inicio"], seg["fim"],
                         colors="#2ecc71", linewidths=2.5,
                         linestyles="--", label="Esperado", zorder=2)

        ax_p.set_yscale("log")
        ax_p.set_ylabel("Hz (log)", fontsize=7)
        ax_p.tick_params(labelsize=7)
        if idx == 0:
            ax_p.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 7. RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def rodar_suite(casos: list[tuple], titulo: str,
                plotar: bool = True) -> list[ResultadoTeste]:
    """
    Executa uma suite completa de testes e imprime o relatório.
    """
    print(f"\n{'═'*70}")
    print(f"  SUITE: {titulo}")
    print(f"  {len(casos)} caso(s) de teste")
    print(f"{'═'*70}")

    resultados = []
    audios     = []
    pyin_outs  = []
    gabaritos  = []

    for audio, gab, nome in casos:
        print(f"\n  Rodando: {nome} ...", end=" ", flush=True)
        out = rodar_pyin(audio)
        res = calcular_metricas(out, gab, nome)
        print(f"detecção={res.taxa_deteccao:.0f}%  acurácia={res.acuracia_nota:.0f}%  "
              f"oitava_err={res.erro_oitava_pct:.0f}%")
        imprimir_resultado(res)
        resultados.append(res)
        audios.append(audio)
        pyin_outs.append(out)
        gabaritos.append(gab)

    # Resumo da suite
    print(f"\n{'─'*70}")
    print(f"  RESUMO — {titulo}")
    print(f"{'─'*70}")
    print(f"  Detecção média   : {np.mean([r.taxa_deteccao   for r in resultados]):.1f}%")
    print(f"  Acurácia média   : {np.mean([r.acuracia_nota   for r in resultados]):.1f}%")
    print(f"  Erro oitava médio: {np.mean([r.erro_oitava_pct for r in resultados]):.1f}%")
    print(f"  Erro mediano méd.: {np.mean([r.mediana_cents   for r in resultados]):.0f} cents")
    print()

    if plotar:
        plotar_suite(resultados, titulo, audios, pyin_outs, gabaritos)

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  AMBIENTE DE TESTES — Pipeline de Transcrição de Baixo")
    print("  Todas as notas são geradas sinteticamente (gabarito 100% conhecido)")
    print("█"*70)

    PLOTAR = True  # mude para False para rodar sem gráficos (mais rápido)

    # Suite 1: Nota única sob condições progressivamente piores
    r1 = rodar_suite(suite_nota_unica(),    "NOTA ÚNICA — A2 (várias condições)", plotar=PLOTAR)

    # Suite 2: Stress test de erro de oitava em todas as notas do baixo
    r2 = rodar_suite(suite_erro_oitava(),   "ERRO DE OITAVA — range completo do baixo", plotar=PLOTAR)

    # Suite 3: Sequências de notas (riffs)
    r3 = rodar_suite(suite_sequencia_riff(), "SEQUÊNCIAS — riffs sintéticos", plotar=PLOTAR)

    # Suite 4: Perturbações de Lee & Shen (2011) adaptadas para áudio
    print("\n")
    listar_ruidos()
    r4 = rodar_suite(suite_ruidos_lee_shen(nota_hz=110.0, nota_nome="A2", duracao=1.5),
                     "RUÍDOS LEE & SHEN (2011) — todos os 16 modelos sobre A2",
                     plotar=PLOTAR)

    print("\n" + "█"*70)
    print("  Todos os testes concluídos.")
    print("█"*70 + "\n")
