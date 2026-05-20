"""
MÓDULO 1b — Detecção de Pitch Polifônica (pYIN por corda)  [v2.1 — filtro SOS]
===============================================================================
Resolve o problema de notas simultâneas que o pYIN monofônico não detecta.

Correção v2.1 (bug crítico)
----------------------------
O filtro Butterworth em forma (b, a) é **numericamente instável** para
frequências muito baixas (28–220 Hz) relativas ao sample-rate padrão do
librosa (22 050 Hz). Os coeficientes explodem, o sinal filtrado vira quase
zero e o teste `rms_global < 1e-5` descarta o canal antes do pYIN rodar —
por isso todas as cordas retornavam 0 segmentos em 0.0 s.

Correções aplicadas
-------------------
1. `_filtro_passa_banda` usa `output="sos"` + `sosfiltfilt` em vez de
   `(b, a)` + `filtfilt`. SOS (Second-Order Sections) é a forma numericamente
   estável recomendada pelo SciPy para qualquer filtro IIR.
2. Ordem reduzida 5 → 3: ordens altas em bandas estreitas ampliam a
   instabilidade; ordem 3 já dá atenuação adequada.
3. Sanitização de `f0_clean` usa `np.where(np.isfinite(...), ..., np.nan)`
   em vez de `nan_to_num(nan=np.nan)` (que não fazia nada de útil).

Correções herdadas da v2 (mantidas)
------------------------------------
1. Correção de oitava via `_corrigir_oitava()`.
2. Máscara de energia por frame via `_mascara_energia()`.
3. Limiar de probabilidade voiced mais exigente (VOICED_PROB_THRESHOLD).
4. Bandas mais estritas por corda.
5. Duração mínima aumentada (0.05 → 0.10 s).

Faixas de filtro:
    ┌────────┬──────┬─────────────┬─────────────┐
    │ Corda  │ Nota │  fmin (Hz)  │  fmax (Hz)  │
    ├────────┼──────┼─────────────┼─────────────┤
    │  B (5) │  B0  │   28.0 Hz   │   58.0 Hz   │
    │  E (4) │  E1  │   38.0 Hz   │   82.0 Hz   │
    │  A (3) │  A1  │   51.0 Hz   │  110.0 Hz   │
    │  D (2) │  D2  │   68.0 Hz   │  147.0 Hz   │
    │  G (1) │  G2  │   92.0 Hz   │  220.0 Hz   │
    └────────┴──────┴─────────────┴─────────────┘

Como integrar com o pipeline existente (idêntico à v1/v2):
    from modulo_pyin_polifonico import analisar_baixo_polifonico, bridge_pyin_para_notas_poli
    segmentos = analisar_baixo_polifonico("arquivo.mp3", num_cordas=5, plotar=False)
    sequencia = bridge_pyin_para_notas_poli(segmentos)
    # → Módulos 2, 3, 4 inalterados
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.signal as sig
import time


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DAS CORDAS
# ══════════════════════════════════════════════════════════════════════════════

CORDAS_5 = [
    {"nome": "B",  "nota": "B0",  "fmin": 28.0,  "fmax":  58.0},
    {"nome": "E",  "nota": "E1",  "fmin": 38.0,  "fmax":  82.0},
    {"nome": "A",  "nota": "A1",  "fmin": 51.0,  "fmax": 110.0},
    {"nome": "D",  "nota": "D2",  "fmin": 68.0,  "fmax": 147.0},
    {"nome": "G",  "nota": "G2",  "fmin": 92.0,  "fmax": 220.0},
]

CORDAS_4 = [
    {"nome": "E",  "nota": "E1",  "fmin": 38.0,  "fmax":  82.0},
    {"nome": "A",  "nota": "A1",  "fmin": 51.0,  "fmax": 110.0},
    {"nome": "D",  "nota": "D2",  "fmin": 68.0,  "fmax": 147.0},
    {"nome": "G",  "nota": "G2",  "fmin": 92.0,  "fmax": 220.0},
]

# Limiar de probabilidade voiced
# 0.10 → muito permissivo (padrão librosa)
# 0.35 → equilibrado para baixo em mix (recomendado)
# 0.50 → só aceita pitch com altíssima confiança
VOICED_PROB_THRESHOLD = 0.20

# Fração do RMS de pico do canal para considerar frame "com energia suficiente"
RMS_FRAME_THRESHOLD = 0.05   # 5% do pico do canal


# ══════════════════════════════════════════════════════════════════════════════
# FILTRO PASSA-BANDA  (v2.1 — SOS estável)
# ══════════════════════════════════════════════════════════════════════════════

def _filtro_passa_banda(y: np.ndarray, sr: int,
                         fmin: float, fmax: float,
                         ordem: int = 3) -> np.ndarray:
    """
    Filtro Butterworth passa-banda em forma SOS (Second-Order Sections).

    A forma (b, a) usada anteriormente é numericamente instável para
    frequências muito baixas (28–220 Hz) com sr=22050, fazendo os
    coeficientes explodirem e o sinal filtrado virar quase zero.
    `output='sos'` + `sosfiltfilt` resolve esse problema.

    ordem=3 (reduzida de 5) porque ordens altas em bandas estreitas
    ampliam a instabilidade residual.
    """
    nyq  = sr / 2.0
    low  = max(fmin / nyq, 1e-4)
    high = min(fmax / nyq, 0.999)
    if low >= high:
        return np.zeros_like(y)
    sos = sig.butter(ordem, [low, high], btype="band", output="sos")
    return sig.sosfiltfilt(sos, y).astype(y.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO DE OITAVA
# ══════════════════════════════════════════════════════════════════════════════

def _corrigir_oitava(f0: np.ndarray, y_band: np.ndarray,
                      sr: int, hop_length: int,
                      fmin_corda: float) -> np.ndarray:
    """
    Detecta e corrige erros de oitava (+1 oitava) no f0 detectado.

    Para cada frame com pitch válido, calcula a energia espectral em torno de
    f0/2 (1 oitava abaixo). Se a energia ali for comparável à de f0, assume
    que a fundamental real é f0/2 e corrige.
    """
    f0_corr = f0.copy()

    # Sanitiza infinitos; preserva NaN (frames não vozeados)
    y_band = np.where(np.isfinite(y_band), y_band, 0.0)

    n_fft = 4096
    S     = np.abs(librosa.stft(y_band, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    n_frames = min(len(f0), S.shape[1])

    for i in range(n_frames):
        fi = f0[i]
        if not np.isfinite(fi):
            continue

        f_baixo = fi / 2.0
        if f_baixo < fmin_corda * 0.85:
            continue

        def energia_em(freq):
            idx = np.argmin(np.abs(freqs - freq))
            w   = max(1, int(freq * 0.05 / (sr / n_fft)))
            lo  = max(0, idx - w)
            hi  = min(S.shape[0], idx + w + 1)
            return np.sum(S[lo:hi, i] ** 2)

        e_fi      = energia_em(fi)
        e_f_baixo = energia_em(f_baixo)

        if e_fi > 0 and (e_f_baixo / e_fi) >= 0.30:
            f0_corr[i] = f_baixo

    return f0_corr


# ══════════════════════════════════════════════════════════════════════════════
# MÁSCARA DE ENERGIA POR FRAME
# ══════════════════════════════════════════════════════════════════════════════

def _mascara_energia(y_band: np.ndarray, sr: int,
                      hop_length: int, n_frames: int,
                      threshold: float = RMS_FRAME_THRESHOLD) -> np.ndarray:
    """
    Retorna booleano True para frames com energia acima do limiar relativo.
    """
    frame_length = hop_length * 2
    rms = librosa.feature.rms(
        y=y_band, frame_length=frame_length, hop_length=hop_length
    )[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    else:
        rms = rms[:n_frames]
    pico = np.max(rms)
    if pico < 1e-8:
        return np.zeros(n_frames, dtype=bool)
    return rms >= (pico * threshold)


# ══════════════════════════════════════════════════════════════════════════════
# PYIN POR CORDA
# ══════════════════════════════════════════════════════════════════════════════

def _pyin_uma_corda(y_harmonic: np.ndarray, sr: int,
                    corda: dict,
                    frame_length: int = 4096,
                    hop_length: int = 1024) -> list[dict]:
    """
    Filtra o sinal harmônico na faixa da corda e roda o pYIN com
    pós-processamentos de correção de oitava e filtragem de energia.
    
    Nota: para cordas com banda muito estreita (B0, E1), reduz frame_length
    para evitar erro de dimensão na matriz de transição do pYIN.
    """
    fmin = corda["fmin"]
    fmax = corda["fmax"]
    nome = corda["nome"]
    
    # Reduz frame_length para bandas muito estreitas (evita erro de pYIN)
    freq_range = fmax - fmin
    if freq_range < 60:  # B0 (28-58 Hz)
        frame_length = 1024
        hop_length = 256
    elif freq_range < 100:  # E1, A1
        frame_length = 1024
        hop_length = 512

    # 1. Filtra na banda da corda (SOS — estável para frequências baixas)
    y_band = _filtro_passa_banda(y_harmonic, sr, fmin, fmax)
    # Sanitiza: substitui inf por 0, mantém NaN como NaN
    y_band = np.where(np.isfinite(y_band), y_band, 0.0)

    rms_global = np.sqrt(np.mean(y_band ** 2))
    if rms_global < 1e-5:
        return []

    # 2. pYIN — com tratamento de erro para bandas muito estreitas
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_band,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except librosa.util.exceptions.ParameterError as e:
        # Fallback: usar pyin com amostragem reduzida para bandas estreitas
        # Downsample para reduzir a complexidade da matriz de transição
        if "must be at least" in str(e):
            print(f"    [FallBack] Reduzindo sample rate para {nome}...")
            # Downsample by 2x reduz sr de 22050 para ~11025
            y_band_ds = librosa.resample(y_band, orig_sr=sr, target_sr=sr//2)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_band_ds,
                sr=sr//2,
                frame_length=frame_length//2,
                hop_length=hop_length//2,
            )
        else:
            raise
    times    = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    n_frames = len(f0)

    # 3. Limiar de probabilidade mais exigente
    voiced_flag_estrito = voiced_flag & (voiced_probs >= VOICED_PROB_THRESHOLD)

    f0_clean = f0.copy()
    f0_clean[~voiced_flag_estrito] = np.nan

    # 4. Descarta frames com pouca energia
    mask_energia = _mascara_energia(y_band, sr, hop_length, n_frames)
    f0_clean[~mask_energia] = np.nan

    # 5. Remove detecções fora da banda desta corda
    mascara_banda = (f0_clean >= fmin * 0.88) & (f0_clean <= fmax * 1.05)
    f0_clean[~mascara_banda] = np.nan

    # 6. Correção de oitava
    f0_clean = _corrigir_oitava(f0_clean, y_band, sr, hop_length, fmin)
    # Sanitiza infinitos, preserva NaN
    f0_clean = np.where(np.isfinite(f0_clean), f0_clean, np.nan)

    # Converte e quantiza
    midi_notes     = librosa.hz_to_midi(f0_clean)
    midi_quantized = np.round(midi_notes)

    # 7. Agrupa frames consecutivos da mesma nota em segmentos
    segmentos  = []
    nota_atual = None
    inicio_idx = 0

    for i, nota in enumerate(midi_quantized):
        if np.isnan(nota):
            if nota_atual is not None:
                segmentos.append({
                    "nota_midi": nota_atual,
                    "inicio":    float(times[inicio_idx]),
                    "fim":       float(times[i]),
                    "corda":     nome,
                })
                nota_atual = None
        else:
            if nota_atual is None:
                nota_atual = nota
                inicio_idx = i
            elif nota != nota_atual:
                segmentos.append({
                    "nota_midi": nota_atual,
                    "inicio":    float(times[inicio_idx]),
                    "fim":       float(times[i]),
                    "corda":     nome,
                })
                nota_atual = nota
                inicio_idx = i

    if nota_atual is not None:
        segmentos.append({
            "nota_midi": nota_atual,
            "inicio":    float(times[inicio_idx]),
            "fim":       float(times[-1]),
            "corda":     nome,
        })

    return segmentos


# ══════════════════════════════════════════════════════════════════════════════
# MESCLAGEM DE SEGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def _mesclar_segmentos(todos: list[dict],
                        tolerancia_overlap: float = 0.03,
                        duracao_minima: float = 0.10) -> list[dict]:
    """
    Combina segmentos de todas as cordas, fundindo duplicatas e preservando
    notas simultâneas genuínas.
    """
    filtrados = [s for s in todos
                 if (s["fim"] - s["inicio"]) >= duracao_minima]

    if not filtrados:
        return []

    filtrados.sort(key=lambda s: s["inicio"])

    fundidos: list[dict] = []
    for seg in filtrados:
        fundiu = False
        for f in fundidos:
            mesma_nota  = f["nota_midi"] == seg["nota_midi"]
            overlap_ini = max(f["inicio"], seg["inicio"])
            overlap_fim = min(f["fim"],    seg["fim"])
            tem_overlap = (overlap_fim - overlap_ini) > tolerancia_overlap
            if mesma_nota and tem_overlap:
                f["inicio"] = min(f["inicio"], seg["inicio"])
                f["fim"]    = max(f["fim"],    seg["fim"])
                fundiu = True
                break
        if not fundiu:
            fundidos.append(dict(seg))

    fundidos.sort(key=lambda s: (s["inicio"], s["nota_midi"]))
    return fundidos


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def analisar_baixo_polifonico(
    audio_path: str,
    num_cordas: int = 5,
    plotar: bool = True,
    duracao_minima: float = 0.10,
    tolerancia_overlap: float = 0.03,
) -> list[dict]:
    """
    Versão polifônica de analisar_baixo() com correção de oitava e filtragem
    de energia por frame.

    Parâmetros
    ----------
    audio_path          : caminho do arquivo de áudio
    num_cordas          : 4 ou 5
    plotar              : exibe gráfico multi-canal
    duracao_minima      : descarta notas mais curtas que este valor (s)
    tolerancia_overlap  : limiar para fundir duplicatas (s)
    """
    print(f"\n--- [POLIFÔNICO v2.1] Processando: {audio_path.split('/')[-1]} ---")

    y, sr = librosa.load(audio_path)
    frame_length = 4096
    hop_length   = 1024

    print("Separando componentes harmônicos (margin=3.0)...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)

    cordas = CORDAS_5 if num_cordas == 5 else CORDAS_4
    todos_segmentos = []

    for corda in cordas:
        print(f"  [pYIN] Corda {corda['nome']} ({corda['nota']}) "
              f"[{corda['fmin']:.0f}–{corda['fmax']:.0f} Hz]...", end=" ", flush=True)
        t0   = time.time()
        segs = _pyin_uma_corda(y_harmonic, sr, corda, frame_length, hop_length)
        print(f"{len(segs)} segmentos  ({time.time()-t0:.1f}s)")
        todos_segmentos.extend(segs)

    segmentos    = _mesclar_segmentos(todos_segmentos, tolerancia_overlap, duracao_minima)
    simultaneas  = _contar_simultaneas(segmentos)
    print(f"\n[✓] {len(segmentos)} segmentos  ({simultaneas} pares simultâneos)")

    if plotar:
        _plotar_polifonico(audio_path, y, y_harmonic, sr, hop_length,
                           cordas, segmentos)

    return segmentos


def _contar_simultaneas(segmentos: list[dict]) -> int:
    count = 0
    for i in range(len(segmentos)):
        for j in range(i + 1, len(segmentos)):
            a, b = segmentos[i], segmentos[j]
            if b["inicio"] >= a["fim"]:
                break
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def bridge_pyin_para_notas_poli(
    segmentos: list[dict],
    duracao_minima: float = 0.10,
) -> list[dict]:
    """
    Drop-in replacement de bridge_pyin_para_notas().
    Preserva campo "corda" e filtra segmentos curtos.
    """
    resultado = []
    for seg in segmentos:
        if (seg["fim"] - seg["inicio"]) < duracao_minima:
            continue
        resultado.append({
            "nome":   librosa.midi_to_note(int(seg["nota_midi"])),
            "inicio": seg["inicio"],
            "fim":    seg["fim"],
            "corda":  seg.get("corda", "?"),
        })
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _plotar_polifonico(audio_path, y, y_harmonic, sr, hop_length,
                        cordas, segmentos):
    n_cordas = len(cordas)
    n_rows   = 2 + n_cordas
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 3 * n_rows), sharex=True,
        gridspec_kw={"height_ratios": [2.5] + [1.2] * n_cordas + [2.0]},
    )
    fig.suptitle(
        f"Transcrição Polifônica (v2.1): {audio_path.split('/')[-1]}",
        fontsize=14, fontweight="bold",
    )

    fmin_global = librosa.note_to_hz("B0")
    fmax_global = 220.0

    ax0 = axes[0]
    C    = np.abs(librosa.cqt(y_harmonic, sr=sr, fmin=fmin_global,
                               n_bins=48, bins_per_octave=12,
                               hop_length=hop_length))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    librosa.display.specshow(C_db, sr=sr, x_axis="time", y_axis="cqt_hz",
                              fmin=fmin_global, bins_per_octave=12,
                              hop_length=hop_length, cmap="magma",
                              alpha=0.8, ax=ax0)
    ax0.set_title("Espectrograma CQT — região do baixo", fontsize=10)
    ax0.set_ylabel("Frequência (Hz)")
    ax0.set_ylim(fmin_global, fmax_global)

    cores_cordas = {c["nome"]: cor for c, cor in
                    zip(cordas, ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"])}

    for idx_c, corda in enumerate(cordas):
        ax  = axes[1 + idx_c]
        cor = cores_cordas[corda["nome"]]
        y_band = _filtro_passa_banda(y_harmonic, sr, corda["fmin"], corda["fmax"])
        t_wave = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(y_band))
        pico   = np.max(np.abs(y_band)) or 1.0
        ax.plot(t_wave, y_band / pico * 0.8, color=cor, linewidth=0.5, alpha=0.5)
        ax.set_facecolor("#1a1a2e")
        ax.set_ylabel(f"{corda['nome']}\n({corda['nota']})", fontsize=8, color=cor)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(labelsize=6)
        segs_corda = [s for s in segmentos if s.get("corda") == corda["nome"]]
        for s in segs_corda:
            ax.axvspan(s["inicio"], s["fim"], alpha=0.30, color=cor)
            mid = (s["inicio"] + s["fim"]) / 2
            nome_nota = librosa.midi_to_note(int(s["nota_midi"]))
            ax.text(mid, 0.5, nome_nota, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                    transform=ax.get_xaxis_transform())

    ax_pr = axes[-1]
    notas_usadas = set()
    for seg in segmentos:
        cor    = cores_cordas.get(seg.get("corda", "?"), "#aaaaaa")
        nome   = librosa.midi_to_note(int(seg["nota_midi"]))
        dur    = seg["fim"] - seg["inicio"]
        midi_n = seg["nota_midi"]
        notas_usadas.add(midi_n)
        if dur