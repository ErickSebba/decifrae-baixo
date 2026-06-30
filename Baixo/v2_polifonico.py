"""
baixo_tab.py — Pipeline Monolítico: Áudio → Tablatura de Baixo
===============================================================
Fluxo completo em um único arquivo:

    Áudio
      ↓  [Módulo 1b] pYIN por corda (polifônico)
    segmentos_raw
      ↓  [Módulo 1.5] limpeza de segmentos
    segmentos
      ↓  bridge → sequencia (nome, inicio, fim, corda)
      ↓  [Módulo 2] posições no braço
    posicoes_dict
      ↓  bridge → notas_dp
      ↓  [Módulo 3] menor caminho (DP)
    caminho
      ↓  bridge → tab_input
      ↓  [Módulo 4] renderização ASCII
    Tablatura impressa

Uso:
    from baixo_tab import pipeline_completo
    resultado = pipeline_completo("musica.mp3")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import math
import time
import statistics

import numpy as np
import librosa
import librosa.display
import scipy.signal as sig


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 1b — Detecção de Pitch Polifônica (pYIN por corda)
# ═════════════════════════════════════════════════════════════════════════════

CORDAS_5 = [
    {"nome": "B",  "nota": "B0",  "fmin": 28.0,  "fmax":  78.0},
    {"nome": "E",  "nota": "E1",  "fmin": 36.0,  "fmax": 105.0},
    {"nome": "A",  "nota": "A1",  "fmin": 55.0,  "fmax": 130.0},
    {"nome": "D",  "nota": "D2",  "fmin": 73.0,  "fmax": 175.0},
    {"nome": "G",  "nota": "G2",  "fmin": 98.0,  "fmax": 250.0},
]

CORDAS_4 = [
    {"nome": "E",  "nota": "E1",  "fmin": 36.0,  "fmax": 105.0},
    {"nome": "A",  "nota": "A1",  "fmin": 55.0,  "fmax": 130.0},
    {"nome": "D",  "nota": "D2",  "fmin": 73.0,  "fmax": 175.0},
    {"nome": "G",  "nota": "G2",  "fmin": 98.0,  "fmax": 250.0},
]

_VOICED_PROB_THRESHOLD = 0.10
_RMS_FRAME_ON          = 0.03
_RMS_FRAME_OFF         = 0.015
_HARMONIC_RATIO        = 0.08
_HARMONIC_N            = 3
_RMS_REF_FACTOR = {"B": 0.0006, "E": 0.0007, "A": 0.0009, "D": 0.0011, "G": 0.0013}


def _filtro_passa_banda(y: np.ndarray, sr: int,
                         fmin: float, fmax: float,
                         ordem: int = 8) -> np.ndarray:
    """Butterworth passa-banda via SOS (numericamente estável para frequências baixas)."""
    nyq  = sr / 2.0
    low  = max(fmin / nyq, 1e-4)
    high = min(fmax / nyq, 0.9999)
    if low >= high:
        return np.zeros_like(y)
    sos = sig.butter(ordem, [low, high], btype="band", output="sos")
    return sig.sosfiltfilt(sos, y).astype(y.dtype)


def _corrigir_oitava(f0: np.ndarray, y_band: np.ndarray,
                      sr: int, hop_length: int,
                      fmin_corda: float) -> np.ndarray:
    """Corrige para fundamental real quando sub-harmônico é dominante."""
    f0_corr = f0.copy()
    y_band  = np.nan_to_num(y_band, nan=0.0, posinf=0.0, neginf=0.0)
    n_fft = 4096
    S     = np.abs(librosa.stft(y_band, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = min(len(f0), S.shape[1])

    for i in range(n_frames):
        fi = f0[i]
        if np.isnan(fi) or not np.isfinite(fi):
            continue
        f_baixo = fi / 2.0
        if f_baixo < fmin_corda * 0.85:
            continue

        def _energia(freq):
            idx = np.argmin(np.abs(freqs - freq))
            w   = max(1, int(freq * 0.05 / (sr / n_fft)))
            lo  = max(0, idx - w)
            hi  = min(S.shape[0], idx + w + 1)
            return float(np.sum(S[lo:hi, i] ** 2))

        e_fi      = _energia(fi)
        e_f_baixo = _energia(f_baixo)
        if e_fi > 0 and (e_f_baixo / e_fi) >= 0.65:
            f0_corr[i] = f_baixo

    return f0_corr


def _mascara_energia(y_band: np.ndarray, sr: int,
                      hop_length: int, n_frames: int) -> np.ndarray:
    """Máscara histerética de energia por frame."""
    frame_length = hop_length * 2
    rms = librosa.feature.rms(
        y=y_band, frame_length=frame_length, hop_length=hop_length
    )[0]
    if len(rms) < n_frames:
        rms = np.pad(rms, (0, n_frames - len(rms)))
    else:
        rms = rms[:n_frames]
    pico = np.max(rms)
    if pico < 1e-10:
        return np.zeros(n_frames, dtype=bool)

    mask, ativo = np.zeros(n_frames, dtype=bool), False
    for i, valor in enumerate(rms):
        if ativo:
            ativo = valor >= (pico * _RMS_FRAME_OFF)
        else:
            ativo = valor >= (pico * _RMS_FRAME_ON)
        mask[i] = ativo
    return mask


def _spectral_gate(y: np.ndarray, sr: int,
                   n_fft: int = 2048, hop_length: int = 512,
                   noise_quantile: float = 0.10,
                   gate_multiplier: float = 1.2) -> np.ndarray:
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    noise_floor = np.quantile(mag, noise_quantile, axis=1, keepdims=True)
    mask = mag >= (noise_floor * gate_multiplier)
    S_clean = S * mask
    return librosa.istft(S_clean, hop_length=hop_length, length=len(y))


def _wiener_denoise(y: np.ndarray, tamanho_janela: int = 31) -> np.ndarray:
    if y.size == 0 or np.all(y == y[0]):
        return y
    tamanho_janela = min(tamanho_janela, y.size if y.ndim == 1 else 31)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            y_d = sig.wiener(y, mysize=tamanho_janela)
        if y_d is not None and np.all(np.isfinite(y_d)):
            return y_d.astype(y.dtype)
    except Exception:
        pass
    return y


def _frame_params(fmin: float) -> tuple[int, int]:
    if fmin < 40:
        fl = 4096
    elif fmin < 70:
        fl = 3072
    else:
        fl = 2048
    return fl, max(256, fl // 4)


def _validar_harmonicos(y_band: np.ndarray, f0: np.ndarray,
                         sr: int, hop_length: int) -> np.ndarray:
    n_fft = max(4096, hop_length * 8)
    S     = np.abs(librosa.stft(y_band, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    noise = np.median(S, axis=1)
    valid = np.zeros(len(f0), dtype=bool)

    for i, fi in enumerate(f0):
        if np.isnan(fi) or fi <= 0:
            continue
        energy = noise_sum = 0.0
        for h in range(1, _HARMONIC_N + 1):
            fh = fi * h
            if fh > freqs[-1]:
                break
            idx = int(np.argmin(np.abs(freqs - fh)))
            w   = max(1, int(fh * 0.02 / (sr / n_fft)))
            lo  = max(0, idx - w)
            hi  = min(len(freqs), idx + w + 1)
            energy    += float(np.sum(S[lo:hi, i]))
            noise_sum += float(np.sum(noise[lo:hi]))
        if energy > 0 and noise_sum > 0 and (energy / (noise_sum + 1e-12)) >= _HARMONIC_RATIO:
            valid[i] = True
    return valid


def _rms_trecho(y_band: np.ndarray, sr: int,
                inicio: float, fim: float) -> float:
    i0 = int(inicio * sr)
    i1 = int(fim    * sr)
    trecho = y_band[i0:i1]
    if len(trecho) == 0:
        return 0.0
    return float(np.sqrt(np.mean(trecho ** 2)))


def _gerar_segmentos(times: np.ndarray, midi_notes: np.ndarray,
                      nome_corda: str, duracao_minima: float,
                      sr: int, hop_length: int,
                      y_band: np.ndarray | None = None) -> list[dict]:
    """Agrupa frames pYIN em segmentos de nota, tolerando pequenas oscilações."""
    if len(times) == 0 or len(midi_notes) == 0:
        return []
    times      = np.asarray(times)
    midi_notes = np.asarray(midi_notes)
    n = min(len(times), len(midi_notes))
    times, midi_notes = times[:n], midi_notes[:n]

    frame_dur  = hop_length / sr
    min_frames = max(3, int(np.ceil(duracao_minima / frame_dur)))
    segmentos  = []

    nota_atual = None
    inicio_idx = 0
    cont       = 0

    def _salvar(nota, i_ini, i_fim):
        duracao = times[i_fim] - times[i_ini]
        if cont >= min_frames and duracao >= duracao_minima:
            seg = {"nota_midi": round(nota),
                   "inicio":    float(times[i_ini]),
                   "fim":       float(times[i_fim]),
                   "corda":     nome_corda}
            if y_band is not None:
                seg["rms_medio"] = _rms_trecho(y_band, sr,
                                                seg["inicio"], seg["fim"])
            segmentos.append(seg)

    for i, nota in enumerate(midi_notes):
        if not np.isfinite(nota):
            if nota_atual is not None:
                _salvar(nota_atual, inicio_idx, i - 1)
            nota_atual, cont = None, 0
            continue
        if nota_atual is None:
            nota_atual, inicio_idx, cont = nota, i, 1
            continue
        if abs(nota - nota_atual) <= 0.4:
            nota_atual = 0.8 * nota_atual + 0.2 * nota
            cont += 1
        else:
            _salvar(nota_atual, inicio_idx, i - 1)
            nota_atual, inicio_idx, cont = nota, i, 1

    if nota_atual is not None:
        _salvar(nota_atual, inicio_idx, len(times) - 1)

    return segmentos


def _pyin_uma_corda(y_harmonic: np.ndarray, sr: int,
                    corda: dict, rms_ref: float,
                    duracao_minima: float = 0.10,
                    voiced_prob_threshold: float = _VOICED_PROB_THRESHOLD) -> list[dict]:
    """Roda pYIN no canal filtrado de uma corda e retorna segmentos."""
    fmin, fmax, nome = corda["fmin"], corda["fmax"], corda["nome"]
    frame_length, hop_length = _frame_params(fmin)

    y_band = _filtro_passa_banda(y_harmonic, sr, fmin, fmax)
    y_band = np.nan_to_num(y_band, nan=0.0, posinf=0.0, neginf=0.0)
    y_band = _spectral_gate(y_band, sr, n_fft=max(frame_length, 2048),
                             hop_length=hop_length)
    y_band = _wiener_denoise(y_band)

    rms_canal = float(np.sqrt(np.mean(y_band ** 2)))
    limiar    = max(rms_ref * _RMS_REF_FACTOR.get(nome, 0.001), 1e-8)
    if rms_canal < limiar:
        return []

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_band,
        fmin=fmin * 0.88, fmax=fmax * 1.08,
        sr=sr, frame_length=frame_length, hop_length=hop_length,
    )
    times    = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    n_frames = len(f0)

    voiced_estrito = voiced_flag & (voiced_probs >= voiced_prob_threshold)
    f0_clean       = f0.copy()
    f0_clean[~voiced_estrito]                          = np.nan
    f0_clean[~_mascara_energia(y_band, sr, hop_length, n_frames)] = np.nan
    f0_clean[~_validar_harmonicos(y_band, f0_clean, sr, hop_length)] = np.nan

    mascara_banda = (f0_clean >= fmin * 0.88) & (f0_clean <= fmax * 1.08)
    f0_clean[~mascara_banda] = np.nan

    nan_mask = np.isnan(f0_clean)
    f0_clean = _corrigir_oitava(f0_clean, y_band, sr, hop_length, fmin)
    f0_clean[nan_mask] = np.nan
    f0_clean           = np.where(np.isinf(f0_clean), np.nan, f0_clean)

    with np.errstate(invalid="ignore"):
        midi_q = np.round(librosa.hz_to_midi(f0_clean))

    return _gerar_segmentos(times, midi_q, nome, duracao_minima,
                             sr, hop_length, y_band=y_band)


def _mesclar_segmentos(todos: list[dict],
                        tolerancia_overlap: float = 0.03,
                        duracao_minima: float = 0.10) -> list[dict]:
    """Funde segmentos duplicados/sobrepostos da mesma nota."""
    filtrados = [s for s in todos if (s["fim"] - s["inicio"]) >= duracao_minima]
    if not filtrados:
        return []
    filtrados.sort(key=lambda s: s["inicio"])
    fundidos: list[dict] = []

    for seg in filtrados:
        fundiu = False
        for f in fundidos:
            if f["nota_midi"] != seg["nota_midi"]:
                continue
            gap         = seg["inicio"] - f["fim"]
            ov_ini      = max(f["inicio"], seg["inicio"])
            ov_fim      = min(f["fim"],    seg["fim"])
            tem_overlap = (ov_fim - ov_ini) > tolerancia_overlap
            adjacente   = 0 <= gap <= tolerancia_overlap
            if tem_overlap or adjacente:
                f["inicio"] = min(f["inicio"], seg["inicio"])
                f["fim"]    = max(f["fim"],    seg["fim"])
                fundiu = True
                break
        if not fundiu:
            fundidos.append(dict(seg))

    fundidos = [s for s in fundidos if (s["fim"] - s["inicio"]) >= duracao_minima]
    fundidos.sort(key=lambda s: (s["inicio"], s["nota_midi"]))
    return fundidos


def analisar_baixo_polifonico(
    audio_path: str,
    num_cordas: int = 5,
    plotar: bool = False,
    duracao_minima: float = 0.10,
    tolerancia_overlap: float = 0.03,
    voiced_prob_threshold: float | None = None,
) -> list[dict]:
    """Carrega áudio e roda pYIN por corda. Retorna lista de segmentos brutos."""
    print(f"\n--- [POLIFÔNICO] Processando: {audio_path.split('/')[-1].split(chr(92))[-1]} ---")
    y, sr         = librosa.load(audio_path)
    print("Separando componentes harmônicos...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
    rms_ref       = float(np.sqrt(np.mean(y_harmonic ** 2)))

    cordas     = CORDAS_5 if num_cordas == 5 else CORDAS_4
    vt         = _VOICED_PROB_THRESHOLD if voiced_prob_threshold is None else voiced_prob_threshold
    todos_segs = []

    for corda in cordas:
        print(f"  [pYIN] Corda {corda['nome']} ({corda['nota']}) "
              f"[{corda['fmin']:.0f}–{corda['fmax']:.0f} Hz]...", end=" ", flush=True)
        t0   = time.time()
        segs = _pyin_uma_corda(y_harmonic, sr, corda, rms_ref,
                                duracao_minima=duracao_minima,
                                voiced_prob_threshold=vt)
        print(f"{len(segs)} segmentos  ({time.time()-t0:.1f}s)")
        todos_segs.extend(segs)

    segmentos = _mesclar_segmentos(todos_segs, tolerancia_overlap, duracao_minima)
    print(f"\n[✓] {len(segmentos)} segmentos após mesclagem")
    return segmentos


# ─────────────────────────────────────────────────────────────────────────────
# Bridge 1b → 2

_CORDAS_MIDI_ABERTA_INF = {
    "G": 43, "D": 38, "A": 33, "E": 28, "B": 23
}


def _inferir_corda_real(nota_midi: float, corda_filtro: str) -> str:
    """Determina a corda mais natural para a nota (menor número de traste)."""
    nota_int     = int(round(nota_midi))
    melhor_corda = corda_filtro
    menor_traste = 999
    for nome, midi_aberta in _CORDAS_MIDI_ABERTA_INF.items():
        traste = nota_int - midi_aberta
        if 0 <= traste <= 24 and traste < menor_traste:
            menor_traste = traste
            melhor_corda = nome
    return melhor_corda


def bridge_pyin_para_notas_poli(
    segmentos: list[dict],
    duracao_minima: float = 0.10,
) -> list[dict]:
    """Converte segmentos brutos em sequência de notas com nome e corda real."""
    resultado = []
    for seg in segmentos:
        if (seg["fim"] - seg["inicio"]) < duracao_minima:
            continue
        corda_real = _inferir_corda_real(seg["nota_midi"],
                                          seg.get("corda", "?"))
        resultado.append({
            "nome":   librosa.midi_to_note(int(seg["nota_midi"])),
            "inicio": seg["inicio"],
            "fim":    seg["fim"],
            "corda":  corda_real,
        })
    return resultado


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 1.5 — Pré-processamento de Segmentos
# ═════════════════════════════════════════════════════════════════════════════

_CORDAS_MIDI_PP = {"G": 43, "D": 38, "A": 33, "E": 28, "B": 23}
_MAX_TRASTE_PP  = 24


def _freq_hz(nota_midi: int) -> float:
    return 440.0 * (2.0 ** ((nota_midi - 69) / 12.0))


def _sobrepostos(a: dict, b: dict, margem: float = 0.10) -> bool:
    ov_ini = max(a["inicio"], b["inicio"])
    ov_fim = min(a["fim"],    b["fim"])
    gap    = (abs(b["inicio"] - a["fim"])
              if b["inicio"] >= a["fim"]
              else abs(a["inicio"] - b["fim"]))
    return (ov_fim - ov_ini) > 0 or gap <= margem


def _traste_pp(nota_midi: int, corda: str) -> int:
    return nota_midi - _CORDAS_MIDI_PP.get(corda, 0)


def _traste_minimo_pp(nota_midi: int) -> int:
    candidatos = [nota_midi - m
                  for m in _CORDAS_MIDI_PP.values()
                  if 0 <= nota_midi - m <= _MAX_TRASTE_PP]
    return min(candidatos) if candidatos else -1


def filtrar_por_onset(
    segmentos: list[dict],
    rms_ratio_onset: float = 0.15,
    janela_busca: float = 2.0,
) -> list[dict]:
    """Descarta segmentos antes do primeiro onset real de energia."""
    if not segmentos:
        return segmentos
    com_rms = [s for s in segmentos if "rms_medio" in s]
    if not com_rms:
        return segmentos

    valores = sorted(s["rms_medio"] for s in com_rms)
    metade  = valores[len(valores) // 2:]
    rms_ref = statistics.median(metade) if metade else 0.0
    if rms_ref == 0.0:
        return segmentos

    limiar      = rms_ref * rms_ratio_onset
    tempo_onset = None
    for seg in sorted(segmentos, key=lambda s: s["inicio"]):
        if seg["inicio"] > janela_busca:
            break
        if seg.get("rms_medio", 0.0) >= limiar:
            tempo_onset = seg["inicio"]
            break

    if tempo_onset is None:
        return segmentos

    antes  = [s for s in segmentos if s["fim"] <= tempo_onset]
    depois = [s for s in segmentos if s["fim"] > tempo_onset]
    if antes:
        nomes = ", ".join(f"{s.get('corda','?')}:{s['nota_midi']}" for s in antes)
        print(f"        [onset] {len(antes)} pré-música descartado(s) "
              f"(antes t={tempo_onset:.2f}s): {nomes}")
    return depois


def filtrar_harmonicos(
    segmentos: list[dict],
    harmonicos_verificar: tuple[int, ...] = (2, 3, 4),
    tolerancia_cents: float = 40.0,
    margem_temporal: float = 0.10,
    rms_ratio_harm: float = 1.5,
    excesso_traste_max: int = 4,
) -> list[dict]:
    """Descarta harmônicos fantasmas gerados por vazamento do filtro passa-banda."""
    if not segmentos:
        return segmentos
    descartar: set[int] = set()

    def _e_fantasma(fund: dict, cand: dict) -> bool:
        freq_f = _freq_hz(fund["nota_midi"])
        freq_c = _freq_hz(cand["nota_midi"])
        if freq_f <= 0:
            return False
        razao = freq_c / freq_f
        for n in harmonicos_verificar:
            if razao <= 0:
                continue
            cents_erro = abs(1200 * math.log2(razao / n))
            if cents_erro > tolerancia_cents:
                continue
            t_cand   = _traste_pp(cand["nota_midi"], cand.get("corda", "?"))
            t_min    = _traste_minimo_pp(cand["nota_midi"])
            invalido = t_cand > _MAX_TRASTE_PP or t_cand < 0
            excesso  = t_cand - t_min if t_min >= 0 else 999
            nao_nat  = excesso > excesso_traste_max
            if invalido:
                return True
            rms_f = fund.get("rms_medio", 0.0)
            rms_c = cand.get("rms_medio", 0.0)
            if nao_nat and rms_f >= rms_c * rms_ratio_harm:
                return True
        return False

    for i, a in enumerate(segmentos):
        if i in descartar:
            continue
        for j, b in enumerate(segmentos):
            if j == i or j in descartar:
                continue
            if not _sobrepostos(a, b, margem_temporal):
                continue
            if _e_fantasma(a, b):
                descartar.add(j)
            elif _e_fantasma(b, a):
                descartar.add(i)

    if descartar:
        removidos = [segmentos[j] for j in sorted(descartar)]
        nomes = ", ".join(
            f"{s.get('corda','?')}:{librosa.midi_to_note(int(s['nota_midi']))}"
            f"(t={_traste_pp(s['nota_midi'], s.get('corda','?'))},"
            f"rms={s.get('rms_medio',0):.4f})"
            for s in removidos
        )
        print(f"        [harmonicos] {len(removidos)} fantasma(s) descartado(s): {nomes}")
    return [s for i, s in enumerate(segmentos) if i not in descartar]


def filtrar_corda_errada(
    segmentos: list[dict],
    janela_sobreposicao: float = 0.15,
    rms_ratio_corda: float = 3.0,
) -> list[dict]:
    """Remove a mesma nota MIDI detectada em duas cordas quando uma tem RMS muito maior."""
    if not segmentos:
        return segmentos

    _PRIO = {"G": 0, "D": 1, "A": 2, "E": 3, "B": 4}
    descartar: set[int] = set()

    for i, a in enumerate(segmentos):
        if i in descartar:
            continue
        for j, b in enumerate(segmentos):
            if j <= i or j in descartar:
                continue
            if a["nota_midi"] != b["nota_midi"]:
                continue
            if a["corda"] == b["corda"]:
                continue
            if not _sobrepostos(a, b, janela_sobreposicao):
                continue
            rms_a = a.get("rms_medio")
            rms_b = b.get("rms_medio")
            if rms_a is not None and rms_b is not None and (rms_a + rms_b) > 0:
                if rms_a >= rms_b * rms_ratio_corda:
                    descartar.add(j)
                elif rms_b >= rms_a * rms_ratio_corda:
                    descartar.add(i)
            else:
                pa, pb = _PRIO.get(a["corda"], 9), _PRIO.get(b["corda"], 9)
                if pa < pb:
                    descartar.add(j)
                elif pb < pa:
                    descartar.add(i)

    if descartar:
        removidos = [segmentos[j] for j in sorted(descartar)]
        print(f"        [corda_errada] {len(removidos)} descartado(s): "
              + ", ".join(f"{s.get('corda','?')}:{s['nota_midi']}" for s in removidos))
    return [s for i, s in enumerate(segmentos) if i not in descartar]


def _juntar_gaps(segmentos: list[dict], gap_maximo: float) -> list[dict]:
    unidos = []
    for seg in segmentos:
        if not unidos:
            unidos.append(dict(seg))
            continue
        u = unidos[-1]
        gap = seg["inicio"] - u["fim"]
        if (u["nota_midi"] == seg["nota_midi"]
                and u["corda"] == seg["corda"]
                and gap <= gap_maximo):
            u["fim"] = max(u["fim"], seg["fim"])
            if "rms_medio" in u and "rms_medio" in seg:
                da = u["fim"] - u["inicio"]
                db = seg["fim"] - seg["inicio"]
                u["rms_medio"] = (u["rms_medio"] * da + seg["rms_medio"] * db) / max(da + db, 1e-9)
        else:
            unidos.append(dict(seg))
    return unidos


def limpar_segmentos(
    segmentos: list[dict],
    duracao_minima: float = 0.12,
    gap_maximo: float     = 0.06,
    nota_curta: float     = 0.08,
    filtrar_onset: bool       = True,
    rms_ratio_onset: float    = 0.15,
    janela_busca_onset: float = 2.0,
    filtrar_harmonicos_flag: bool = True,
    harmonicos_verificar: tuple   = (2, 3, 4),
    tolerancia_cents: float       = 35.0,
    margem_temporal_harm: float   = 0.10,
    rms_ratio_harm: float         = 1.5,
    filtrar_vazamento: bool    = True,
    janela_sobreposicao: float = 0.15,
    rms_ratio_corda: float     = 3.0,
) -> list[dict]:
    """Pipeline de limpeza de segmentos em 7 etapas sequenciais."""
    if not segmentos:
        return []

    segs = sorted(segmentos, key=lambda s: s["inicio"])

    if filtrar_onset:
        segs = filtrar_por_onset(segs, rms_ratio_onset, janela_busca_onset)

    if filtrar_harmonicos_flag:
        segs = filtrar_harmonicos(segs, harmonicos_verificar, tolerancia_cents,
                                   margem_temporal_harm, rms_ratio_harm)

    if filtrar_vazamento:
        segs = filtrar_corda_errada(segs, janela_sobreposicao, rms_ratio_corda)

    segs = _juntar_gaps(segs, gap_maximo)
    segs = [s for s in segs if (s["fim"] - s["inicio"]) >= duracao_minima]

    # Remove nota curta isolada entre duas iguais (ornamento falso)
    resultado = []
    for i, atual in enumerate(segs):
        if 0 < i < len(segs) - 1:
            ant, prox = segs[i - 1], segs[i + 1]
            dur = atual["fim"] - atual["inicio"]
            if dur < nota_curta and ant["nota_midi"] == prox["nota_midi"]:
                continue
        resultado.append(atual)
    segs = resultado

    # Funde duplicatas MIDI consecutivas na mesma corda
    final = []
    for seg in segs:
        if (final
                and final[-1]["nota_midi"] == seg["nota_midi"]
                and final[-1]["corda"]     == seg["corda"]):
            final[-1]["fim"] = seg["fim"]
        else:
            final.append(seg)

    return final


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — Posições no Braço do Baixo
# ═════════════════════════════════════════════════════════════════════════════

_MAPA_NOTAS = {
    "C": 0, "C#": 1, "C♯": 1, "Db": 1, "D": 2, "D#": 3, "D♯": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "F♯": 6, "Gb": 6, "G": 7, "G#": 8, "G♯": 8,
    "Ab": 8, "A": 9, "A#": 10, "A♯": 10, "Bb": 10, "B": 11,
}

# corda aberta → número de corda (1=G, 5=B)
_CORDAS_INFO = {20: 1, 15: 2, 10: 3, 5: 4, 0: 5}

_MAX_CASAS = 24

_CORDA_NOME_PARA_NUM = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5}


def _prioridade_corda(posicao: list[int]) -> tuple[int, int, int]:
    corda, traste = posicao
    prio = 0 if corda <= 3 else 1 if corda == 4 else 2
    return (prio, traste, corda)


def _nota_para_indice(nota: str) -> int:
    nome   = nota[:-1]
    oitava = int(nota[-1])
    return (oitava * 12) + _MAPA_NOTAS[nome] - 11


def calcular_casas_baixo(vetor_notas: list[str]) -> dict[str, list[list[int]]]:
    """Calcula todas as posições válidas [corda, traste] para cada nota."""
    notas_unicas = list(dict.fromkeys(vetor_notas))
    resultado    = {}
    for nome in notas_unicas:
        val           = _nota_para_indice(nome)
        possibilidades = []
        for val_corda, num_corda in _CORDAS_INFO.items():
            traste = val - val_corda
            if 0 <= traste <= _MAX_CASAS:
                possibilidades.append([num_corda, traste])
        possibilidades.sort(key=_prioridade_corda)
        resultado[nome] = possibilidades
    return resultado


def _mapear_corda(corda) -> int | None:
    if corda is None:
        return None
    if isinstance(corda, int):
        return corda
    if isinstance(corda, str):
        return _CORDA_NOME_PARA_NUM.get(corda.upper())
    return None


def bridge_notas_para_dp(
    sequencia_notas: list[dict],
    posicoes_dict: dict[str, list[list[int]]],
    preferir_corda_detectada: bool = True,
) -> list[dict]:
    """Formata sequência de notas para entrada do algoritmo DP."""
    notas_dp = []
    for nota in sequencia_notas:
        nome = nota["nome"]
        if nome not in posicoes_dict or not posicoes_dict[nome]:
            print(f"[AVISO] Nota '{nome}' sem posição válida — ignorada.")
            continue
        posicoes = list(posicoes_dict[nome])
        if preferir_corda_detectada:
            cp = _mapear_corda(nota.get("corda"))
            if cp in (1, 2, 3, 4):
                posicoes.sort(key=lambda p: (p[0] != cp, _prioridade_corda(p)))
            else:
                posicoes.sort(key=_prioridade_corda)
        else:
            posicoes.sort(key=_prioridade_corda)
        entrada = {"nome": nome, "posicoes": posicoes}
        for chave in ("inicio", "fim", "corda"):
            if chave in nota:
                entrada[chave] = nota[chave]
        notas_dp.append(entrada)
    return notas_dp


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — Menor Caminho (Programação Dinâmica)
# ═════════════════════════════════════════════════════════════════════════════

def _custo_fingering(pos1: list[int], pos2: list[int],
                      peso_corda: float = 2.0,
                      peso_traste: float = 1.0) -> float:
    """Custo de movimento entre posições, penalizando cordas graves."""
    if pos2[0] <= 3:
        pen = 0.0
    elif pos2[0] == 4:
        pen = 0.5
    else:
        pen = 1.5
    return abs(pos2[1] - pos1[1]) * peso_traste + abs(pos2[0] - pos1[0]) * peso_corda + pen


def menor_caminho_DP(
    notas: list[dict],
    fn_custo=None,
    preferir_corda_detectada: bool = True,
    bonus_corda: float = 0.5,
) -> tuple[float, list[dict]]:
    """
    DP (DAG shortest path) sobre as posições possíveis de cada nota.
    Retorna (custo_total, caminho), onde caminho[i] = {"nota", "posicao", ...}.
    """
    if not notas:
        raise ValueError("Lista de notas vazia.")
    for i, n in enumerate(notas):
        if not n.get("posicoes"):
            raise ValueError(f"Nota [{i}] '{n.get('nome')}' sem posições.")

    fn = fn_custo or _custo_fingering
    N  = len(notas)
    dp = [[None] * len(notas[i]["posicoes"]) for i in range(N)]

    # Caso base
    for j in range(len(notas[0]["posicoes"])):
        dp[0][j] = (0.0, None)

    # Preenchimento
    for i in range(1, N):
        cp = _mapear_corda(notas[i].get("corda")) if preferir_corda_detectada else None
        for j, pos_j in enumerate(notas[i]["posicoes"]):
            melhor, de = float("inf"), None
            for k, pos_k in enumerate(notas[i - 1]["posicoes"]):
                c = dp[i - 1][k][0] + fn(pos_k, pos_j)
                if cp is not None and pos_j[0] == cp:
                    c -= bonus_corda
                if c < melhor:
                    melhor, de = c, k
            dp[i][j] = (melhor, de)

    # Mínimo na última nota
    best_j      = min(range(len(notas[-1]["posicoes"])), key=lambda j: dp[-1][j][0])
    custo_total = dp[-1][best_j][0]

    # Backtracking
    caminho, j = [], best_j
    for i in range(N - 1, -1, -1):
        entrada = {"nota": notas[i]["nome"], "posicao": notas[i]["posicoes"][j]}
        for chave, valor in notas[i].items():
            if chave not in ("nome", "posicoes"):
                entrada[chave] = valor
        caminho.append(entrada)
        if i > 0:
            j = dp[i][j][1]

    return custo_total, list(reversed(caminho))


def bridge_caminho_para_tab(caminho: list[dict]) -> list[list]:
    """Converte caminho do DP para formato [[corda, traste, tempo_idx], ...]."""
    return [[p["posicao"][0], p["posicao"][1], i] for i, p in enumerate(caminho)]


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — Renderização de Tablatura ASCII
# ═════════════════════════════════════════════════════════════════════════════

_NOMES_5 = {1: "G", 2: "D", 3: "A", 4: "E", 5: "B"}
_NOMES_4 = {1: "G", 2: "D", 3: "A", 4: "E"}


def print_tab_multiplas_notas(
    sequencia: list[list[int]],
    num_cordas: int = 5,
) -> str:
    """Imprime e retorna tablatura ASCII de um baixo de 4 ou 5 cordas."""
    nomes    = _NOMES_5 if num_cordas == 5 else _NOMES_4
    tab      = {i: f"{nomes[i]}|--" for i in range(1, num_cordas + 1)}

    if not sequencia:
        resultado = "\n".join(tab[i] for i in range(1, num_cordas + 1))
        print(resultado)
        return resultado

    # Agrupa por tempo_idx
    blocos, bloco_atual, tempo_ref = [], [], sequencia[0][2]
    for corda, casa, tempo in sequencia:
        if tempo == tempo_ref:
            bloco_atual.append((corda, casa))
        else:
            blocos.append(bloco_atual)
            bloco_atual = [(corda, casa)]
            tempo_ref   = tempo
    blocos.append(bloco_atual)

    for bloco in blocos:
        if any(n[0] == 0 for n in bloco):       # comando de pulo de tempo
            for c in range(1, num_cordas + 1):
                tab[c] += "---"
            continue

        largura = max(len(str(n[1])) for n in bloco) if bloco else 1
        notas_b = {n[0]: n[1] for n in bloco}

        for c in range(1, num_cordas + 1):
            if c in notas_b:
                s     = str(notas_b[c])
                tab[c] += s + "-" * (largura - len(s) + 1)
            else:
                tab[c] += "-" * (largura + 1)

        for c in range(1, num_cordas + 1):
            tab[c] += "-"

    linhas    = [tab[i] for i in range(1, num_cordas + 1)]
    resultado = "\n".join(linhas)
    print(resultado)
    return resultado


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═════════════════════════════════════════════════════════════════════════════

def pipeline_completo(
    audio_path: str,
    duracao_minima: float = 0.05,
    num_cordas: int = 5,
    tolerancia_overlap: float = 0.03,
    preferir_corda_detectada: bool = True,
    # ── Pré-processamento ────────────────────────────────────────────────────
    pre_processar: bool = True,
    pp_duracao_minima: float = 0.12,
    pp_gap_maximo: float = 0.06,
    pp_nota_curta: float = 0.08,
    pp_filtrar_onset: bool = True,
    pp_rms_ratio_onset: float = 0.15,
    pp_janela_busca_onset: float = 2.0,
    pp_filtrar_vazamento: bool = True,
    pp_janela_sobreposicao: float = 0.15,
    pp_rms_ratio_corda: float = 3.0,
) -> dict:
    """
    Executa o pipeline completo: áudio → tablatura.

    Parâmetros
    ----------
    audio_path               : Caminho do arquivo de áudio.
    duracao_minima           : Notas abaixo deste valor (s) descartadas pelo bridge.
    num_cordas               : 4 ou 5 cordas.
    tolerancia_overlap       : Limiar para fundir duplicatas no pYIN.
    preferir_corda_detectada : Bônus na DP para posições na corda detectada.
    pre_processar            : Liga/desliga o Módulo 1.5.
    pp_*                     : Parâmetros finos do pré-processamento.

    Retorna
    -------
    dict com as saídas de cada etapa:
        segmentos_raw, segmentos, sequencia, posicoes_dict,
        notas_dp, custo_total, caminho, tab_input.
    """
    pp_str = (f"ATIVO (dur≥{pp_duracao_minima}s, gap≤{pp_gap_maximo}s)"
              if pre_processar else "DESATIVADO")

    print("\n" + "=" * 60)
    print("  PIPELINE: Áudio → Tablatura de Baixo [POLIFÔNICO]")
    print(f"  Preferir corda detectada : {preferir_corda_detectada}")
    print(f"  Pré-processamento        : {pp_str}")
    print("=" * 60)

    # ── Etapa 1: pYIN por corda ───────────────────────────────────────────────
    print("\n[1/4] Detecção de pitch polifônica (pYIN por corda)...")
    segmentos_raw = analisar_baixo_polifonico(
        audio_path,
        num_cordas=num_cordas,
        duracao_minima=duracao_minima,
        tolerancia_overlap=tolerancia_overlap,
    )
    print(f"      → {len(segmentos_raw)} segmentos brutos.")

    # ── Etapa 1.5: Pré-processamento ─────────────────────────────────────────
    if pre_processar:
        print("\n[1.5/4] Pré-processamento (Módulo 1.5)...")
        segmentos = limpar_segmentos(
            segmentos_raw,
            duracao_minima=pp_duracao_minima,
            gap_maximo=pp_gap_maximo,
            nota_curta=pp_nota_curta,
            filtrar_onset=pp_filtrar_onset,
            rms_ratio_onset=pp_rms_ratio_onset,
            janela_busca_onset=pp_janela_busca_onset,
            filtrar_vazamento=pp_filtrar_vazamento,
            janela_sobreposicao=pp_janela_sobreposicao,
            rms_ratio_corda=pp_rms_ratio_corda,
        )
        removidos = len(segmentos_raw) - len(segmentos)
        print(f"        → {len(segmentos)} segmentos limpos "
              f"({removidos} removidos).")
    else:
        segmentos = segmentos_raw
        print("\n[1.5/4] Pré-processamento desativado.")

    sequencia = bridge_pyin_para_notas_poli(segmentos,
                                             duracao_minima=duracao_minima)
    print(f"      → {len(sequencia)} notas após filtragem.")
    if not sequencia:
        print("[ERRO] Nenhuma nota detectada. Verifique o áudio.")
        return {}

    # ── Etapa 2: Posições no braço ────────────────────────────────────────────
    print("\n[2/4] Calculando posições no braço...")
    nomes_unicos  = list(dict.fromkeys(n["nome"] for n in sequencia))
    posicoes_dict = calcular_casas_baixo(nomes_unicos)
    for nota, pos in posicoes_dict.items():
        print(f"      {nota}: {pos}")

    notas_dp = bridge_notas_para_dp(sequencia, posicoes_dict,
                                     preferir_corda_detectada=preferir_corda_detectada)
    if not notas_dp:
        print("[ERRO] Nenhuma nota com posição válida para o DP.")
        return {}

    # ── Etapa 3: Menor caminho (DP) ───────────────────────────────────────────
    print(f"\n[3/4] Menor caminho entre {len(notas_dp)} notas (DP)...")
    custo_total, caminho = menor_caminho_DP(
        notas_dp,
        preferir_corda_detectada=preferir_corda_detectada,
    )
    print(f"      → Custo total: {custo_total:.1f}")
    for passo in caminho:
        ci = (f"  [corda detectada: {passo.get('corda','?')}]"
              if "corda" in passo else "")
        print(f"         {passo['nota']}: corda {passo['posicao'][0]}, "
              f"traste {passo['posicao'][1]}{ci}")

    # ── Etapa 4: Tablatura ASCII ──────────────────────────────────────────────
    tab_input = bridge_caminho_para_tab(caminho)
    print(f"\n[4/4] Tablatura ({num_cordas} cordas):\n")
    print_tab_multiplas_notas(tab_input, num_cordas=num_cordas)

    print("\n" + "=" * 60)
    print("  Pipeline concluído.")
    print("=" * 60 + "\n")

    return {
        "segmentos_raw": segmentos_raw,
        "segmentos":     segmentos,
        "sequencia":     sequencia,
        "posicoes_dict": posicoes_dict,
        "notas_dp":      notas_dp,
        "custo_total":   custo_total,
        "caminho":       caminho,
        "tab_input":     tab_input,
    }
