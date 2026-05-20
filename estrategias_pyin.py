"""
ESTRATÉGIAS ALTERNATIVAS DE DETECÇÃO POLIFÔNICA
================================================
Quatro abordagens para resolver os problemas identificados nos testes:
    cross-talk harmônico, recall baixo em cordas vizinhas, notas fantasma.

Interface comum de todas as estratégias:
    fn(y, sr, num_cordas=5, duracao_minima=0.10, tolerancia_overlap=0.03)
    → list[dict]  com campos  {nota_midi, inicio, fim, corda}

Estratégias implementadas
--------------------------
1. BASE v3          — referência, sem modificações
2. REMOÇÃO PÓS-DETECÇÃO — detecta em todas as bandas, depois remove segmentos
                           que são harmônicos inteiros (2×–6×) de notas já
                           detectadas em bandas mais graves (pós-processamento puro)
3. NOTCH PROGRESSIVO    — processa cordas grave→agudo; após cada corda aplica
                           notch espectral nos harmônicos detectados antes de
                           passar para a próxima banda
4. SUBTRAÇÃO ITERATIVA  — ideia do usuário: após detectar em cada corda, estima
                           a componente espectral via STFT (sem precisar da fase
                           exata) e subtrai do sinal residual antes da próxima corda
"""

import numpy as np
import librosa
import librosa.effects
import scipy.signal as sig

from modulo_pyin_polifonico import (
    _filtro_passa_banda,
    _pyin_uma_corda,
    _mesclar_segmentos,
    CORDAS_5,
    CORDAS_4,
    analisar_array_polifonico,
)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS COMUNS
# ══════════════════════════════════════════════════════════════════════════════

def _get_cordas(num_cordas: int) -> list[dict]:
    return CORDAS_5 if num_cordas == 5 else CORDAS_4


def _hpss(y: np.ndarray, margin: float = 3.0) -> tuple[np.ndarray, float]:
    """Retorna (y_harmônico, rms_ref) do sinal."""
    y_h = librosa.effects.hpss(y.astype(np.float32), margin=margin)[0]
    rms = float(np.sqrt(np.mean(y_h ** 2)))
    return y_h, rms


def _stft_params(sr: int) -> dict:
    return {"hop_length": 512, "n_fft": 4096}


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 1 — BASE v3
# ══════════════════════════════════════════════════════════════════════════════

def estrategia_base(y: np.ndarray, sr: int, num_cordas: int = 5,
                     duracao_minima: float = 0.10,
                     tolerancia_overlap: float = 0.03) -> list[dict]:
    """
    Referência: analisar_array_polifonico() da v3 sem nenhuma modificação.
    Todos os resultados das outras estratégias são comparados contra esta.
    """
    return analisar_array_polifonico(
        y, sr,
        num_cordas         = num_cordas,
        duracao_minima     = duracao_minima,
        tolerancia_overlap = tolerancia_overlap,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 2 — REMOÇÃO PÓS-DETECÇÃO DE HARMÔNICOS
# ══════════════════════════════════════════════════════════════════════════════
#
# Lógica: roda pYIN em todas as bandas normalmente, depois aplica um
# pós-processamento que varre os segmentos do mais grave ao mais agudo e
# descarta qualquer segmento cuja frequência seja múltipla inteira (2×–6×) de
# um segmento já aceito, com sobreposição temporal suficiente.
#
# Vantagem: simples, sem modificar o sinal — só filtra a lista de segmentos.
# Limitação: a nota grave precisa ter sido detectada para que seu harmônico
# seja removido. Se a nota grave for perdida, o harmônico sobrevive.

def _e_harmonico_de(seg_agudo: dict, seg_grave: dict,
                     tol_cents: float = 80.0,
                     min_overlap_s: float = 0.05) -> bool:
    """
    Retorna True se seg_agudo for harmônico inteiro (2×–6×) de seg_grave
    com sobreposição temporal >= min_overlap_s.
    """
    f_g = float(librosa.midi_to_hz(seg_grave["nota_midi"]))
    f_a = float(librosa.midi_to_hz(seg_agudo["nota_midi"]))
    if f_g <= 0 or f_a <= 0 or f_a <= f_g:
        return False
    ratio = f_a / f_g
    n     = round(ratio)
    if n < 2 or n > 6:
        return False
    erro_cents = abs(1200.0 * np.log2(ratio / n))
    if erro_cents > tol_cents:
        return False
    overlap = (min(seg_agudo["fim"], seg_grave["fim"]) -
               max(seg_agudo["inicio"], seg_grave["inicio"]))
    return overlap >= min_overlap_s


def estrategia_remocao_harmonicos(y: np.ndarray, sr: int, num_cordas: int = 5,
                                   duracao_minima: float = 0.10,
                                   tolerancia_overlap: float = 0.03) -> list[dict]:
    """
    Estratégia 2 — Remoção Pós-Detecção de Harmônicos.
    Detecta em todas as bandas, depois remove harmônicos fantasma.
    """
    y_h, _ = _hpss(y)
    cordas = _get_cordas(num_cordas)

    todos = []
    for corda in cordas:
        segs = _pyin_uma_corda(y_h, sr, corda)
        todos.extend(segs)

    # Aplica duração mínima e ordena grave→agudo
    todos = [s for s in todos if (s["fim"] - s["inicio"]) >= duracao_minima]
    todos.sort(key=lambda s: s["nota_midi"])

    # Marca harmônicos para remoção
    remover = set()
    for i, seg_a in enumerate(todos):
        if i in remover:
            continue
        for j in range(i):
            if j in remover:
                continue
            if _e_harmonico_de(seg_a, todos[j]):
                remover.add(i)
                break

    limpos = [s for k, s in enumerate(todos) if k not in remover]
    return _mesclar_segmentos(limpos, tolerancia_overlap, duracao_minima)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 3 — NOTCH PROGRESSIVO
# ══════════════════════════════════════════════════════════════════════════════
#
# Lógica: processa as cordas da mais grave à mais aguda. Após detectar notas
# numa corda, aplica notch espectral estreito nos HARMÔNICOS (2×, 3×, …, 6×)
# dessas notas no sinal residual. A fundamental é preservada.
# A próxima corda recebe um sinal onde os harmônicos das notas graves já
# foram atenuados, reduzindo o cross-talk.
#
# Diferença da Estratégia 4: atenua apenas harmônicos (não a fundamental).
# A Estratégia 4 subtrai fundamental + harmônicos.

def _notch_harmonicos(y: np.ndarray, sr: int, segmentos: list[dict],
                       hop_length: int = 512, n_fft: int = 4096,
                       atenuacao: float = 0.06) -> np.ndarray:
    """
    Atenua harmônicos (2×–6×) das notas detectadas no espectro STFT.
    A fundamental (h=1) é PRESERVADA — já foi detectada e reportada.

    atenuacao: fração de energia que sobra após o notch (0.06 = 6% restante).
    """
    if not segmentos:
        return y

    D     = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
    dfreq = freqs[1] - freqs[0]

    for seg in segmentos:
        f0 = float(librosa.midi_to_hz(seg["nota_midi"]))
        # Pequena margem temporal para suavizar bordas
        mask   = (times >= seg["inicio"] - 0.03) & (times <= seg["fim"] + 0.03)
        frames = np.where(mask)[0]
        if frames.size == 0:
            continue

        # h=2 em diante — salta a fundamental
        for h in range(2, 7):
            fh = f0 * h
            if fh >= sr / 2.0:
                break
            k      = np.argmin(np.abs(freqs - fh))
            w      = max(2, int(fh * 0.04 / dfreq))   # ±4% em Hz
            lo, hi = max(0, k - w), min(D.shape[0], k + w + 1)
            D[lo:hi, frames] *= atenuacao

    return librosa.istft(D, hop_length=hop_length, length=len(y)).astype(y.dtype)


def estrategia_notch_progressivo(y: np.ndarray, sr: int, num_cordas: int = 5,
                                  duracao_minima: float = 0.10,
                                  tolerancia_overlap: float = 0.03) -> list[dict]:
    """
    Estratégia 3 — Notch Progressivo.
    Processa grave→agudo. Após cada corda, aplica notch nos harmônicos
    das notas detectadas antes de rodar pYIN na próxima banda.
    """
    y_h, _  = _hpss(y)
    cordas  = _get_cordas(num_cordas)
    y_res   = y_h.copy()
    todos   = []

    for corda in cordas:
        segs    = _pyin_uma_corda(y_res, sr, corda)
        todos.extend(segs)

        # Notch dos harmônicos no resíduo (para proteger as próximas cordas)
        segs_ok = [s for s in segs if (s["fim"] - s["inicio"]) >= duracao_minima]
        if segs_ok:
            y_res = _notch_harmonicos(y_res, sr, segs_ok)

    return _mesclar_segmentos(todos, tolerancia_overlap, duracao_minima)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATÉGIA 4 — SUBTRAÇÃO ITERATIVA  (ideia do usuário)
# ══════════════════════════════════════════════════════════════════════════════
#
# Ideia original do usuário:
#   "Assim que eu rodar o pYIN e achar uma nota, ao rodar novamente eu removeria
#    o sinal encontrado. Como temos o problema de eliminar a fase exata, rodamos
#    uma FFT/STFT para saber os parâmetros médios da componente a ser removida."
#
# Implementação:
#   Após detectar segmentos em cada corda, usa o STFT do sinal residual para
#   estimar a energia frame-a-frame na frequência detectada e nos seus harmônicos.
#   Atenua esses bins no STFT e reconstrói via iSTFT — sem precisar da fase exata.
#
# Diferença da Estratégia 3:
#   Remove TAMBÉM a fundamental (h=1), não só os harmônicos.
#   O sinal residual "vê" progressivamente menos as notas já detectadas.
#
# Parâmetro fator: quanto subtrair da energia nos bins detectados.
#   fator=0.82 → deixa 18% nos bins da fundamental.
#   Harmônicos superiores são subtraídos progressivamente menos (0.9^(h-1)).
#   Isso evita artefatos por subtração excessiva.

def _subtracao_espectral(y: np.ndarray, sr: int, segmentos: list[dict],
                          hop_length: int = 512, n_fft: int = 4096,
                          fator: float = 0.82) -> np.ndarray:
    """
    Subtração espectral frame-a-frame: atenua fundamental + harmônicos
    de cada segmento detectado no espectro STFT do sinal residual.

    Não requer fase: opera diretamente nos coeficientes complexos do STFT.
    D[k, t] *= (1 - fator_h)  → magnitude reduzida, fase preservada.

    fator_h decresce com h para evitar subtração excessiva nos harmônicos:
        h=1: fator * 0.9^0  = fator        (fundamental: maior remoção)
        h=2: fator * 0.9^1  = 0.9*fator    (2° harmônico: um pouco menos)
        ...
    """
    if not segmentos:
        return y

    D     = librosa.stft(y.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
    dfreq = freqs[1] - freqs[0]

    for seg in segmentos:
        f0     = float(librosa.midi_to_hz(seg["nota_midi"]))
        # Margem temporal ligeiramente maior que no notch (cobre transientes)
        mask   = (times >= seg["inicio"] - 0.05) & (times <= seg["fim"] + 0.05)
        frames = np.where(mask)[0]
        if frames.size == 0:
            continue

        # h=1 é a fundamental — diferença-chave em relação à Estratégia 3
        for h in range(1, 7):
            fh = f0 * h
            if fh >= sr / 2.0:
                break
            k      = np.argmin(np.abs(freqs - fh))
            w      = max(2, int(fh * 0.03 / dfreq))   # ±3% em Hz
            lo, hi = max(0, k - w), min(D.shape[0], k + w + 1)
            fator_h = fator * (0.9 ** (h - 1))
            D[lo:hi, frames] *= (1.0 - fator_h)

    return librosa.istft(D, hop_length=hop_length, length=len(y)).astype(y.dtype)


def estrategia_subtracao_iterativa(y: np.ndarray, sr: int, num_cordas: int = 5,
                                    duracao_minima: float = 0.10,
                                    tolerancia_overlap: float = 0.03) -> list[dict]:
    """
    Estratégia 4 — Subtração Iterativa (ideia do usuário).
    Processa grave→agudo. Após cada corda, subtrai a componente espectral
    das notas detectadas (fundamental + harmônicos) do sinal residual.
    Cada corda subsequente "vê" o sinal progressivamente mais limpo.
    """
    y_h, _  = _hpss(y)
    cordas  = _get_cordas(num_cordas)
    y_res   = y_h.copy()
    todos   = []

    for corda in cordas:
        rms_ref = float(np.sqrt(np.mean(y_res ** 2)))
        if rms_ref < 1e-8:
            break

        segs = _pyin_uma_corda(y_res, sr, corda)
        todos.extend(segs)

        segs_ok = [s for s in segs if (s["fim"] - s["inicio"]) >= duracao_minima]
        if segs_ok:
            y_res = _subtracao_espectral(y_res, sr, segs_ok)

    return _mesclar_segmentos(todos, tolerancia_overlap, duracao_minima)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DAS ESTRATÉGIAS (usado pelo comparador)
# ══════════════════════════════════════════════════════════════════════════════

ESTRATEGIAS: dict[str, callable] = {
    "1. Base v3":            estrategia_base,
    "2. Remoção Harmônicos": estrategia_remocao_harmonicos,
    "3. Notch Progressivo":  estrategia_notch_progressivo,
    "4. Subtração Iterativa": estrategia_subtracao_iterativa,
}

CORES_ESTRATEGIAS = {
    "1. Base v3":            "#3498db",   # azul
    "2. Remoção Harmônicos": "#e74c3c",   # vermelho
    "3. Notch Progressivo":  "#2ecc71",   # verde
    "4. Subtração Iterativa": "#f39c12",  # laranja
}
