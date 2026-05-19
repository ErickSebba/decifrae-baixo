"""
MÓDULO DE RUÍDOS — Distúrbios de Lee & Shen (2011) Adaptados para Áudio Musical
=================================================================================
Baseado nos modelos matemáticos de distúrbios de qualidade de energia elétrica
de Lee & Shen (2011), reaproveitados para modelar perturbações em sinais de áudio.

Adaptações em relação ao original (domínio elétrico → domínio musical):
    Original              →  Este módulo
    ─────────────────────────────────────────────────────────────────────
    f  = 60 Hz (rede)     →  f  = frequência da nota musical (ex: 110 Hz para A2)
    sf = 3360 Hz          →  sf = sample rate de áudio (padrão: 22050 Hz)
    cycles = n ciclos     →  duration = duração em segundos
    t1, t2 em segundos    →  t1, t2 em segundos (mantidos iguais)

A matemática de cada modelo é preservada sem alteração.
Apenas os parâmetros globais de contexto foram substituídos.

Analogias musicais de cada distúrbio:
    SAG                   → amplitude cai em uma janela de tempo (corda muffled, 
                            dedo mal posicionado)
    SWELL                 → amplitude sobe em uma janela (crescendo, pressão 
                            crescente no arco)
    INTERRUPTION          → sag extremo (alpha≈1) → nota quase muta, palm mute 
                            pesado
    FLICKER               → modulação de amplitude por frequência baixa → tremolo, 
                            vibrato de amplitude inconsistente, instabilidade de bow
    OSCILLATORY_TRANSIENT → oscilação transitória amortecida adicionada ao sinal 
                            → fret buzz, simpatia de corda solta ressoando
    HARMONIC              → adiciona harmônicos ímpares (3°, 5°, 7°) → distorção 
                            leve, overdrive, clipping suave
    SAG_HARMONIC          → distorção + queda de amplitude (overdrive com mute)
    SWELL_HARMONIC        → distorção + crescimento de amplitude
    NOTCH                 → entalhes periódicos (a cada ciclo da fundamental) → 
                            pickup noise, interferência de RF periódica
    SPIKE                 → picos positivos periódicos → ruído de ataque de palheta,
                            click de traste, pop de cabo
    INTERRUPTION_HARMONIC → mute pesado + distorção
    FLICKER_HARMONIC      → tremolo + distorção
    FLICKER_SAG           → tremolo + mute parcial
    FLICKER_SWELL         → tremolo + crescimento (vibrato expressivo)
    IMPULSIVE_TRANSIENT   → perturbação breve e isolada → bump de cabo, toque 
                            acidental de corda

Uso básico:
    from modulo_ruidos import aplicar_ruido, RuidoMusical, suite_ruidos_musicais
    sinal_perturbado, descricao = aplicar_ruido(sinal, freq_hz=110.0,
                                                tipo=RuidoMusical.FLICKER)

Uso no ambiente de testes:
    from modulo_ruidos import suite_ruidos_musicais
    casos = suite_ruidos_musicais(nota_hz=110.0, duracao=1.5)
    # → passe para rodar_suite() em modulo_teste.py
"""

import numpy as np
from numpy import arange
from numpy import multiply as mult
from enum import Enum
import random
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES PADRÃO
# ══════════════════════════════════════════════════════════════════════════════

SR_AUDIO    = 22050   # sample rate padrão de áudio
AMPLITUDE   = 0.8     # amplitude de pico padrão


# ══════════════════════════════════════════════════════════════════════════════
# ENUM DE TIPOS
# ══════════════════════════════════════════════════════════════════════════════

class RuidoMusical(Enum):
    """
    Enum de todos os tipos de perturbação disponíveis.
    O valor inteiro corresponde ao índice original de Lee & Shen para rastreabilidade.
    """
    NORMAL                = 0
    SAG                   = 1    # queda de amplitude janelada
    SWELL                 = 2    # crescimento de amplitude janelado
    INTERRUPTION          = 3    # mute quase total (sag extremo)
    FLICKER               = 4    # tremolo (modulação AM por senoide lenta)
    OSCILLATORY_TRANSIENT = 5    # ressonância amortecida transitória (fret buzz)
    HARMONIC              = 6    # distorção (harmônicos ímpares)
    SAG_HARMONIC          = 7    # mute + distorção
    SWELL_HARMONIC        = 8    # crescimento + distorção
    NOTCH                 = 9    # entalhes periódicos (pickup noise)
    SPIKE                 = 10   # picos periódicos (click de traste/palheta)
    INTERRUPTION_HARMONIC = 11   # mute total + distorção
    FLICKER_HARMONIC      = 12   # tremolo + distorção
    FLICKER_SAG           = 13   # tremolo + mute parcial
    FLICKER_SWELL         = 14   # tremolo + crescimento
    IMPULSIVE_TRANSIENT   = 15   # bump isolado de cabo/interferência


# Analogia musical para exibição
DESCRICAO_MUSICAL = {
    RuidoMusical.NORMAL:                "Sinal limpo (baseline)",
    RuidoMusical.SAG:                   "Queda de amplitude janelada (dedo mal posicionado / muffled)",
    RuidoMusical.SWELL:                 "Crescimento de amplitude janelado (crescendo / mais pressão)",
    RuidoMusical.INTERRUPTION:          "Mute quase total (palm mute pesado, alpha≈1)",
    RuidoMusical.FLICKER:               "Tremolo / vibrato de amplitude (modulação AM lenta)",
    RuidoMusical.OSCILLATORY_TRANSIENT: "Fret buzz / ressonância parasita amortecida",
    RuidoMusical.HARMONIC:              "Distorção suave (harmônicos ímpares, 3°/5°/7°)",
    RuidoMusical.SAG_HARMONIC:          "Mute + distorção simultâneos",
    RuidoMusical.SWELL_HARMONIC:        "Crescimento + distorção simultâneos",
    RuidoMusical.NOTCH:                 "Pickup noise / interferência RF periódica (entalhes)",
    RuidoMusical.SPIKE:                 "Click de traste / pop de palheta (picos periódicos)",
    RuidoMusical.INTERRUPTION_HARMONIC: "Mute total + distorção",
    RuidoMusical.FLICKER_HARMONIC:      "Tremolo + distorção",
    RuidoMusical.FLICKER_SAG:           "Tremolo + mute parcial",
    RuidoMusical.FLICKER_SWELL:         "Tremolo + crescimento (vibrato expressivo)",
    RuidoMusical.IMPULSIVE_TRANSIENT:   "Bump de cabo / interferência impulsiva isolada",
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES (do original de Lee & Shen — sem modificação matemática)
# ══════════════════════════════════════════════════════════════════════════════

def _u(x: np.ndarray) -> np.ndarray:
    """Função degrau de Heaviside. u(x) = 1 se x > 0, 0 caso contrário."""
    return np.heaviside(x, 0)


def _time_axis(duracao: float, sr: int) -> np.ndarray:
    """
    Gera o eixo de tempo uniforme.
    Adaptação: usa duração em segundos em vez de cycles/f.
    """
    return arange(0, duracao, 1.0 / sr)


# ══════════════════════════════════════════════════════════════════════════════
# GERADORES DE SINAL POR TIPO (matemática original preservada)
# ══════════════════════════════════════════════════════════════════════════════

def _gen_normal(f: float, duracao: float, sr: int, A: float) -> np.ndarray:
    t = _time_axis(duracao, sr)
    return A * np.sin(2 * np.pi * f * t)


def _gen_sag(f: float, duracao: float, sr: int, A: float,
             alpha: float, t1: float, t2: float) -> np.ndarray:
    """
    Queda de amplitude entre t1 e t2.
    alpha: profundidade da queda (0=sem efeito, 1=mute total).
    Eq.: x(t) = A * [1 - alpha*(u(t-t1) - u(t-t2))] * sin(2πft)
    """
    t = _time_axis(duracao, sr)
    envelope = 1 - alpha * (_u(t - t1) - _u(t - t2))
    return A * envelope * np.sin(2 * np.pi * f * t)


def _gen_swell(f: float, duracao: float, sr: int, A: float,
               alpha: float, t1: float, t2: float) -> np.ndarray:
    """
    Crescimento de amplitude entre t1 e t2.
    alpha: magnitude do crescimento.
    Eq.: x(t) = A * [1 + alpha*(u(t-t1) - u(t-t2))] * sin(2πft)
    """
    t = _time_axis(duracao, sr)
    envelope = 1 + alpha * (_u(t - t1) - _u(t - t2))
    return A * envelope * np.sin(2 * np.pi * f * t)


def _gen_interruption(f, duracao, sr, A, alpha, t1, t2) -> np.ndarray:
    """
    Mute quase total (sag com alpha próximo de 1).
    Matematicamente idêntico ao sag — distinção é paramétrica (alpha ≈ 0.9–1.0).
    """
    return _gen_sag(f, duracao, sr, A, alpha, t1, t2)


def _gen_flicker(f: float, duracao: float, sr: int, A: float,
                 alpha: float, beta: float) -> np.ndarray:
    """
    Modulação de amplitude por senoide de baixa frequência beta (Hz).
    Eq.: x(t) = A * [1 + alpha*sin(2π*beta*t)] * sin(2πft)
    alpha: profundidade da modulação (0.1–0.2 no original).
    beta:  frequência do tremolo em Hz (5–20 Hz no elétrico; 2–8 Hz musical).
    """
    t = _time_axis(duracao, sr)
    modulator = 1 + alpha * np.sin(2 * np.pi * beta * t)
    return A * modulator * np.sin(2 * np.pi * f * t)


def _gen_oscillatory_transient(f: float, duracao: float, sr: int, A: float,
                                 alpha: float, tau: float,
                                 fn: float, t1: float, t2: float) -> np.ndarray:
    """
    Transiente oscilatório amortecido adicionado ao sinal base.
    Modela fret buzz ou simpatia de corda ressoando brevemente após o ataque.
    Eq.: x(t) = A*sin(2πft) + alpha*exp(-(t-t1)/tau) * [u(t-t1)-u(t-t2)] * sin(2πfn*t)

    fn:  frequência da oscilação parasita (Hz) — ex: fn próxima de um harmônico da nota
    tau: constante de decaimento do transiente (segundos)
    """
    t    = _time_axis(duracao, sr)
    base = A * np.sin(2 * np.pi * f * t)
    c    = alpha * np.exp(-((t - t1) / tau))
    d    = _u(t - t1) - _u(t - t2)
    e    = np.sin(2 * np.pi * fn * t)
    transiente = mult(mult(c, d), e)
    y = base + transiente
    # Normaliza para não ultrapassar amplitude alvo
    pico = np.max(np.abs(y))
    return y / pico * A if pico > 0 else y


def _gen_harmonic(f: float, duracao: float, sr: int, A: float,
                   alpha: list[float]) -> np.ndarray:
    """
    Adiciona harmônicos ímpares (3°, 5°, 7°) ao sinal fundamental.
    Modela distorção suave, overdrive leve, ou timbre rico em harmônicos.
    Eq.: x(t) = a1*sin(2πft) + alpha[0]*sin(3*2πft) + alpha[1]*sin(5*2πft) + ...

    a1 é calculado para manter a norma total unitária (igual ao original).
    alpha: lista de amplitudes dos harmônicos [a3, a5, a7].
    """
    t  = _time_axis(duracao, sr)
    a1 = np.sqrt(1 - sum(a**2 for a in alpha))
    s  = a1 * np.sin(2 * np.pi * f * t)
    for k, a in enumerate(alpha):
        ordem = 2 * (k + 1) + 1   # 3, 5, 7
        s += a * np.sin(f * 2 * np.pi * t * ordem)
    pico = np.max(np.abs(s))
    return A * s / pico if pico > 0 else s


def _gen_sag_harmonic(f, duracao, sr, A, alpha_s, t1, t2,
                       alpha) -> np.ndarray:
    """SAG aplicado sobre sinal com harmônicos."""
    t         = _time_axis(duracao, sr)
    envelope  = 1 - alpha_s * (_u(t - t1) - _u(t - t2))
    h         = _gen_harmonic(f, duracao, sr, A=1.0, alpha=alpha)
    y         = envelope * h
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_swell_harmonic(f, duracao, sr, A, alpha_s, t1, t2,
                         alpha) -> np.ndarray:
    """SWELL aplicado sobre sinal com harmônicos."""
    t        = _time_axis(duracao, sr)
    envelope = 1 + alpha_s * (_u(t - t1) - _u(t - t2))
    h        = _gen_harmonic(f, duracao, sr, A=1.0, alpha=alpha)
    y        = envelope * h
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_notch(f: float, duracao: float, sr: int, A: float,
               k: float, t1: float, t2: float) -> np.ndarray:
    """
    Entalhes (notches) periódicos sincronizados com a fundamental.
    Cada ciclo da nota recebe um entalhe negativo na janela [t1, t2].
    Modela interferência de pickup, hum de aterramento periódico.
    Eq.: x(t) = A*sin(2πft) - k * Σ[u(t-(t1+n/f)) - u(t-(t2+n/f))] * sign(sin(2πft))
    """
    t      = _time_axis(duracao, sr)
    base   = np.sin(2 * np.pi * f * t)
    notch_ = np.zeros(len(t))
    n_ciclos = int(duracao * f) + 1
    for n in range(n_ciclos):
        notch_ += k * (_u(t - (t1 + n / f)) - _u(t - (t2 + n / f)))
    y = base + np.multiply(notch_, -np.sign(base))
    return A * y


def _gen_spike(f: float, duracao: float, sr: int, A: float,
               k: float, t1: float, t2: float) -> np.ndarray:
    """
    Picos positivos periódicos sincronizados com a fundamental.
    Modela click de traste, pop de palheta, ruído de contato.
    Eq.: x(t) = A*sin(2πft) + k * Σ[u(t-(t1+n/f)) - u(t-(t2+n/f))] * sign(sin(2πft))
    """
    t      = _time_axis(duracao, sr)
    base   = np.sin(2 * np.pi * f * t)
    spike_ = np.zeros(len(t))
    n_ciclos = int(duracao * f) + 1
    for n in range(n_ciclos):
        spike_ += k * (_u(t - (t1 + n / f)) - _u(t - (t2 + n / f)))
    y = base + np.multiply(spike_, +np.sign(base))
    return A * y


def _gen_interruption_harmonic(f, duracao, sr, A, alpha_i, t1, t2,
                                alpha) -> np.ndarray:
    """INTERRUPTION (mute quase total) + harmônicos ímpares."""
    t        = _time_axis(duracao, sr)
    envelope = 1 - alpha_i * (_u(t - t1) - _u(t - t2))
    h        = _gen_harmonic(f, duracao, sr, A=1.0, alpha=alpha)
    y        = envelope * h
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_flicker_harmonic(f, duracao, sr, A, alpha_f, beta_f,
                           alpha) -> np.ndarray:
    """FLICKER (tremolo) + harmônicos ímpares."""
    t        = _time_axis(duracao, sr)
    modulator = 1 + alpha_f * np.sin(2 * np.pi * beta_f * t)
    h         = _gen_harmonic(f, duracao, sr, A=1.0, alpha=alpha)
    y         = modulator * h
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_flicker_sag(f, duracao, sr, A, alpha, t1, t2,
                      alpha_f, beta_f) -> np.ndarray:
    """FLICKER (tremolo) aplicado sobre sag."""
    t         = _time_axis(duracao, sr)
    sag_env   = 1 - alpha * (_u(t - t1) - _u(t - t2))
    fl_signal = _gen_flicker(f, duracao, sr, A=1.0, alpha=alpha_f, beta=beta_f)
    y         = sag_env * fl_signal
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_flicker_swell(f, duracao, sr, A, alpha, t1, t2,
                        alpha_f, beta_f) -> np.ndarray:
    """FLICKER (tremolo) aplicado sobre swell."""
    t          = _time_axis(duracao, sr)
    swell_env  = 1 + alpha * (_u(t - t1) - _u(t - t2))
    fl_signal  = _gen_flicker(f, duracao, sr, A=1.0, alpha=alpha_f, beta=beta_f)
    y          = swell_env * fl_signal
    pico = np.max(np.abs(y))
    return A * y / pico if pico > 0 else y


def _gen_impulsive_transient(f, duracao, sr, A, alpha, t1, t2) -> np.ndarray:
    """
    Perturbação impulsiva breve e isolada (matematicamente idêntico ao sag).
    Distingue-se parametricamente: janela muito curta (t2-t1 pequeno).
    Modela bump de cabo, clique de conector, interferência isolada.
    """
    return _gen_sag(f, duracao, sr, A, alpha, t1, t2)


# ══════════════════════════════════════════════════════════════════════════════
# GERADOR DE PARÂMETROS ALEATÓRIOS (adaptado para áudio musical)
# ══════════════════════════════════════════════════════════════════════════════

class ParamsAleatorios:
    """
    Gera parâmetros aleatórios válidos para cada tipo de perturbação,
    adaptados para duração e frequências de áudio musical.

    Diferenças em relação ao RandomParams original:
        - t1, t2 baseados em `duracao` (segundos) em vez de cycles/f
        - beta (flicker) reduzido para 2–8 Hz (tremolo musical realista)
        - fn (oscillatory transient) baseado em harmônicos da nota musical
        - tau escalado para duração do áudio
    """

    def __init__(self, tipo: RuidoMusical, f: float, duracao: float):
        self.f       = f
        self.duracao = duracao

        # Janela de tempo para perturbações janeladas
        # Garante que t1 e t2 estejam dentro da duração do sinal
        tamanho = random.uniform(0.2, 0.6) * duracao
        self.t1 = random.uniform(0.05 * duracao, (duracao - tamanho) * 0.95)
        self.t2 = self.t1 + tamanho

        if tipo == RuidoMusical.SAG:
            self.alpha = random.uniform(0.1, 0.9)

        elif tipo == RuidoMusical.SWELL:
            self.alpha = random.uniform(0.1, 0.8)

        elif tipo == RuidoMusical.INTERRUPTION:
            self.alpha = random.uniform(0.9, 1.0)

        elif tipo == RuidoMusical.FLICKER:
            # beta musical: 2–8 Hz (tremolo de instrumento)
            # O original usava 5–20 Hz que é adequado para elétrico mas rápido demais para áudio
            self.alpha = random.uniform(0.05, 0.25)
            self.beta  = random.uniform(2.0, 8.0)

        elif tipo == RuidoMusical.OSCILLATORY_TRANSIENT:
            # fn: próximo de um harmônico da nota (2x, 3x, 4x a fundamental)
            # Simula fret buzz que ressoa em uma frequência relacionada
            self.alpha = random.uniform(0.1, 0.5)
            self.fn    = f * random.choice([2, 3, 4, 5])
            # tau: decai em 5–20% da duração total
            self.tau   = random.uniform(0.05, 0.2) * duracao
            # Janela menor para o transiente
            self.t1    = random.uniform(0.1, 0.4) * duracao
            self.t2    = self.t1 + random.uniform(0.05, 0.15) * duracao

        elif tipo == RuidoMusical.HARMONIC:
            self.alpha = [
                random.uniform(0.05, 0.20),
                random.uniform(0.03, 0.15),
                random.uniform(0.02, 0.10),
            ]

        elif tipo in (RuidoMusical.SAG_HARMONIC, RuidoMusical.SWELL_HARMONIC):
            self.alpha_s = random.uniform(0.1, 0.7)
            self.alpha   = [
                random.uniform(0.05, 0.20),
                random.uniform(0.03, 0.15),
                random.uniform(0.02, 0.10),
            ]

        elif tipo in (RuidoMusical.NOTCH, RuidoMusical.SPIKE):
            self.k  = random.uniform(0.05, 0.3)
            # Janela do notch/spike: fração muito pequena do período da nota
            periodo = 1.0 / f
            self.t1 = random.uniform(0, 0.3 * periodo)
            self.t2 = self.t1 + random.uniform(0.01, 0.08) * periodo

        elif tipo == RuidoMusical.INTERRUPTION_HARMONIC:
            self.alpha_i = random.uniform(0.9, 1.0)
            self.alpha   = [
                random.uniform(0.05, 0.20),
                random.uniform(0.03, 0.15),
                random.uniform(0.02, 0.10),
            ]

        elif tipo == RuidoMusical.FLICKER_HARMONIC:
            self.alpha_f = random.uniform(0.05, 0.20)
            self.beta_f  = random.uniform(2.0, 8.0)
            self.alpha   = [
                random.uniform(0.05, 0.20),
                random.uniform(0.03, 0.15),
                random.uniform(0.02, 0.10),
            ]

        elif tipo in (RuidoMusical.FLICKER_SAG, RuidoMusical.FLICKER_SWELL):
            self.alpha   = random.uniform(0.1, 0.7)
            self.alpha_f = random.uniform(0.05, 0.20)
            self.beta_f  = random.uniform(2.0, 8.0)

        elif tipo == RuidoMusical.IMPULSIVE_TRANSIENT:
            self.alpha = random.uniform(0.3, 0.8)
            # Janela muito curta: 1–5% da duração
            self.t1 = random.uniform(0.1, 0.8) * duracao
            self.t2 = self.t1 + random.uniform(0.01, 0.05) * duracao


# ══════════════════════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def gerar_sinal_perturbado(tipo: RuidoMusical,
                            freq_hz: float,
                            duracao: float,
                            sr: int = SR_AUDIO,
                            amplitude: float = AMPLITUDE,
                            params: ParamsAleatorios = None) -> tuple[np.ndarray, str]:
    """
    Gera um sinal com a perturbação especificada a partir de parâmetros aleatórios.

    Parâmetros
    ----------
    tipo      : RuidoMusical   tipo de perturbação (enum)
    freq_hz   : float          frequência fundamental da nota em Hz
    duracao   : float          duração do sinal em segundos
    sr        : int            sample rate em Hz
    amplitude : float          amplitude de pico (0.0 – 1.0)
    params    : ParamsAleatorios  se None, gera parâmetros aleatórios automaticamente

    Retorna
    -------
    (sinal: np.ndarray, descricao: str)
    """
    if params is None:
        params = ParamsAleatorios(tipo, freq_hz, duracao)

    p = params
    f = freq_hz
    A = amplitude
    d = duracao

    if tipo == RuidoMusical.NORMAL:
        y = _gen_normal(f, d, sr, A)

    elif tipo == RuidoMusical.SAG:
        y = _gen_sag(f, d, sr, A, p.alpha, p.t1, p.t2)

    elif tipo == RuidoMusical.SWELL:
        y = _gen_swell(f, d, sr, A, p.alpha, p.t1, p.t2)

    elif tipo == RuidoMusical.INTERRUPTION:
        y = _gen_interruption(f, d, sr, A, p.alpha, p.t1, p.t2)

    elif tipo == RuidoMusical.FLICKER:
        y = _gen_flicker(f, d, sr, A, p.alpha, p.beta)

    elif tipo == RuidoMusical.OSCILLATORY_TRANSIENT:
        y = _gen_oscillatory_transient(f, d, sr, A, p.alpha, p.tau, p.fn, p.t1, p.t2)

    elif tipo == RuidoMusical.HARMONIC:
        y = _gen_harmonic(f, d, sr, A, p.alpha)

    elif tipo == RuidoMusical.SAG_HARMONIC:
        y = _gen_sag_harmonic(f, d, sr, A, p.alpha_s, p.t1, p.t2, p.alpha)

    elif tipo == RuidoMusical.SWELL_HARMONIC:
        y = _gen_swell_harmonic(f, d, sr, A, p.alpha_s, p.t1, p.t2, p.alpha)

    elif tipo == RuidoMusical.NOTCH:
        y = _gen_notch(f, d, sr, A, p.k, p.t1, p.t2)

    elif tipo == RuidoMusical.SPIKE:
        y = _gen_spike(f, d, sr, A, p.k, p.t1, p.t2)

    elif tipo == RuidoMusical.INTERRUPTION_HARMONIC:
        y = _gen_interruption_harmonic(f, d, sr, A, p.alpha_i, p.t1, p.t2, p.alpha)

    elif tipo == RuidoMusical.FLICKER_HARMONIC:
        y = _gen_flicker_harmonic(f, d, sr, A, p.alpha_f, p.beta_f, p.alpha)

    elif tipo == RuidoMusical.FLICKER_SAG:
        y = _gen_flicker_sag(f, d, sr, A, p.alpha, p.t1, p.t2, p.alpha_f, p.beta_f)

    elif tipo == RuidoMusical.FLICKER_SWELL:
        y = _gen_flicker_swell(f, d, sr, A, p.alpha, p.t1, p.t2, p.alpha_f, p.beta_f)

    elif tipo == RuidoMusical.IMPULSIVE_TRANSIENT:
        y = _gen_impulsive_transient(f, d, sr, A, p.alpha, p.t1, p.t2)

    else:
        raise ValueError(f"Tipo de ruído desconhecido: {tipo}")

    descricao = DESCRICAO_MUSICAL.get(tipo, str(tipo))
    return y.astype(np.float32), descricao


def aplicar_ruido(sinal: np.ndarray, freq_hz: float,
                   tipo: RuidoMusical,
                   sr: int = SR_AUDIO,
                   params: ParamsAleatorios = None) -> tuple[np.ndarray, str]:
    """
    Aplica uma perturbação sobre um sinal existente, em vez de gerar do zero.
    O sinal perturbado é gerado com os mesmos parâmetros de duração e SR do original.

    Parâmetros
    ----------
    sinal   : np.ndarray    sinal original (será substituído, não somado)
    freq_hz : float         frequência da nota em Hz (necessário para parametrização)
    tipo    : RuidoMusical  tipo de perturbação
    sr      : int           sample rate
    params  : opcional      parâmetros fixos (None = aleatório)

    Retorna
    -------
    (sinal_perturbado: np.ndarray, descricao: str)
    """
    duracao = len(sinal) / sr
    amplitude = float(np.max(np.abs(sinal))) or AMPLITUDE
    y, desc = gerar_sinal_perturbado(tipo, freq_hz, duracao, sr, amplitude, params)
    # Ajusta tamanho para coincidir exatamente com o sinal original
    n = len(sinal)
    if len(y) >= n:
        return y[:n], desc
    else:
        return np.pad(y, (0, n - len(y))), desc


# ══════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTES (interface com modulo_teste.py)
# ══════════════════════════════════════════════════════════════════════════════

def suite_ruidos_musicais(nota_hz: float = 110.0,
                           nota_nome: str = "A2",
                           duracao: float = 1.5,
                           sr: int = SR_AUDIO,
                           seed: int = 42) -> list[tuple]:
    """
    Gera uma suite completa de casos de teste com todos os tipos de perturbação.
    Formato compatível com rodar_suite() de modulo_teste.py.

    Parâmetros
    ----------
    nota_hz   : frequência da nota de referência
    nota_nome : nome da nota (para o gabarito)
    duracao   : duração de cada sinal em segundos
    sr        : sample rate
    seed      : semente aleatória para reprodutibilidade

    Retorna
    -------
    list de (audio: np.ndarray, gabarito: list[dict], nome_do_teste: str)
    Compatível diretamente com rodar_suite() de modulo_teste.py.
    """
    random.seed(seed)
    np.random.seed(seed)

    gabarito_base = [{
        "freq_hz":   nota_hz,
        "nota_nome": nota_nome,
        "inicio":    0.0,
        "fim":       duracao,
    }]

    casos = []
    for tipo in RuidoMusical:
        try:
            audio, descricao = gerar_sinal_perturbado(tipo, nota_hz, duracao, sr)
            nome = f"{nota_nome} | {descricao}"
            casos.append((audio, gabarito_base, nome))
        except Exception as e:
            print(f"[AVISO] Falha ao gerar {tipo.name}: {e}")

    return casos


def listar_ruidos() -> None:
    """Imprime tabela de todos os tipos de perturbação com suas analogias musicais."""
    print("\n" + "═" * 70)
    print("  Perturbações Disponíveis (Lee & Shen 2011 → domínio musical)")
    print("═" * 70)
    for tipo, desc in DESCRICAO_MUSICAL.items():
        print(f"  {tipo.value:>2}. {tipo.name:<25} → {desc}")
    print("═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO STANDALONE (visualização dos sinais gerados)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    listar_ruidos()

    NOTA_HZ   = 110.0    # A2
    DURACAO   = 1.5
    SR        = SR_AUDIO

    # Gera todos os sinais e plota em grade
    tipos = list(RuidoMusical)
    n     = len(tipos)
    cols  = 4
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5))
    fig.suptitle(f"Perturbações de Lee & Shen (2011) — A2 ({NOTA_HZ} Hz), {DURACAO}s",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for idx, tipo in enumerate(tipos):
        ax = axes_flat[idx]
        try:
            audio, descricao = gerar_sinal_perturbado(tipo, NOTA_HZ, DURACAO, SR)
            t = np.linspace(0, DURACAO, len(audio))
            ax.plot(t, audio, linewidth=0.6, color="#2c7bb6")
            ax.set_title(f"{tipo.name}", fontsize=8, fontweight="bold")
            ax.set_xlabel(descricao[:35], fontsize=6, color="#555")
            ax.set_ylim(-1.1, 1.1)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Erro:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")
            ax.set_title(tipo.name, fontsize=8)

    # Oculta subplots vazios
    for idx in range(len(tipos), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
