# -*- coding: utf-8 -*-
"""
v3_viterbi.py — Decifrae / Transcrição de Baixo — VERSÃO 3 (POLIFÔNICO + VITERBI)
================================================================================
V3 = V2 (pYIN por corda + filtros + DP de fingering) com UMA mudança central:

    A limpeza heurística por frame da V2 (voiced_prob + máscara de energia +
    validação harmônica + correção de oitava GREEDY) é substituída, em cada
    corda, por uma DECODIFICAÇÃO DE VITERBI sobre um HMM de pitch.

Por que Viterbi?
  - Os erros de oitava da V2 são SUSTENTADOS, não picos isolados — um filtro
    de mediana não resolve, mas um custo de transição que penaliza saltos de
    ±12 semitons sim.
  - Vazamento entre bandas vira "ruído de baixa energia" — o estado de
    SILÊNCIO do HMM vence esses frames, suprimindo notas-fantasma.

Modelo (por corda):
  Estados  : {silêncio} ∪ {MIDI de corda_aberta .. corda_aberta+24}
  Emissão  : proximidade do f0 do pYIN (gaussiana em cents) ponderada por
             voiced_prob + energia espectral na fundamental e no 2º harmônico.
  Transição: ficar = 0 ; passo pequeno = w_step·Δ ; OITAVA (±12/±24) = w_oct ;
             entrar/sair do silêncio = w_sil.

Reaproveita TODO o restante da V2 (bandpass, mesclagem, filtros de
onset/harmônico/corda-errada, DP, render de tab).

Uso:
    from v3_viterbi import pipeline_completo_v3
    resultado = pipeline_completo_v3("musica.mp3", num_cordas=5)
"""

import time
import numpy as np
import librosa

import v2_polifonico as v2   # reaproveita toda a infraestrutura da V2


# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros do HMM (ajustáveis)
# ─────────────────────────────────────────────────────────────────────────────
_VIT_SIGMA_CENTS = 70.0    # largura da gaussiana de emissão (proximidade f0)
_VIT_W_STEP      = 0.55    # custo por semitom de salto melódico
_VIT_W_OCTAVE    = 7.0     # custo de salto de oitava (penaliza erro de oitava)
_VIT_W_SIL       = 3.2     # custo de entrar/sair do silêncio
_VIT_W_EMIT      = 1.0     # peso global da emissão
_VIT_N_TRASTES   = 24      # alcance em semitons a partir da corda solta

_MIDI_CORDA_ABERTA = {"B": 23, "E": 28, "A": 33, "D": 38, "G": 43}


def _viterbi_pitch(f0: np.ndarray, vprob: np.ndarray,
                   S: np.ndarray, freqs: np.ndarray,
                   midi_aberta: int,
                   n_trastes: int = _VIT_N_TRASTES) -> np.ndarray:
    """
    Decodifica a linha monofônica mais provável de UMA corda.
    Retorna um vetor MIDI (float) com np.nan onde o caminho ótimo é silêncio.
    """
    T = len(f0)
    if T == 0:
        return np.array([])

    pitches  = np.arange(midi_aberta, midi_aberta + n_trastes + 1)  # estados de nota
    P        = len(pitches)
    SIL      = P                      # índice do estado de silêncio
    nstates  = P + 1
    pitch_hz = 440.0 * 2.0 ** ((pitches - 69) / 12.0)

    # bins da FFT para fundamental e 2º harmônico de cada estado
    bins_f0 = np.array([int(np.argmin(np.abs(freqs - h))) for h in pitch_hz])
    bins_h2 = np.array([int(np.argmin(np.abs(freqs - 2 * h))) for h in pitch_hz])
    S_max   = float(np.max(S)) + 1e-9

    # ── Matriz de emissão (custo = -log verossimilhança) ─────────────────────
    emit = np.zeros((T, nstates))
    nS = S.shape[1]
    for i in range(T):
        col      = S[:, min(i, nS - 1)]
        col_max  = float(np.max(col)) + 1e-9
        f0i      = f0[i]
        tem_f0   = np.isfinite(f0i) and f0i > 0
        for k in range(P):
            e = float(col[bins_f0[k]]) + 0.5 * float(col[bins_h2[k]])
            e_norm = e / col_max
            if tem_f0:
                cents = abs(1200.0 * np.log2(f0i / pitch_hz[k]))
                prox  = np.exp(-(cents / _VIT_SIGMA_CENTS) ** 2)
            else:
                prox = 0.0
            score      = _VIT_W_EMIT * (0.55 * prox * vprob[i] + 0.45 * e_norm)
            emit[i, k] = -np.log(score + 1e-3)
        # silêncio: barato quando não há voz nem energia local relevante
        sil_score   = (1.0 - vprob[i]) * (1.0 - col_max / S_max)
        emit[i, SIL] = -np.log(sil_score + 1e-3)

    # ── Custos de transição (matriz nstates × nstates) ───────────────────────
    trans = np.zeros((nstates, nstates))
    for a in range(nstates):
        for b in range(nstates):
            if a == SIL and b == SIL:
                trans[a, b] = 0.0
            elif a == SIL or b == SIL:
                trans[a, b] = _VIT_W_SIL
            else:
                d = abs(pitches[a] - pitches[b])
                if d == 0:
                    trans[a, b] = 0.0
                elif d in (12, 24):
                    trans[a, b] = _VIT_W_OCTAVE
                else:
                    trans[a, b] = _VIT_W_STEP * d

    # ── Viterbi (programação dinâmica) ───────────────────────────────────────
    delta = np.empty((T, nstates))
    psi   = np.zeros((T, nstates), dtype=int)
    delta[0] = emit[0]
    for i in range(1, T):
        for b in range(nstates):
            custos      = delta[i - 1] + trans[:, b]
            melhor      = int(np.argmin(custos))
            psi[i, b]   = melhor
            delta[i, b] = custos[melhor] + emit[i, b]

    # backtracking
    caminho      = np.empty(T, dtype=int)
    caminho[-1]  = int(np.argmin(delta[-1]))
    for i in range(T - 2, -1, -1):
        caminho[i] = psi[i + 1, caminho[i + 1]]

    # converte estados em MIDI (silêncio -> NaN)
    midi = np.full(T, np.nan)
    for i, est in enumerate(caminho):
        if est != SIL:
            midi[i] = pitches[est]
    return midi


def _pyin_uma_corda_v3(y_harmonic: np.ndarray, sr: int,
                       corda: dict, rms_ref: float,
                       duracao_minima: float = 0.10) -> list[dict]:
    """Igual à V2 até o pYIN; depois decodifica por Viterbi em vez de heurística."""
    fmin, fmax, nome = corda["fmin"], corda["fmax"], corda["nome"]
    frame_length, hop_length = v2._frame_params(fmin)

    # ── Pré-processamento idêntico ao da V2 ──────────────────────────────────
    y_band = v2._filtro_passa_banda(y_harmonic, sr, fmin, fmax)
    y_band = np.nan_to_num(y_band, nan=0.0, posinf=0.0, neginf=0.0)
    y_band = v2._spectral_gate(y_band, sr, n_fft=max(frame_length, 2048),
                               hop_length=hop_length)
    y_band = v2._wiener_denoise(y_band)

    rms_canal = float(np.sqrt(np.mean(y_band ** 2)))
    limiar    = max(rms_ref * v2._RMS_REF_FACTOR.get(nome, 0.001), 1e-8)
    if rms_canal < limiar:
        return []

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_band, fmin=fmin * 0.88, fmax=fmax * 1.08,
        sr=sr, frame_length=frame_length, hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # voiced_prob zerado onde o pYIN não tem voz (entra como evidência fraca)
    vprob = np.nan_to_num(voiced_probs, nan=0.0)
    vprob = np.where(voiced_flag, vprob, vprob * 0.3)

    # STFT do canal para a emissão do HMM
    n_fft = max(4096, hop_length * 8)
    S     = np.abs(librosa.stft(y_band, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # alinha tamanhos
    n = min(len(f0), S.shape[1])
    f0, vprob, times, S = f0[:n], vprob[:n], times[:n], S[:, :n]

    midi_aberta = _MIDI_CORDA_ABERTA.get(nome, int(round(librosa.hz_to_midi(fmin))))
    midi_q = _viterbi_pitch(f0, vprob, S, freqs, midi_aberta)

    return v2._gerar_segmentos(times, midi_q, nome, duracao_minima,
                               sr, hop_length, y_band=y_band)


def analisar_baixo_polifonico_v3(audio_path: str, num_cordas: int = 5,
                                 duracao_minima: float = 0.10,
                                 tolerancia_overlap: float = 0.03) -> list[dict]:
    """Versão V3 de analisar_baixo_polifonico: pYIN + Viterbi por corda."""
    print(f"\n--- [V3 VITERBI] Processando: {audio_path.split('/')[-1]} ---")
    y, sr         = librosa.load(audio_path)
    print("Separando componentes harmônicos...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
    rms_ref       = float(np.sqrt(np.mean(y_harmonic ** 2)))

    cordas     = v2.CORDAS_5 if num_cordas == 5 else v2.CORDAS_4
    todos_segs = []
    for corda in cordas:
        print(f"  [pYIN+Viterbi] Corda {corda['nome']} ({corda['nota']}) "
              f"[{corda['fmin']:.0f}\u2013{corda['fmax']:.0f} Hz]...", end=" ", flush=True)
        t0   = time.time()
        segs = _pyin_uma_corda_v3(y_harmonic, sr, corda, rms_ref,
                                  duracao_minima=duracao_minima)
        print(f"{len(segs)} segmentos  ({time.time()-t0:.1f}s)")
        todos_segs.extend(segs)

    segmentos = v2._mesclar_segmentos(todos_segs, tolerancia_overlap, duracao_minima)
    print(f"\n[\u2713] {len(segmentos)} segmentos após mesclagem")
    return segmentos


def pipeline_completo_v3(audio_path: str, num_cordas: int = 5,
                         duracao_minima: float = 0.05,
                         **kwargs) -> dict:
    """
    Pipeline completo da V3: usa a detecção com Viterbi e reaproveita
    limpeza, bridges, DP e render de tab da V2.
    """
    print("\n" + "=" * 60)
    print("  PIPELINE: Áudio → Tablatura [V3: POLIFÔNICO + VITERBI]")
    print("=" * 60)

    print("\n[1/4] Detecção pYIN + Viterbi por corda...")
    segmentos_raw = analisar_baixo_polifonico_v3(
        audio_path, num_cordas=num_cordas, duracao_minima=duracao_minima)
    print(f"      → {len(segmentos_raw)} segmentos brutos.")

    print("\n[1.5/4] Pré-processamento (reaproveitado da V2)...")
    segmentos = v2.limpar_segmentos(segmentos_raw)
    print(f"        → {len(segmentos)} segmentos limpos "
          f"({len(segmentos_raw) - len(segmentos)} removidos).")

    sequencia = v2.bridge_pyin_para_notas_poli(segmentos, duracao_minima=duracao_minima)
    print(f"      → {len(sequencia)} notas após filtragem.")
    if not sequencia:
        print("[ERRO] Nenhuma nota detectada.")
        return {}

    print("\n[2/4] Calculando posições no braço...")
    nomes_unicos  = list(dict.fromkeys(n["nome"] for n in sequencia))
    posicoes_dict = v2.calcular_casas_baixo(nomes_unicos)

    notas_dp = v2.bridge_notas_para_dp(sequencia, posicoes_dict,
                                       preferir_corda_detectada=True)
    if not notas_dp:
        print("[ERRO] Nenhuma nota com posição válida para o DP.")
        return {}

    print(f"\n[3/4] Menor caminho entre {len(notas_dp)} notas (DP)...")
    custo_total, caminho = v2.menor_caminho_DP(notas_dp, preferir_corda_detectada=True)
    print(f"      → Custo total: {custo_total:.1f}")

    tab_input = v2.bridge_caminho_para_tab(caminho)
    print(f"\n[4/4] Tablatura ({num_cordas} cordas):\n")
    v2.print_tab_multiplas_notas(tab_input, num_cordas=num_cordas)

    print("\n" + "=" * 60 + "\n  Pipeline V3 concluído.\n" + "=" * 60 + "\n")
    return {
        "segmentos_raw": segmentos_raw, "segmentos": segmentos,
        "sequencia": sequencia, "posicoes_dict": posicoes_dict,
        "notas_dp": notas_dp, "custo_total": custo_total,
        "caminho": caminho, "tab_input": tab_input,
    }


if __name__ == "__main__":
    import sys
    caminho = sys.argv[1] if len(sys.argv) > 1 else "Corda_Sol_G.mp3"
    pipeline_completo_v3(caminho, num_cordas=5)
