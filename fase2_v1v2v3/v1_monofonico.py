# -*- coding: utf-8 -*-
"""
v1_monofonico.py — Decifrae / Transcrição de Baixo — VERSÃO 1 (MONOFÔNICA)
==========================================================================
Primeira abordagem: um único pYIN sobre o sinal inteiro (banda larga B0–G4),
quantização por arredondamento e segmentação ingênua.

Características (preservadas fielmente para o benchmark):
  - HPSS (margin=2.0) antes do pYIN.
  - Um pYIN no sinal todo, fmin=B0, fmax=G4.
  - Máscara apenas pelo voiced_flag do pYIN.
  - Quantização np.round (sem suavização).
  - Segmentação que troca de nota a cada mudança de semitom.
  - LIMITAÇÃO HISTÓRICA: o filtro de 50 ms só existia no plot; a função
    devolvia TODOS os segmentos (inclusive ruído ultracurto).

Uso:
    from v1_monofonico import transcrever
    segmentos, nomes = transcrever("musica.mp3")
"""

import time
import numpy as np
import librosa

FMIN = librosa.note_to_hz("B0")   # ~30.87 Hz
FMAX = librosa.note_to_hz("G4")   # ~392 Hz
FRAME_LENGTH = 4096
HOP_LENGTH   = 1024


def transcrever(audio_path: str, verbose: bool = True) -> tuple[list[dict], list[str]]:
    """Transcreve um áudio monofonicamente. Retorna (segmentos, nomes_das_notas)."""
    if verbose:
        print(f"\n--- [V1 MONOFÔNICO] Processando: {audio_path.split('/')[-1]} ---")
    y, sr = librosa.load(audio_path)

    y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)

    if verbose:
        print("Executando pYIN (sinal inteiro, banda larga)...")
    t0 = time.time()
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harmonic, fmin=FMIN, fmax=FMAX, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    if verbose:
        print(f"[\u2713] pYIN concluído em {time.time() - t0:.2f}s")

    # Limpeza: só o voiced_flag
    f0_clean = f0.copy()
    f0_clean[~voiced_flag] = np.nan

    # Quantização por arredondamento puro
    midi_q = np.round(librosa.hz_to_midi(f0_clean))
    times  = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)

    # Segmentação ingênua: nova nota a cada mudança de semitom
    segmentos = []
    nota_atual, inicio_idx = None, 0
    for i, nota in enumerate(midi_q):
        if np.isnan(nota):
            if nota_atual is not None:
                segmentos.append({"nota_midi": nota_atual,
                                  "inicio": float(times[inicio_idx]),
                                  "fim": float(times[i])})
                nota_atual = None
        else:
            if nota_atual is None:
                nota_atual, inicio_idx = nota, i
            elif nota != nota_atual:
                segmentos.append({"nota_midi": nota_atual,
                                  "inicio": float(times[inicio_idx]),
                                  "fim": float(times[i])})
                nota_atual, inicio_idx = nota, i
    if nota_atual is not None:
        segmentos.append({"nota_midi": nota_atual,
                          "inicio": float(times[inicio_idx]),
                          "fim": float(times[-1])})

    # ATENÇÃO: nenhuma filtragem de duração aqui (comportamento original).
    nomes = [librosa.midi_to_note(int(s["nota_midi"])) for s in segmentos]
    if verbose:
        print(f"[\u2713] {len(segmentos)} segmentos (sem filtro de duração)")
    return segmentos, nomes


if __name__ == "__main__":
    import sys
    caminho = sys.argv[1] if len(sys.argv) > 1 else "Corda_Sol_G.mp3"
    segs, nomes = transcrever(caminho)
    print("\nNotas detectadas:", nomes)
