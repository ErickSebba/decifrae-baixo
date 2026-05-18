"""
MÓDULO 1 — Detecção de Pitch (pYIN)
====================================
Entrada : caminho de arquivo de áudio (str)
Saída   : segmentos (list[dict]) + plota o gráfico

Cada segmento:
    {
        "nota_midi": float,   # nota MIDI quantizada (ex: 34.0 = A#1)
        "inicio":    float,   # timestamp de início em segundos
        "fim":       float,   # timestamp de fim   em segundos
    }

Como integrar com o Módulo 2:
    from modulo_pyin import analisar_baixo
    segmentos = analisar_baixo("arquivo.mp3", plotar=False)
    # → passe segmentos para bridge_pyin_para_notas()
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time


def analisar_baixo(audio_path: str, plotar: bool = True) -> list[dict]:
    """
    Processa um arquivo de áudio e detecta as notas do baixo via pYIN.

    Parâmetros
    ----------
    audio_path : str
        Caminho para o arquivo de áudio (.mp3 ou .wav).
    plotar : bool
        Se True, exibe o gráfico duplo (espectrograma + piano roll).
        Defina False ao usar como parte do pipeline para não bloquear a execução.

    Retorna
    -------
    list[dict]
        Lista de segmentos de nota detectados, ordenados por tempo.
        Cada dict contém: "nota_midi" (float), "inicio" (float), "fim" (float).
    """
    print(f"\n--- Processando: {audio_path.split('/')[-1]} ---")

    y, sr = librosa.load(audio_path)

    fmin = librosa.note_to_hz('B0')   # ~30.87 Hz
    fmax = librosa.note_to_hz('G4')   # ~392.00 Hz (teto alto para solos/slaps)
    frame_length = 4096
    hop_length   = 1024

    print("Separando componentes harmônicos...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)

    print("Executando pYIN...")
    start_time = time.time()

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y_harmonic,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    print(f"[✓] pYIN concluído em {time.time() - start_time:.2f}s")

    # Remove frames de silêncio/ruído
    f0_clean = f0.copy()
    f0_clean[~voiced_flag] = np.nan

    # Converte para MIDI e quantiza para nota exata
    midi_notes     = librosa.hz_to_midi(f0_clean)
    midi_quantized = np.round(midi_notes)
    times          = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # Agrupa frames consecutivos da mesma nota em segmentos
    segmentos   = []
    nota_atual  = None
    inicio_idx  = 0

    for i, nota in enumerate(midi_quantized):
        if np.isnan(nota):
            if nota_atual is not None:
                segmentos.append({
                    "nota_midi": nota_atual,
                    "inicio":    times[inicio_idx],
                    "fim":       times[i],
                })
                nota_atual = None
        else:
            if nota_atual is None:
                nota_atual = nota
                inicio_idx = i
            elif nota != nota_atual:
                segmentos.append({
                    "nota_midi": nota_atual,
                    "inicio":    times[inicio_idx],
                    "fim":       times[i],
                })
                nota_atual = nota
                inicio_idx = i

    if nota_atual is not None:
        segmentos.append({
            "nota_midi": nota_atual,
            "inicio":    times[inicio_idx],
            "fim":       times[-1],
        })

    if plotar:
        _plotar(audio_path, y, y_harmonic, sr, fmin, fmax, hop_length,
                times, f0_clean, segmentos)

    print(f"[✓] {len(segmentos)} segmentos de nota detectados.")
    return segmentos


# ── Função de bridge (exportada para uso no pipeline) ─────────────────────────

def bridge_pyin_para_notas(segmentos: list[dict],
                            duracao_minima: float = 0.05) -> list[dict]:
    """
    Converte a saída do pYIN para o formato esperado pelo Módulo 2.

    Filtra segmentos muito curtos (ruídos) e converte nota_midi → nome de nota.

    Parâmetros
    ----------
    segmentos      : list[dict]   Saída direta de analisar_baixo().
    duracao_minima : float        Segmentos menores que este valor (segundos)
                                  são descartados. Padrão: 0.05 s (50 ms).

    Retorna
    -------
    list[dict]
        Lista de notas filtradas, cada uma com:
            "nome"   (str)   — ex.: "A#1"
            "inicio" (float) — timestamp de início em segundos
            "fim"    (float) — timestamp de fim em segundos
    """
    resultado = []
    for seg in segmentos:
        duracao = seg["fim"] - seg["inicio"]
        if duracao < duracao_minima:
            continue
        nome = librosa.midi_to_note(int(seg["nota_midi"]))
        resultado.append({
            "nome":   nome,
            "inicio": seg["inicio"],
            "fim":    seg["fim"],
        })
    return resultado


# ── Visualização interna ──────────────────────────────────────────────────────

def _plotar(audio_path, y, y_harmonic, sr, fmin, fmax, hop_length,
            times, f0_clean, segmentos):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5]},
    )
    fig.suptitle(
        f'Transcrição de Baixo: {audio_path.split("/")[-1]}',
        fontsize=16, fontweight="bold",
    )

    # AX1: Espectrograma + pYIN
    C    = np.abs(librosa.cqt(y_harmonic, sr=sr, fmin=fmin,
                               n_bins=60, bins_per_octave=12,
                               hop_length=hop_length))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    librosa.display.specshow(C_db, sr=sr, x_axis="time", y_axis="cqt_hz",
                              fmin=fmin, bins_per_octave=12,
                              hop_length=hop_length, cmap="magma",
                              alpha=0.7, ax=ax1)
    ax1.plot(times, f0_clean, label="Pitch ($f_0$) Detectado",
             color="lime", linewidth=3)
    ax1.set_title("Visão DSP: Espectrograma e Frequência Fundamental", fontsize=12)
    ax1.set_ylabel("Frequência (Hz)")
    ax1.set_ylim(fmin, fmax)
    ax1.legend(loc="upper right")

    # AX2: Piano Roll
    cores     = plt.cm.tab20.colors
    mapa_cores = {}
    idx_cor    = 0
    notas_usadas = set()

    for seg in segmentos:
        nome_nota = librosa.midi_to_note(seg["nota_midi"])
        notas_usadas.add(seg["nota_midi"])
        if nome_nota not in mapa_cores:
            mapa_cores[nome_nota] = cores[idx_cor % len(cores)]
            idx_cor += 1
        start_t  = seg["inicio"]
        duration = seg["fim"] - start_t
        if duration > 0.05:
            rect = mpatches.Rectangle(
                (start_t, seg["nota_midi"] - 0.4), duration, 0.8,
                color=mapa_cores[nome_nota], ec="white", lw=1.5,
            )
            ax2.add_patch(rect)
            if duration > 0.15:
                ax2.text(start_t + duration / 2, seg["nota_midi"], nome_nota,
                         ha="center", va="center", color="black",
                         fontweight="bold", fontsize=9)

    ax2.set_title("Visão do Músico: Transcrição Quantizada (Piano Roll)", fontsize=12)
    ax2.set_ylabel("Nota (MIDI)")
    ax2.set_xlabel("Tempo (segundos)")
    if notas_usadas:
        ax2.set_ylim(min(notas_usadas) - 2, max(notas_usadas) + 2)
        yticks = np.arange(min(notas_usadas) - 1, max(notas_usadas) + 2)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([librosa.midi_to_note(int(y)) for y in yticks])
    ax2.grid(axis="x", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", linestyle=":",  alpha=0.3)
    ax2.set_xlim(0, librosa.get_duration(y=y, sr=sr))

    plt.tight_layout()
    plt.show()
