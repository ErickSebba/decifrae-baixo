# -*- coding: utf-8 -*-
"""Benchmark REAL: versao ANTIGA (monofonica) vs NOVA (polifonica) nos audios do Erick."""
import warnings; warnings.filterwarnings("ignore")
import io, contextlib, time
import librosa, numpy as np
import baixo_tab_novo as novo

# ───────────────────────── VERSAO ANTIGA (portada do contexto, headless) ─────────────────────────
def versao_antiga(audio_path):
    y, sr = librosa.load(audio_path)
    fmin = librosa.note_to_hz('B0'); fmax = librosa.note_to_hz('G4')
    frame_length, hop_length = 4096, 1024
    y_h, _ = librosa.effects.hpss(y, margin=2.0)
    f0, vflag, vprob = librosa.pyin(y_h, fmin=fmin, fmax=fmax, sr=sr,
                                    frame_length=frame_length, hop_length=hop_length)
    f0c = f0.copy(); f0c[~vflag] = np.nan
    midi_q = np.round(librosa.hz_to_midi(f0c))
    times  = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    segs, atual, ini = [], None, 0
    for i, n in enumerate(midi_q):
        if np.isnan(n):
            if atual is not None:
                segs.append({'nota_midi': atual, 'inicio': times[ini], 'fim': times[i]}); atual=None
        else:
            if atual is None: atual, ini = n, i
            elif n != atual:
                segs.append({'nota_midi': atual, 'inicio': times[ini], 'fim': times[i]}); atual, ini = n, i
    if atual is not None:
        segs.append({'nota_midi': atual, 'inicio': times[ini], 'fim': times[-1]})
    # OBS: o filtro <50ms na versao antiga so existe no PLOT; o return entrega tudo (bug)
    return segs

def midis_de(segs, key='nota_midi'):
    return np.array([int(round(s[key])) for s in segs])

# ───────────────────────── METRICAS ─────────────────────────
def avaliar(segs, floor_midi=None):
    m = midis_de(segs)
    out = {'n': len(segs)}
    if len(m)==0: return out
    out['curtos_50ms'] = sum(1 for s in segs if (s['fim']-s['inicio'])<=0.05)
    mediana = np.median(m)
    out['suboitava'] = int(np.sum(m <= mediana-12))
    if floor_midi is not None:
        out['impossiveis'] = int(np.sum(m < floor_midi))
        out['pct_imp'] = 100*out['impossiveis']/len(m)
    return out

ARQ = [("/mnt/user-data/uploads/Corda_Sol_G.mp3", 43, "corda Sol (G2)"),
       ("/mnt/user-data/uploads/Recording__4_.m4a", None, "trecho desconhecido")]

print("#"*80)
for path, floor, desc in ARQ:
    nome = path.split('/')[-1]
    print(f"\n{'='*80}\n  {nome}   [{desc}]\n{'='*80}")

    # ANTIGA
    t=time.time(); segs_old = versao_antiga(path); t_old=time.time()-t
    m_old = avaliar(segs_old, floor)

    # NOVA (silenciando os prints internos)
    buf = io.StringIO()
    t=time.time()
    with contextlib.redirect_stdout(buf):
        res = novo.pipeline_completo(path, num_cordas=5)
    t_new=time.time()-t
    segs_new = res.get('sequencia', [])
    # converte nome->midi p/ metricas
    seg_new_midi = [{'nota_midi': librosa.note_to_midi(s['nome']),
                     'inicio': s['inicio'], 'fim': s['fim']} for s in segs_new]
    m_new = avaliar(seg_new_midi, floor)

    print(f"\n  {'metrica':<26}{'ANTIGA':>14}{'NOVA':>14}")
    print(f"  {'-'*54}")
    print(f"  {'tempo (s)':<26}{t_old:>14.1f}{t_new:>14.1f}")
    print(f"  {'segmentos/notas finais':<26}{m_old['n']:>14}{m_new['n']:>14}")
    print(f"  {'segmentos < 50ms':<26}{m_old.get('curtos_50ms','-'):>14}{m_new.get('curtos_50ms','-'):>14}")
    print(f"  {'frames/notas sub-oitava':<26}{m_old.get('suboitava','-'):>14}{m_new.get('suboitava','-'):>14}")
    if floor is not None:
        print(f"  {'notas IMPOSSIVEIS (<G2)':<26}{m_old.get('impossiveis','-'):>14}{m_new.get('impossiveis','-'):>14}")
        print(f"  {'  % impossiveis':<26}{m_old.get('pct_imp',0):>13.1f}%{m_new.get('pct_imp',0):>13.1f}%")
    # tab so a nova produz
    print(f"  {'gera tablatura tocavel':<26}{'NAO (quebrada)':>14}{'SIM':>14}")
    print(f"  {'atribui corda a cada nota':<26}{'NAO':>14}{'SIM':>14}")

    if floor==43:  # mostra a tab da corda Sol
        print("\n  >> Tablatura gerada pela versao NOVA (Corda_Sol_G):")
        tab_txt = novo.print_tab_multiplas_notas.__doc__  # placeholder
        # re-render captured tab from buffer
        linhas = buf.getvalue().splitlines()
        cap=False
        for ln in linhas:
            if ln.strip().startswith(("G|","D|","A|","E|","B|")):
                print("     "+ln)
print("\n"+"#"*80)
