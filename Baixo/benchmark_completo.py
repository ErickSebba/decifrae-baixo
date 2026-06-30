# -*- coding: utf-8 -*-
"""Benchmark das tres versoes (V1 mono / V2 poli / V3 viterbi) nos audios reais."""
import warnings; warnings.filterwarnings("ignore")
import io, contextlib, time, json
import librosa, numpy as np
import v1_monofonico as v1
import v2_polifonico as v2
import v3_viterbi as v3

def _silenciar(fn, *a, **k):
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, time.time()-t0, buf.getvalue()

def midis(seq, key):
    return np.array([librosa.note_to_midi(s['nome']) if key=='nome'
                     else int(round(s[key])) for s in seq])

def metricas(midi_arr, segs, floor=None):
    out = {}
    out['n'] = len(segs)
    if len(midi_arr)==0:
        return out
    out['curtos'] = sum(1 for s in segs if (s['fim']-s['inicio'])<=0.05)
    durs = [s['fim']-s['inicio'] for s in segs]
    out['dur_media'] = float(np.mean(durs))
    med = np.median(midi_arr)
    out['suboitava'] = int(np.sum(midi_arr <= med-12))   # proxy de erro de oitava
    out['n_notas_distintas'] = int(len(np.unique(midi_arr)))
    if floor is not None:
        out['impossiveis'] = int(np.sum(midi_arr < floor))
        out['pct_imp'] = round(100*out['impossiveis']/len(midi_arr),1)
    return out

ARQ = [("Corda_Sol_G.mp3", 43, "corda Sol (G2)")]

RESULT = {}
for path, floor, desc in ARQ:
    nome = path.split('/')[-1]
    RESULT[nome] = {'desc': desc, 'floor': floor}

    # V1
    (segs1, _), t1, _ = _silenciar(v1.transcrever, path, verbose=False)
    m1 = metricas(midis(segs1,'nota_midi'), segs1, floor); m1['tempo']=round(t1,1)

    # V2
    r2, t2, _ = _silenciar(v2.pipeline_completo, path, num_cordas=5)
    seq2 = r2.get('sequencia', [])
    m2 = metricas(midis(seq2,'nome'), seq2, floor); m2['tempo']=round(t2,1)

    # V3
    r3, t3, _ = _silenciar(v3.pipeline_completo_v3, path, num_cordas=5)
    seq3 = r3.get('sequencia', [])
    m3 = metricas(midis(seq3,'nome'), seq3, floor); m3['tempo']=round(t3,1)

    RESULT[nome]['V1']=m1; RESULT[nome]['V2']=m2; RESULT[nome]['V3']=m3

    print(f"\n{'='*70}\n  {nome}  [{desc}]\n{'='*70}")
    hdr = f"  {'metrica':<22}{'V1':>10}{'V2':>10}{'V3':>10}"
    print(hdr); print("  "+"-"*52)
    chaves = [('tempo','tempo (s)'),('n','notas finais'),('curtos','lixo<50ms'),
              ('suboitava','sub-oitava'),('n_notas_distintas','notas distintas')]
    if floor is not None:
        chaves += [('impossiveis','impossiveis'),('pct_imp','% impossiveis')]
    for k,lbl in chaves:
        print(f"  {lbl:<22}{str(m1.get(k,'-')):>10}{str(m2.get(k,'-')):>10}{str(m3.get(k,'-')):>10}")

with open("resultados_benchmark.json","w") as f:
    json.dump(RESULT, f, indent=2, ensure_ascii=False)
print("\n[salvo] resultados_benchmark.json")
