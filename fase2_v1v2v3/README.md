# Decifrae — Transcrição de Baixo: V1 / V2 / V3

Três versões do pipeline áudio → tablatura, mais o benchmark comparativo.

## Instalação

```bash
pip install librosa soundfile scipy numpy
# ffmpeg necessário para ler .m4a (ex.: sudo apt install ffmpeg)
```

## Arquivos

| Arquivo | Descrição |
|---|---|
| `v1_monofonico.py` | Versão 1: um pYIN no sinal inteiro (banda larga). Segmentação ingênua. |
| `v2_polifonico.py` | Versão 2: pYIN por corda (5 bandpass) + filtros + correção de oitava + DP de fingering + tab. |
| `v3_viterbi.py`    | Versão 3: V2 com decodificação por Viterbi (HMM de pitch por corda). **Importa `v2_polifonico`.** |
| `benchmark_completo.py` | Roda as três versões nos dois áudios e imprime as tabelas. Salva `resultados_benchmark.json`. |

> `v3_viterbi.py` faz `import v2_polifonico`, então mantenha os dois no mesmo diretório.

## Como rodar

Benchmark completo (coloque `Corda_Sol_G.mp3` e `Recording__4_.m4a` na mesma pasta):

```bash
python benchmark_completo.py
```

Uma versão isolada em um áudio qualquer:

```bash
python v1_monofonico.py minha_musica.mp3
python v2_polifonico.py          # editar o caminho no __main__, ou:
python v3_viterbi.py minha_musica.mp3
```

Uso programático:

```python
from v3_viterbi import pipeline_completo_v3
res = pipeline_completo_v3("minha_musica.mp3", num_cordas=5)
print(res["tab_input"])
```

## Parâmetros do Viterbi (v3)

No topo de `v3_viterbi.py`:

- `_VIT_SIGMA_CENTS` — largura da gaussiana de emissão (proximidade do f0).
- `_VIT_W_STEP` — custo por semitom de salto melódico.
- `_VIT_W_OCTAVE` — custo de salto de oitava (↑ para reprimir mais erro de oitava).
- `_VIT_W_SIL` — custo de entrar/sair do silêncio.
