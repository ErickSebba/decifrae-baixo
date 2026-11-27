# Decifraê — Módulo de Transcrição de Baixo (áudio → tablatura)

Transcrição de **baixo elétrico de áudio para tablatura**: detecta as notas com pYIN, separa o
sinal em bandas por corda, escolhe a posição de cada nota no braço por programação dinâmica
(menor caminho) e escreve a tablatura. É o código de pesquisa do módulo de baixo do
[decifrae.com.br](https://decifrae.com.br) e a base de um artigo sobre **vazamento entre cordas**
(*cross-talk*).

*In English: transcription of electric bass audio into tablature: pYIN pitch detection, per-string
band separation, Viterbi smoothing and dynamic programming to choose the fretboard position of each
note. Evaluated on one recording with objective ground truth (only the G string is played).*

Saída da V3 em `Corda_Sol_G.mp3`, um áudio em que só a corda G é tocada:

```
G|---------------------12-----------------9--7-----9--7-----9--------------9--------
D|------10--12--10--9----------14------------------------9---------12--10-----8--5--
A|--10-----------------------------10--7-----------------------10-------------------
E|-------------------------12-------------------------------------------------------
B|----------------------------------------------9-----------------------------------
```

As notas que aparecem fora da corda G são justamente o vazamento entre cordas que o artigo estuda:
a mesma nota é detectada em mais de uma banda.

## Estrutura

```
Decifra-Baixo/
├── README.md                     ← este arquivo
├── HISTORICO.md                  ← linha do tempo arquivo-a-arquivo (= git log)
├── PARAMETROS.md                 ← todos os parâmetros do pipeline (espaço de busca)
├── docs/
│   ├── dossie_modulo_teste_baixo.md      dossiê técnico, 11 seções
│   ├── Decifrae_Evolucao_V1_V2_V3.docx   análise do benchmark V1→V2→V3
│   └── figuras/comparacao_*.png          saídas do comparador de estratégias
├── fase1_modular/                ← maio/2026 — arquitetura modular + bancada sintética
└── fase2_v1v2v3/                 ← junho/2026 — V1/V2/V3 consolidadas + benchmark real
```

## As duas fases em uma frase cada

| Fase | Pasta | O que é |
|---|---|---|
| **1 — modular** (18–20/05/2026) | `fase1_modular/` | Pipeline em 4 módulos (pYIN → posições → DP → tab), extensão polifônica por corda (módulo 1b), bancada de testes 100 % sintética, suíte de 16 ruídos Lee & Shen, 4 estratégias anti-cross-talk e um comparador que gera figuras. |
| **2 — V1/V2/V3** (30/06/2026) | `fase2_v1v2v3/` | Os mesmos conceitos consolidados em três versões autocontidas para benchmark em **áudio real**: V1 monofônica, V2 polifônica monolítica (bandas por corda + filtros + DP + tab) e V3 = V2 com decodificação de Viterbi por corda. Inclui `benchmark_completo.py` e a saída bruta `resultados_benchmark.json`. |

A fase 2 **não substitui** a fase 1: a bancada sintética, o módulo de ruídos e as estratégias
só existem na fase 1; o Viterbi e o benchmark em áudio real só existem na fase 2.

## Resultado principal (fase 2, `Corda_Sol_G.mp3`)

O áudio contém só a corda G (solta = G2, MIDI 43); qualquer nota abaixo de G2 é fisicamente
impossível — isso dá um ground truth objetivo sem anotação manual.

| Métrica | V1 | V2 | V3 |
|---|---|---|---|
| Notas finais | 44 | 27 | 23 |
| Lixo < 50 ms | 18 | 0 | 0 |
| Sub-oitava | 15 | 1 | 1 |
| **Notas impossíveis (< G2)** | **18 (40,9 %)** | **4 (14,8 %)** | **3 (13,0 %)** |
| Tempo (s) | 3,2 | 3,8 | 3,9 |

Fonte: `fase2_v1v2v3/resultados_benchmark.json`, gerado por `benchmark_completo.py`.

**Achado que motiva o artigo** (docs/Decifrae_Evolucao_V1_V2_V3.docx, §3.3): o ganho do
Viterbi na métrica de oitava é pequeno porque a V2 já resolve o erro de oitava
estruturalmente com as bandas por corda. O **resíduo de 13 % não é erro de oitava — é a
mesma nota aparecendo em mais de uma banda** (vazamento/cross-talk entre cordas adjacentes),
que a decodificação por corda isolada não elimina. Um experimento isolado mostrou que
restringir `fmin` à fundamental da corda zera as notas impossíveis (13,8 % → 0 %), mas só
quando a corda é conhecida de antemão. A frente proposta é a **arbitração global entre
cordas** (Viterbi acoplado ou arbitração por energia harmônica frame a frame).

## Como rodar

```bash
pip install librosa soundfile scipy numpy matplotlib
# ffmpeg é necessário para .m4a
```

**Fase 2 (recomendado para reproduzir os números):**

```bash
cd fase2_v1v2v3
python benchmark_completo.py            # roda V1, V2 e V3 em Corda_Sol_G.mp3 e salva resultados_benchmark.json
python v3_viterbi.py Corda_Sol_G.mp3    # uma versão isolada
python v1_monofonico.py minha_musica.mp3
```

`v3_viterbi.py` faz `import v2_polifonico` — os dois precisam ficar na mesma pasta.
Detalhes em [fase2_v1v2v3/README.md](fase2_v1v2v3/README.md).

**Fase 1 (bancada sintética e estratégias):**

```bash
cd fase1_modular
python pipeline.py                  # pipeline monofônico nos áudios definidos no __main__ (ver abaixo)
python pipeline_polifonico.py       # mesmo, com o módulo 1b polifônico
python modulo_teste_1.py            # suites sintéticas monofônicas (3 suites)
python modulo_teste_2.py            # + suite dos 16 ruídos Lee & Shen
python modulo_teste_polifonico.py   # 5 suites polifônicas (P/R/F1, fantasmas, oitava)
python comparador_estrategias.py    # 4 estratégias × 5 suites → comparacao_*.png no diretório atual
python modulo_pyin_polifonico.py arquivo.mp3 [num_cordas]
```

Os scripts da fase 1 abrem janelas do matplotlib por padrão (`PLOTAR = True` no `__main__`).

## Áudios de teste e o que se sabe deles

| Arquivo | Onde | Ground truth |
|---|---|---|
| `Corda_Sol_G.mp3` | `fase2_v1v2v3/` | Só a corda G tocada. Qualquer nota < G2 (MIDI 43) é erro. **Único áudio com verdade objetiva.** |
| `Money_PinkFloyd.mp3`, `Como_tudo_Deve_Ser.mp3` | não versionados | Trechos de músicas usados nos testes da fase 1. Ficam fora do repositório por direitos autorais; para rodar `pipeline.py`, coloque um áudio seu em `fase1_modular/` e ajuste o caminho no `__main__`. O riff de referência sintético de *Money* continua em `RIFF_MONEY_HZ` (`modulo_teste_1.py`). |
| `Recording__4_.m4a` | **ausente** | Citado no `.docx` e no README da fase 2 (V2 50 → V3 31 notas). Não estava na pasta ao organizar o repositório. |

Toda a validação com gabarito conhecido é **sintética** (gerada em código nos `modulo_teste_*.py`).

## Errata da documentação

- `docs/dossie_modulo_teste_baixo.md` (12/07/2026) foi escrito analisando só os módulos da
  fase 1. Por isso marca a seção 4.4 "Viterbi" como `[NÃO IMPLEMENTADO]` e a seção 6
  "Resultados" como `[NÃO MEDIDO]`. Ambos existem na fase 2: `v3_viterbi.py` e
  `resultados_benchmark.json`. O dossiê foi mantido intacto; use-o com essa ressalva.
- `fase2_v1v2v3/benchmark_baixo_tab.py` é o primeiro benchmark (V1 vs V2), gerado em outro
  ambiente: importa `baixo_tab_novo` (nome anterior de `v2_polifonico.py`) e usa caminhos
  `/mnt/user-data/uploads/`. Não roda aqui; foi mantido porque é a origem dos números de
  `Recording__4_` citados no `.docx`. O benchmark válido é `benchmark_completo.py`.
- `fase2_v1v2v3/README.md` diz que o benchmark roda "nos dois áudios"; a versão commitada
  de `benchmark_completo.py` roda só em `Corda_Sol_G.mp3` (`ARQ` tem uma entrada).

## Histórico

Os scripts estão exatamente como foram escritos; na organização do repositório só a estrutura de
pastas, o `.gitignore` e os documentos são novos. O `git log` reconstrói a progressão real com as datas de modificação originais dos arquivos
(maio → julho/2026). Resumo legível em [HISTORICO.md](HISTORICO.md).

## Material exploratório

`material-exploratorio/` guarda a exploração inicial (jun a set/2026), anterior a este repositório: notebooks de detecção de pitch polifônica do baixo (pYIN por corda) e de FFT com librosa, e scripts MATLAB de CQT e FFT.

## O que aprendi

- **Processamento de áudio do zero:** pYIN para altura, CQT e FFT para análise espectral,
  filtros por banda para separar as cordas e Viterbi para suavizar a sequência de notas.
- **A bancada de testes sintética foi a decisão que mais mudou o resultado.** Com gabarito
  conhecido e 16 tipos de ruído deu para comparar estratégias de forma objetiva, e foi ela que
  mostrou que o erro restante não era de oitava, e sim vazamento entre cordas.
- **Iterar com método:** ajustar cada função até funcionar bem, documentar o resultado de cada
  versão (V1, V2, V3) e manter testes para cada etapa do pipeline.
