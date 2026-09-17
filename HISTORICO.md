# Histórico do módulo de baixo — linha do tempo reconstruída

O projeto não foi versionado durante o desenvolvimento. Esta linha do tempo foi
reconstruída em 16/09/2026 a partir das **datas de modificação dos arquivos** e do conteúdo
dos docstrings; cada linha corresponde a um commit do `git log`, datado com a data original.
Os caminhos abaixo são os **atuais** (pós-reorganização); nos commits históricos os arquivos
estavam na raiz e em `Baixo/`.

## Fase 1 — arquitetura modular (18–20 de maio de 2026)

| Data | Commit | Arquivos | O que aconteceu |
|---|---|---|---|
| 18/05 10:45 | `ce86e84` | `modulo_pyin.py`, `modulo_caminho.py`, `modulo_tablatura.py` | Módulos 1, 3 e 4. pYIN em banda larga B0–G4 com HPSS `margin=2.0`; DP com custo Manhattan; tab ASCII. |
| 18/05 16:35 | `fd88c44` | `modulo_notas.py`, `pipeline.py`, `Money_PinkFloyd.mp3`, `Como_tudo_Deve_Ser.mp3` | Módulo 2 (nota → posições no braço) e integração dos 4 módulos via *bridges*. Primeiros áudios reais. |
| 19/05 17:10 | `f06cc99` | `modulo_teste_1.py` | Bancada sintética v1: geradores com gabarito, perturbadores e 3 suites (nota única, erro de oitava, riffs). |
| 19/05 20:18 | `04f510c` | `modulo_ruidos.py`, `modulo_teste_2.py` | 16 perturbações de Lee & Shen (2011) adaptadas do domínio elétrico para a frequência da nota; bancada v2 = v1 + suite 4. |
| 19/05 21:07 | `0151b52` | `modulo_teste_polifonico.py`, `pipeline_polifonico.py`, `modulo_pyin_polifonico (1).py` | Virada para polifonia por corda. O `(1).py` era um snapshot "v2.1 — filtro SOS" baixado do navegador, truncado na linha 508; registra o bug do Butterworth `(b, a)` instável em 28–220 Hz que zerava todas as cordas. |
| 20/05 19:31 | `2515923` | `modulo_pyin_polifonico.py` (− `(1).py`) | Módulo 1b consolidado: HPSS 3.0 → passa-banda SOS ordem 3 → pYIN por corda → máscara RMS + voiced_prob + correção de oitava → fusão. Rascunho truncado removido. |
| 20/05 19:38 | `ed8c209` | `estrategias_pyin.py`, `comparador_estrategias.py`, `docs/figuras/comparacao_*.png` | 4 estratégias anti-cross-talk (base, remoção de harmônicos, notch progressivo, subtração espectral iterativa) e o comparador 4 × 5 suites com heatmap/barras/radar. |

**Estado ao fim da fase 1:** pipeline funcional em módulos, validação só sintética, sem
Viterbi, sem números em áudio real. Problema identificado: cross-talk harmônico entre bandas.

## Fase 2 — V1/V2/V3 e benchmark em áudio real (30 de junho de 2026)

| Data | Commit | Arquivos | O que aconteceu |
|---|---|---|---|
| 30/06 16:35–17:06 | `1bf852f` | `v1_monofonico.py`, `v2_polifonico.py`, `v3_viterbi.py`, `Corda_Sol_G.mp3`, `benchmark_baixo_tab.py` | Consolidação em 3 versões autocontidas. V1 = porte fiel do módulo 1 (baseline). V2 = tudo num arquivo (1034 linhas): bandas alargadas, histerese RMS, validação harmônica, spectral gate, filtros pós-detecção, DP com preferência pela corda detectada. V3 = V2 com Viterbi num HMM de pitch por corda. `Corda_Sol_G.mp3` traz o primeiro ground truth objetivo. `benchmark_baixo_tab.py` é o benchmark V1 vs V2 legado (não roda aqui). |
| 30/06 17:18 | `cbeffb8` | `benchmark_completo.py`, `resultados_benchmark.json` | Benchmark das 3 versões. Notas impossíveis: 40,9 % → 14,8 % → 13,0 %. |
| 30/06 20:47 | `7d6e9a0` | `fase2_v1v2v3/README.md`, `docs/Decifrae_Evolucao_V1_V2_V3.docx` | README de execução e análise escrita do benchmark, com o achado do vazamento entre bandas. |

**Estado ao fim da fase 2:** três versões comparáveis, um áudio com verdade objetiva, e o
diagnóstico de que o erro residual é a mesma nota em mais de uma banda.

## Fase 3 — documentação (12 de julho de 2026)

| Data | Commit | Arquivos | O que aconteceu |
|---|---|---|---|
| 12/07 23:27 | `c5cb708` | `docs/dossie_modulo_teste_baixo.md` | Dossiê técnico-científico em 11 seções. Analisa só a fase 1 — ver errata no README. |

## Organização do repositório (16 de setembro de 2026)

| Commit | O que aconteceu |
|---|---|
| reorganização | `git init`; reconstrução dos commits acima com datas originais; `git mv` para `fase1_modular/`, `fase2_v1v2v3/`, `docs/`; `.gitignore`. Removidos do disco sem entrar no git: `PerformanceTest_Windows_x86-64.exe` (86 MB, instalador PassMark) e `Baixo/Arquivos/Money_PinkFloyd.mp3` (duplicata idêntica). |
| documentação | `README.md`, `HISTORICO.md`, `PARAMETROS.md`, `ARQUIVOS_PARA_NOVO_CHAT.md`. |

## Mapa de correspondência fase 1 → fase 2

| Conceito | Fase 1 | Fase 2 |
|---|---|---|
| pYIN monofônico | `modulo_pyin.analisar_baixo()` | `v1_monofonico.transcrever()` |
| pYIN por corda | `modulo_pyin_polifonico.analisar_baixo_polifonico()` | `v2_polifonico.analisar_baixo_polifonico()` (bandas e limiares diferentes — ver PARAMETROS.md) |
| Posições no braço | `modulo_notas.calcular_casas_baixo()` | `v2_polifonico.calcular_casas_baixo()` |
| Caminho mínimo | `modulo_caminho.menor_caminho_DP()` (Manhattan) | `v2_polifonico.menor_caminho_DP()` (`_custo_fingering`, `bonus_corda`) |
| Tablatura | `modulo_tablatura.print_tab_multiplas_notas()` | `v2_polifonico.print_tab_multiplas_notas()` |
| Limpeza de segmentos | — | `v2_polifonico.limpar_segmentos()` (onset, harmônicos, corda errada, gaps) |
| Viterbi | — | `v3_viterbi._viterbi_pitch()` |
| Bancada sintética / ruídos / estratégias | `modulo_teste_*.py`, `modulo_ruidos.py`, `estrategias_pyin.py`, `comparador_estrategias.py` | — |
| Benchmark em áudio real | — | `benchmark_completo.py` |
