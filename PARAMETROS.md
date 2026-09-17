# Parâmetros do pipeline — espaço de busca para a camada de otimização

Todos os valores abaixo foram lidos diretamente do código em 16/09/2026 (linha indicada).
Não há arquivo de configuração: os parâmetros são constantes de módulo ou defaults de
função. Este documento é o inventário para quem for construir a camada de otimização.

## 1. Bandas por corda (o coração do cross-talk)

Afinação de 5 cordas: B0 (MIDI 23), E1 (28), A1 (33), D2 (38), G2 (43). 4 cordas: sem B.

| Corda | Fundamental solta | Fase 1 `modulo_pyin_polifonico.CORDAS_5` (L64) | Fase 2 `v2_polifonico.CORDAS_5` (L44) |
|---|---|---|---|
| B | 30,87 Hz | 28 – 58 Hz | 28 – 78 Hz |
| E | 41,20 Hz | 38 – 82 Hz | 36 – 105 Hz |
| A | 55,00 Hz | 51 – 110 Hz | 55 – 130 Hz |
| D | 73,42 Hz | 68 – 147 Hz | 73 – 175 Hz |
| G | 98,00 Hz | 92 – 220 Hz | 98 – 250 Hz |

Observações:
- A fase 1 limitou cada banda a ~1 oitava (12ª casa) para forçar a fundamental. A fase 2
  alargou os tetos (≈ 20ª casa) e subiu os pisos até a corda solta.
- Em ambas, as bandas **se sobrepõem**: a nota A2 (110 Hz) cabe nas bandas A, D e G da
  fase 2 — é exatamente aí que "a mesma nota aparece em mais de uma banda".
- Margens de segurança passadas ao pYIN: fase 1 `fmin·0.90`, `fmax·1.05`; fase 2
  `fmin·0.88`, `fmax·1.08` (L287) e máscara `mascara_banda` com as mesmas margens (L299).
- V1 monofônica: banda única B0–G4 (`FMIN`, `FMAX`, L26–27), sem separação por corda.

## 2. Pré-processamento

| Parâmetro | V1 / fase 1 mono | Fase 1 poli | V2 | V3 |
|---|---|---|---|---|
| HPSS `margin` | 2.0 | 3.0 | 3.0 (L359) | 3.0 (L191) |
| Butterworth passa-banda | — | SOS, `ordem=3` | SOS, `ordem=8` (default de `_filtro_passa_banda`; a chamada em L274 não sobrescreve) | herda V2 |
| Spectral gate | — | — | `n_fft=2048`, `hop=512`, `noise_quantile=0.10`, `gate_multiplier=1.2` | herda V2 (L150) |
| Wiener | — | — | `_wiener_denoise(tamanho_janela=31)` | herda |
| `frame_length` / `hop_length` | 4096 / 1024 | 4096 / 1024; B: 1024/256, E e A: 1024/512 | `_frame_params(fmin)`: fmin<40 → 4096; <70 → 3072; senão 2048; hop = max(256, fl/4) (L165) | herda |

## 3. Aceitação de frames (V2 — a pilha heurística que a V3 substitui)

| Constante | Valor | Linha | Papel |
|---|---|---|---|
| `_VOICED_PROB_THRESHOLD` | 0.10 | L59 | mínimo de `voiced_prob` do pYIN |
| `_RMS_FRAME_ON` | 0.03 | L60 | histerese: liga frame acima deste RMS relativo |
| `_RMS_FRAME_OFF` | 0.015 | L61 | histerese: desliga abaixo deste |
| `_HARMONIC_RATIO` | 0.08 | L62 | energia mínima nos harmônicos vs ruído mediano |
| `_HARMONIC_N` | 3 | L63 | nº de harmônicos verificados |
| `_RMS_REF_FACTOR` | B .0006, E .0007, A .0009, D .0011, G .0013 | L64 | piso de RMS por corda |

Fase 1 (`modulo_pyin_polifonico.py`): `VOICED_PROB_THRESHOLD = 0.20` (L83),
`RMS_FRAME_THRESHOLD = 0.05` (L87), correção de oitava por energia em f0/2.

## 4. Viterbi (V3 — `v3_viterbi.py`, L43–50)

HMM por corda: estados = silêncio + um por semitom de `midi_aberta` a `midi_aberta + 24`.

| Constante | Valor | Papel |
|---|---|---|
| `_VIT_SIGMA_CENTS` | 70.0 | largura da gaussiana de emissão (proximidade f0 ↔ estado) |
| `_VIT_W_STEP` | 0.55 | custo por semitom de salto melódico |
| `_VIT_W_OCTAVE` | 7.0 | custo de salto de ±12 ou ±24 semitons |
| `_VIT_W_SIL` | 3.2 | custo de entrar/sair do silêncio |
| `_VIT_W_EMIT` | 1.0 | peso global da emissão |
| `_VIT_N_TRASTES` | 24 | alcance de cada corda |
| `_MIDI_CORDA_ABERTA` | B 23, E 28, A 33, D 38, G 43 | |

Emissão (L85–93): `score = W_EMIT · (0.55 · prox · vprob + 0.45 · e_norm)`, onde
`prox = exp(−(cents/σ)²)` e `e_norm = (|S[f0]| + 0.5·|S[2f0]|) / max(coluna)`;
custo = `−log(score + 1e-3)`. Silêncio: `(1 − vprob)·(1 − col_max/S_max)`.
Os pesos internos **0.55 / 0.45 / 0.5** estão literais no código e também são candidatos
à otimização. Transições ficam fixas por corda; **não há acoplamento entre cordas** — o
Viterbi de cada corda decide sozinho, o que deixa passar a mesma nota em bandas vizinhas.

## 5. Segmentação e fusão

| Parâmetro | Fase 1 mono | Fase 1 poli | V2 | V3 |
|---|---|---|---|---|
| `duracao_minima` (segmento) | 0.05 s | 0.10 s | 0.10 s (por corda), 0.05 s em `pipeline_completo` | 0.10 / 0.05 |
| `tolerancia_overlap` (fusão) | — | 0.03 s | 0.03 s | 0.03 s |

## 6. Limpeza pós-detecção (só V2/V3 — `limpar_segmentos`, defaults)

| Parâmetro | Default | Filtro |
|---|---|---|
| `duracao_minima` | 0.12 s | descarta curtos |
| `gap_maximo` | 0.06 s | `_juntar_gaps` une mesma nota separada por gap ≤ este |
| `nota_curta` | 0.08 s | |
| `rms_ratio_onset` / `janela_busca_onset` | 0.15 / 2.0 s | `filtrar_por_onset` |
| `harmonicos_verificar` / `tolerancia_cents` / `margem_temporal_harm` / `rms_ratio_harm` | (2,3,4) / 35 cents / 0.10 s / 1.5 | `filtrar_harmonicos` (nota fantasma = harmônico de outra) |
| `excesso_traste_max` | 4 | idem |
| `janela_sobreposicao` / `rms_ratio_corda` | 0.15 s / 3.0 | `filtrar_corda_errada` (**é o filtro de vazamento entre bandas**) |

`pipeline_completo` expõe os mesmos com prefixo `pp_` (`pp_duracao_minima=0.12`, …).

## 7. Mapeamento nota → corda/casa e caminho mínimo

| Parâmetro | Fase 1 | V2/V3 |
|---|---|---|
| Trastes máximos | 24 (`_MAX_CASAS`) | 24 |
| Custo | Manhattan `|Δtraste| + |Δcorda|` (Euclidiano e ponderado `peso_traste=2.0` implementados, não usados) | `_custo_fingering(peso_corda=2.0, peso_traste=1.0)` |
| Preferir corda detectada | — | `preferir_corda_detectada=True`, `bonus_corda=0.5` |

## 8. Estratégias anti-cross-talk (fase 1, `estrategias_pyin.py`)

| Estratégia | Parâmetros |
|---|---|
| 2. Remoção de harmônicos | `tol_cents=80`, `min_overlap_s=0.05` |
| 3. Notch progressivo | `n_fft=4096`, `hop=512`, `atenuacao=0.06` |
| 4. Subtração espectral iterativa | `n_fft=4096`, `hop=512`, `fator=0.82` |

## 9. Bancada sintética (fase 1)

`SR = 22050`, `TOLERANCIA_CENTS = 50`, `TOLERANCIA_MIDI = 0.5`, `HOP_LENGTH = 1024`
(polifônico), `seed = 42`; ruídos Lee & Shen: `SR_AUDIO = 22050`, `AMPLITUDE = 0.8`.

## 10. Métricas do benchmark (fase 2, `benchmark_completo.py`)

- `curtos`: segmentos ≤ 0.05 s.
- `suboitava`: notas ≤ mediana − 12 semitons (proxy de erro de oitava).
- `impossiveis` / `pct_imp`: notas < `floor` (43 = G2 para `Corda_Sol_G.mp3`).
- `n_notas_distintas`: cardinalidade de pitches (proxy de fantasmas).
