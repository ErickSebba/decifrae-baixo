# Dossiê técnico-científico do módulo de testes e ruídos do Decifraê

## 1. RESUMO EXECUTIVO
O módulo de testes do Decifraê é um ambiente sintético de validação para a transcrição de baixo baseada em pYIN, construído para observar, sob gabarito conhecido, como a detecção de pitch degrada quando a forma de onda se afasta do caso ideal.
A primeira versão (`modulo_teste_1.py`) mede a resposta do pYIN monofônico em nota única, com e sem decay, com e sem harmônicos, ruído branco, interferência harmônica, silêncio e sequências curtas.
A segunda versão (`modulo_teste_2.py`) amplia esse ambiente com uma suíte sistemática de ruídos inspirada em Lee & Shen (2011), adaptada do domínio elétrico para áudio musical, cobrindo 16 classes de perturbação.
A versão polifônica (`modulo_teste_polifonico.py`) muda o foco do teste: em vez de apenas saber se o pitch foi detectado, passa a avaliar se o sistema consegue recuperar múltiplas notas simultâneas, sobreposição parcial, erros de oitava e falsos positivos por corda.
O conjunto de testes não usa áudio externo: todo o gabarito é sintético, com frequência, início e fim definidos no código.
Isso torna o módulo útil como bancada controlada, mas não como validação final de campo, porque não existe ground truth real de tablatura no repositório.
O pipeline principal do baixo está dividido em pYIN monofônico, versão polifônica por corda, e uma etapa final de mapeamento nota→posição→caminho mínimo→tablatura ASCII.
A arquitetura atual privilegia rastreabilidade e depuração: cada etapa produz estruturas intermediárias simples, em listas de dicionários.
A robustez contra ruído foi tratada de forma incremental, com HPSS, limiares de energia, filtro passa-banda por corda, correção de oitava e fusão de segmentos.
A parte de decisões de digitação no braço é resolvida por programação dinâmica com custo Manhattan; isso ainda é uma heurística de engenharia, não uma validação biomecânica empírica.
Não há integração FastAPI nem frontend React neste workspace; essas camadas devem ser tratadas como [NÃO IMPLEMENTADO] se mencionadas em uma defesa.

## 2. IDENTIFICAÇÃO: objetivo, entrada, saída, stack, maturidade, dependências
**Objetivo.** Validar e estressar o pipeline de transcrição de baixo, isolando falhas de pitch detection, ruído e ambiguidade de digitação antes da geração da tablatura.

**Entrada.** No módulo de testes, a entrada típica é áudio sintético em `numpy.ndarray` com `sr = 22050` Hz, acompanhado de gabarito estruturado em `list[dict]` com campos como `freq_hz`, `nota_nome`, `inicio` e `fim`.
No pipeline principal, a entrada é um arquivo de áudio em disco (`str`).

**Saída.**
- `modulo_teste_1.py` e `modulo_teste_2.py`: métricas por caso, diagnósticos textuais, gráficos comparativos e arrays intermediários do pYIN.
- `modulo_teste_polifonico.py`: `precision`, `recall`, `F1`, número de notas fantasma, erros de oitava e diagnóstico por nota.
- Pipeline principal: lista de segmentos detectados, notas filtradas, posições possíveis no braço, caminho mínimo e tablatura ASCII.

**Stack.** `numpy`, `librosa`, `matplotlib`, `scipy.signal`, `dataclasses`, `random`, `warnings`, `enum`.
Na camada polifônica e de estratégia, o código também usa `matplotlib.patches`, `matplotlib.gridspec` e `librosa.effects.hpss`, `librosa.pyin`, `librosa.stft`.

**Maturidade.** Protótipo experimental / pesquisa aplicada em estágio de bancada. O repositório já contém testes sintéticos, suítes de stress e comparador de estratégias, mas não contém protocolo publicado nem benchmark externo consolidado.

**Dependências.**
- Obrigatórias: `librosa`, `numpy`, `matplotlib`, `scipy`.
- Opcionais/infra: execução em console e ambiente Python local.
- Ausentes neste workspace: FastAPI, React, banco de dados, dataset anotado real, avaliação automatizada persistida.

## 3. ARQUITETURA DO PIPELINE: diagrama textual + formato dos dados em cada aresta
### Fluxo principal
Áudio em arquivo
→ `[Módulo 1] analisar_baixo()`
→ `segmentos = list[dict]`
→ `[bridge_pyin_para_notas()]`
→ `sequencia = list[dict]`
→ `[Módulo 2] calcular_casas_baixo()`
→ `posicoes_dict = dict[str, list[list[int]]]`
→ `[bridge_notas_para_dp()]`
→ `notas_dp = list[dict]`
→ `[Módulo 3] menor_caminho_DP()`
→ `caminho = list[dict]` + `custo_total`
→ `[bridge_caminho_para_tab()]`
→ `tab_input = list[list[int]]`
→ `[Módulo 4] print_tab_multiplas_notas()`
→ tablatura ASCII

### Fluxo polifônico
Áudio em arquivo ou array
→ HPSS com `margin=3.0`
→ filtro passa-banda por corda
→ pYIN por corda
→ máscara de energia + threshold de voiced probability + correção de oitava
→ agrupamento em segmentos por corda
→ fusão de duplicatas com sobreposição temporal
→ `segmentos = list[dict]` com `{nota_midi, inicio, fim, corda}`
→ bridge polifônica para notas nomeadas
→ comparação com gabarito sintético

### Formatos nas arestas
- `segmentos` monofônicos: `{"nota_midi": float, "inicio": float, "fim": float}`.
- `sequencia` monofônica: `{"nome": str, "inicio": float, "fim": float}`.
- `posicoes_dict`: chave nota textual, valor `[[corda, traste], ...]`.
- `notas_dp`: `{"nome": str, "posicoes": [[corda, traste], ...], "inicio": float?, "fim": float?}`.
- `caminho`: `{"nota": str, "posicao": [corda, traste], "inicio": float?, "fim": float?}`.
- `tab_input`: `[corda, traste, tempo_idx]`.
- `segmentos` polifônicos: `{"nota_midi": float, "inicio": float, "fim": float, "corda": str}`.

## 4. DETALHAMENTO CAMADA POR CAMADA

### 4.1 Pré-processamento (mono, resample, filtro passa-baixa/passa-alta, normalização, gating)
**(a) Propósito.** Reduzir fontes de erro antes do pitch tracking: percussão, conteúdo fora da banda útil e frames com energia insuficiente.

**(b) Algoritmo e referência canônica.**
- `librosa.effects.hpss()` para separação harmônico-percussiva, com `margin=2.0` no módulo monofônico e `margin=3.0` no polifônico.
- Filtro Butterworth passa-banda em forma SOS via `scipy.signal.butter(..., output="sos")` + `sosfiltfilt`.
- Máscara de energia baseada em RMS por frame.

**(c) Formulação matemática.**
- HPSS não é explicitado algébricamente no código; é uma transformação espectral de separação harmônica/percussiva do `librosa`.
- Filtro passa-banda: o código aplica um Butterworth de ordem `3` sobre a banda `[fmin, fmax]` normalizada por Nyquist.
- RMS por frame: `rms = sqrt(mean(y_frame^2))`.
- Gating: um frame é aceito se `rms >= pico_rms × threshold`.

**(d) Hiperparâmetros com valores.**
- `margin=2.0` no monofônico.
- `margin=3.0` no polifônico.
- `ordem=3` no Butterworth.
- `frame_length=4096`, `hop_length=1024` como padrão de análise.
- `RMS_FRAME_THRESHOLD = 0.05`.
- `VOICED_PROB_THRESHOLD = 0.20`.

**(e) Justificativa.** O baixo tem harmônicos fortes e ataques que podem parecer pitch; separar a componente harmônica e limitar frames frágeis reduz falsos positivos.

**(f) Alternativas descartadas.**
- Filtro `(b, a)` clássico foi explicitamente substituído por SOS por instabilidade numérica em bandas baixas.
- Limiar voiced mais alto foi discutido no código, mas o valor ativo ficou em `0.20`.

**(g) Custo.** O HPSS e o STFT da correção de oitava adicionam custo espectral por trecho; no polifônico, o custo cresce com o número de cordas analisadas.

**(h) Fragilidades.**
- HPSS pode remover parte da energia útil em ataques percussivos com harmônicos reais.
- O gating por RMS pode eliminar notas muito fracas ou sustain quase morto.
- O filtro por corda depende de bandas fixas e não se adapta ao timbre real.

### 4.2 Detecção de pitch com pYIN
**(a) Propósito.** Estimar a fundamental por frame e separar frames vozeados de não vozeados.

**(b) Algoritmo e referência canônica.** pYIN do `librosa`, inspirado no YIN probabilístico para tracking de f0 com modelo oculto de voicing.

**(c) Formulação matemática.** O código não reimplementa a função de diferença do YIN; ele chama `librosa.pyin()`. Operacionalmente, os frames retornam `f0`, `voiced_flag` e `voiced_probs`.

**(d) Hiperparâmetros com valores.**
- Monofônico: `fmin = librosa.note_to_hz("B0") ≈ 30.87 Hz`.
- Monofônico: `fmax = librosa.note_to_hz("G4") ≈ 392 Hz`.
- Polifônico por corda:
  - B: `28–58 Hz`
  - E: `38–82 Hz`
  - A: `51–110 Hz`
  - D: `68–147 Hz`
  - G: `92–220 Hz`
- `frame_length=4096`, `hop_length=1024` no monofônico.
- Em bandas estreitas, `_pyin_uma_corda()` reduz para `frame_length=1024` e `hop_length=256` para a corda B, e `frame_length=1024`, `hop_length=512` para E/A.
- `fmin` e `fmax` do pYIN por corda recebem margens de segurança: `fmin * 0.90` e `fmax * 1.05`.

**(e) Justificativa.** O pYIN continua sendo o núcleo do sistema porque é robusto em comparação com detecção ingênua por pico espectral, e permite limiares de voicing e quantização em MIDI.

**(f) Alternativas descartadas.** Não há, neste workspace, outro detector de pitch competindo no mesmo módulo. O comparador de estratégias atua sobre o mesmo pYIN por corda, não substitui o pYIN.

**(g) Custo.** Cada chamada ao pYIN é cara em relação a uma operação puramente vetorial. Na versão polifônica, o custo cresce aproximadamente com o número de cordas analisadas, porque o áudio filtrado é reprocessado por banda.

**(h) Fragilidades.**
- O pYIN é monofônico por construção; em mistura simultânea ele tende a escolher uma trajetória dominante.
- Em baixo elétrico, o 2º harmônico pode ser mais forte que a fundamental, gerando erro de oitava.
- Em trechos com transientes ou ruído, `voiced_flag` pode oscilar.

### 4.3 Extensão polifônica (V2)
**(a) Propósito.** Aumentar a chance de recuperar notas simultâneas, associando a cada corda uma faixa espectral própria.

**(b) Algoritmo e referência canônica.** Estrutura de análise por banda: `HPSS → passa-banda por corda → pYIN por corda → fusão de segmentos`.

**(c) Formulação matemática.** Não há separação cega de fontes; a polifonia é aproximada por decomposição por bandas. A segmentação final funde eventos iguais com sobreposição temporal maior que `tolerancia_overlap`.

**(d) Hiperparâmetros com valores.**
- `CORDAS_5 = [{nome:"B", fmin:28, fmax:58}, {"E",38,82}, {"A",51,110}, {"D",68,147}, {"G",92,220}]`.
- `CORDAS_4 = [{"E",38,82}, {"A",51,110}, {"D",68,147}, {"G",92,220}]`.
- `duracao_minima = 0.10 s` na versão polifônica.
- `tolerancia_overlap = 0.03 s`.

**(e) Justificativa.** O baixo de múltiplas cordas ocupa faixas espectrais parcialmente separáveis; restringir a busca por corda reduz cross-talk e ajuda a localizar notas simultâneas.

**(f) Alternativas descartadas.** O repositório contém quatro estratégias comparativas, mas não substitui a análise por corda por um modelo separado de source separation completo.

**(g) Custo.** Mais caro que o monofônico, porque roda um pYIN por corda e ainda executa filtros e fusões.

**(h) Fragilidades.**
- Cordas com bandas sobrepostas continuam gerando ambiguidade.
- Harmônicos de uma corda podem cair na banda de outra.
- O método não decompõe verdadeiramente as fontes; apenas restringe o espaço de busca.

### 4.4 Suavização temporal / Viterbi (V3)
**(a) Propósito.** Suavizar a trajetória temporal para reduzir saltos espúrios frame a frame.

**(b) Algoritmo e referência canônica.** [NÃO IMPLEMENTADO] neste workspace. Não há busca por `Viterbi`, `HMM` ou `smoothing` nas rotinas centrais, e nenhum módulo define um decodificador Viterbi.

**(c) Formulação matemática.** [NÃO IMPLEMENTADO].

**(d) Hiperparâmetros com valores.** [NÃO IMPLEMENTADO].

**(e) Justificativa.** Como ideia de projeto, faz sentido para estabilizar nota/voicing entre frames; porém não existe implementação visível aqui.

**(f) Alternativas descartadas.** A suavização presente no código atual é indireta, via filtro de energia, limiares e fusão de segmentos, não via sequência oculta.

**(g) Custo.** [NÃO IMPLEMENTADO].

**(h) Fragilidades.** Sem suavização explícita, a segmentação ainda herda os saltos do pYIN em ataques, silêncios curtos e transições rápidas.

### 4.5 Segmentação em notas
**(a) Propósito.** Converter uma sequência framewise de f0 em eventos musicais com início, fim e nota quantizada.

**(b) Algoritmo e referência canônica.** Agrupamento de frames consecutivos com a mesma nota MIDI arredondada; quebra em `NaN` ou troca de pitch.

**(c) Formulação matemática.** A nota quantizada é `midi_quant = round(hz_to_midi(f0))`. O evento começa no primeiro frame da nota e termina quando surge `NaN` ou uma nova nota.

**(d) Hiperparâmetros com valores.**
- Monofônico: duração mínima no bridge `0.05 s`.
- Polifônico: duração mínima `0.10 s`.
- Em visualização, segmentos com duração menor que `0.03 s` ou `0.05 s` podem ser ocultados nos gráficos.

**(e) Justificativa.** O resultado do pYIN é framewise; para tablatura é necessário um evento discreto por nota.

**(f) Alternativas descartadas.** Não há detecção explícita de onset/offset por energia ou mudança espectral; o agrupamento depende do `voiced_flag` e da quantização MIDI.

**(g) Custo.** Linear no número de frames.

**(h) Fragilidades.** Notas com sustain longo podem ser quebradas por quedas curtas de voicing; ataques curtos podem ser descartados pelo limiar mínimo.

### 4.6 Quantização rítmica
**(a) Propósito.** Converter durações contínuas em grade rítmica.

**(b) Algoritmo e referência canônica.** [NÃO IMPLEMENTADO]. O código não estima BPM nem quantiza para grade métrica; apenas preserva `inicio` e `fim` em segundos.

**(c) Formulação matemática.** [NÃO IMPLEMENTADO].

**(d) Hiperparâmetros com valores.** [NÃO IMPLEMENTADO].

**(e) Justificativa.** A ausência de quantização rítmica evita introduzir erro adicional sem referência de tempo global.

**(f) Alternativas descartadas.** Não há detecção de tempo, beat tracking ou rubato modelado.

**(g) Custo.** [NÃO IMPLEMENTADO].

**(h) Fragilidades.** A tablatura gerada conserva timing contínuo, mas não uma métrica musical editorial.

### 4.7 Mapeamento nota → (corda, casa)
**(a) Propósito.** Dada uma nota textual, enumerar todas as posições fisicamente válidas no braço do baixo e escolher uma trajetória global coerente.

**(b) Algoritmo e referência canônica.**
- `modulo_notas.py`: conversão de nome para índice interno e geração das posições válidas por nota.
- `modulo_caminho.py`: programação dinâmica em grafo acíclico por sequência de notas.

**(c) Formulação matemática.**
- Índice interno de nota: `índice = (oitava × 12) + mapa_nota - 11`.
- Em cada nota, a posição válida em uma corda é `casa = valor_nota - valor_corda`, desde que `0 <= casa <= 24`.
- O menor caminho usa custo Manhattan: `|Δtraste| + |Δcorda|`.

**(d) Hiperparâmetros com valores.**
- 5 cordas: G2, D2, A1, E1, B0.
- 4 cordas: G2, D2, A1, E1.
- Limite de trastes: `24`.
- Custo padrão: Manhattan.
- Alternativa implementada, mas não padrão: custo Euclidiano e custo ponderado (`peso_traste=2.0`).

**(e) Justificativa.** O sistema precisa escolher uma única posição por nota, mas a mesma nota aparece em várias cordas; o DP busca reduzir deslocamento global da mão.

**(f) Alternativas descartadas.** O código traz comentários para custo Euclidiano e ponderado, mas o pipeline usa Manhattan.

**(g) Custo.** O DP é quadrático no número de posições candidatas por nota, multiplicado pela sequência de notas.

**(h) Fragilidades.**
- A mesma nota pode soar em várias cordas com timbres diferentes; o custo atual não modela timbre.
- Não há validação empírica da biomecânica da mão.
- O caminho matematicamente ótimo pode ser impraticável para o instrumentista.
- O custo não incorpora posição de mão, legato, barré, corda solta preferencial ou ergonomia real.

### 4.8 Geração e renderização da tablatura
**(a) Propósito.** Transformar a sequência de posições escolhidas em ASCII tabular legível.

**(b) Algoritmo e referência canônica.** `print_tab_multiplas_notas()` agrupa por `tempo_idx` e monta linhas por corda.

**(c) Formulação matemática.** Não há matemática contínua; a saída é uma concatenação de caracteres com alinhamento por bloco temporal.

**(d) Hiperparâmetros com valores.**
- `num_cordas=5` por padrão.
- Convenção de `corda == 0` para pulo de tempo, usada para adicionar espaço visual extra.

**(e) Justificativa.** A tablatura ASCII é o formato operacional mais simples para depuração e apresentação.

**(f) Alternativas descartadas.** Não há exportação para MusicXML, MIDI ou formato gráfico neste workspace.

**(g) Custo.** Linear no número de eventos.

**(h) Fragilidades.** A saída é textual e não carrega informações de articulação, dinâmica, duração exata ou dedilhado da mão direita.

### 4.9 Integração (API FastAPI, contrato JSON, frontend React, latência)
**(a) Propósito.** Encapsular o pipeline em serviço e interface visual.

**(b) Algoritmo e referência canônica.** [NÃO IMPLEMENTADO] no workspace atual.

**(c) Formulação matemática.** [NÃO IMPLEMENTADO].

**(d) Hiperparâmetros com valores.** [NÃO IMPLEMENTADO].

**(e) Justificativa.** Não há evidência no código de servidor HTTP, contrato JSON ou frontend.

**(f) Alternativas descartadas.** O que existe hoje é execução por script e chamadas diretas entre módulos.

**(g) Custo.** [NÃO IMPLEMENTADO].

**(h) Fragilidades.** A ausência de API limita reprodutibilidade em ambiente web e integração com interface de usuário.

## 5. HISTÓRICO DE VERSÕES (V1 → V2 → V3)
| Versão | Mudança | Motivação | Efeito observado |
|---|---|---|---|
| V1 (`modulo_teste_1.py`) | Testes monofônicos com sinais sintéticos: seno puro, decay, harmônicos, ruído branco, interferência harmônica, silêncio e sequências curtas | Criar baseline controlado para o pYIN | [NÃO MEDIDO] no repositório; o código apenas calcula e imprime métricas |
| V2 (`modulo_teste_2.py`) | Adição da suíte Lee & Shen adaptada ao áudio musical, com 16 ruídos e `seed=42` | Estressar robustez sob distúrbios sistemáticos | [NÃO MEDIDO] no repositório; aumenta a cobertura de cenários degradados |
| V3 (`modulo_teste_polifonico.py`) | Mudança para avaliação por corda, com notas simultâneas, sobreposição parcial, silêncio/fantasma e correção de oitava | Testar o caso crítico que o pYIN monofônico não resolve | [NÃO MEDIDO] no repositório; introduz `precision`, `recall`, `F1` e métricas de falsos positivos |

Observação: no workspace também existe `comparador_estrategias.py`, que compara quatro estratégias polifônicas sobre as mesmas cinco suites, mas não armazena resultados consolidados históricos.

## 6. AVALIAÇÃO: datasets, protocolo, métricas, resultados, baselines
**Datasets.**
- `modulo_teste_1.py`: áudio totalmente sintético, gerado internamente.
- `modulo_teste_2.py`: áudio sintético com ruídos de Lee & Shen adaptados.
- `modulo_teste_polifonico.py`: áudio sintético polifônico, incluindo sobreposição e silêncio.

**Protocolo.**
- Cada caso tem gabarito explícito em `list[dict]`.
- Os testes chamam `rodar_pyin()` ou `rodar_polifonico_em_array()` e comparam saída com o gabarito.
- O comparador de estratégias usa as cinco suites polifônicas e agrega médias por suite e geral.

**Métricas.**
- Monofônico: taxa de detecção, taxa de silêncio correto, acurácia de nota dentro de `±50 cents`, erro de oitava, mediana e máximo do erro em cents.
- Polifônico: `precision`, `recall`, `F1`, número de notas fantasma, número de erros de oitava, consistência por nota.
- No comparador: médias por suite e geral, mais heatmaps e gráficos agrupados.

**Resultados.** [NÃO MEDIDO]. O código imprime resultados quando executado, mas este dossiê não inventa números sem a execução efetiva dos testes nesta sessão.

**Baselines.**
- Monofônico: onda pura e sequência sem ruído.
- Polifônico: nota única por corda, intervalos simultâneos limpos, e ruído nulo em silêncio.

## 7. PROBLEMAS EM ABERTO
1. **pYIN confunde fundamental e harmônico forte.**
   Sintoma: erro de oitava em `A2`, `B2`, `E1`, `A1`, `D2`, `G2` e casos slap.
   Camada: detecção de pitch.
   Causa provável: no baixo, o 2º harmônico pode ter mais energia que a fundamental.
   Hipótese de solução: reforçar a correção espectral de oitava ou usar um modelo explicitamente treinado para baixo.
   Dificuldade: alta, porque o problema varia com timbre, técnica e mix.
   Classificação: [PESQUISA].

2. **Polifonia não é separação real de fontes.**
   Sintoma: sobreposição simultânea pode gerar notas faltantes ou duplicadas.
   Camada: extensão polifônica por corda.
   Causa provável: a abordagem atual é banda por corda, não source separation.
   Hipótese de solução: separar fontes antes do pYIN ou usar modelo polifônico multirresolução.
   Dificuldade: alta.
   Classificação: [PESQUISA].

3. **Ambiguidade corda/casa para a mesma nota.**
   Sintoma: uma mesma nota pode aparecer em várias cordas com resultados distintos de timbre e tocabilidade.
   Camada: mapeamento nota → posição.
   Causa provável: o custo atual não considera timbre nem contexto musical.
   Hipótese de solução: adicionar modelo de preferência por corda, contexto harmônico e timbre.
   Dificuldade: média/alta.
   Classificação: [PESQUISA].

4. **Menor caminho matemático pode ser impraticável para a mão esquerda.**
   Sintoma: DP escolhe uma sequência de posições com baixo custo de deslocamento, mas ergonomicamente estranha.
   Camada: programação dinâmica / caminho mínimo.
   Causa provável: o custo Manhattan não modela anatomia, posição da mão, nem técnica.
   Hipótese de solução: custo multiobjetivo com restrições biomecânicas validadas por instrumentistas.
   Dificuldade: alta.
   Classificação: [PESQUISA].

5. **Ausência de ground truth de tablatura real.**
   Sintoma: métricas atuais são indiretas e sintéticas; não há referência humana anotada no repositório.
   Camada: avaliação.
   Causa provável: falta de corpus anotado de baixo com tablatura e timing alinhados.
   Hipótese de solução: construir dataset anotado ou alinhar MIDI/tablatura de referência.
   Dificuldade: alta e operacionalmente cara.
   Classificação: [PESQUISA].

6. **Ruídos musicais podem ser mais agressivos do que a mixagem real.**
   Sintoma: algumas perturbações de Lee & Shen geram degradadores fortes, úteis para stress mas não necessariamente representativos de música gravada.
   Camada: módulo de ruídos.
   Causa provável: a adaptação preserva a matemática, mas altera o contexto acústico.
   Hipótese de solução: calibrar os parâmetros com gravações reais de baixo em banda/mix.
   Dificuldade: média.
   Classificação: [ENGENHARIA].

7. **Segmentação dependente de limiares fixos.**
   Sintoma: notas curtas ou fracas podem sumir; notas ruidosas podem quebrar em fragmentos.
   Camada: segmentação e gating.
   Causa provável: `duracao_minima`, `RMS_FRAME_THRESHOLD` e `VOICED_PROB_THRESHOLD` são globais.
   Hipótese de solução: limiares adaptativos por contexto espectral ou por corda.
   Dificuldade: média.
   Classificação: [ENGENHARIA].

8. **Viterbi prometido no contexto, mas não implementado aqui.**
   Sintoma: a narrativa de projeto fala em suavização temporal, mas o código não mostra essa etapa.
   Camada: modelagem temporal.
   Causa provável: a versão atual ficou na filtragem heurística.
   Hipótese de solução: inserir um decoder de estados discretos com custo temporal.
   Dificuldade: média.
   Classificação: [PESQUISA].

## 8. PERGUNTAS DE PESQUISA CANDIDATAS
1. Quais combinações de filtragem por corda, correção de oitava e gating por energia melhoram mais a recuperação de pitch em baixo elétrico com timbres variados?
2. Até que ponto uma estratégia de análise por corda substitui, na prática, um modelo polifônico de separação de fontes para transcrição de baixo?
3. Como projetar uma função de custo de tablatura que equilibre custo mecânico, preferências de timbre e estabilidade harmônica?
4. O ganho de precisão da correção de oitava justifica a complexidade computacional adicional em tempo quase real?
5. Como quantificar, com ground truth humano, a diferença entre a tablatura matematicamente ótima e a tablatura executável por instrumentistas?
6. Quais ruídos sintéticos são mais preditivos das falhas reais do sistema em gravações de baixo ao vivo?

## 9. ATIVOS REUTILIZÁVEIS PARA UM TCC
- Suíte sintética monofônica com gabarito conhecido.
- Suíte de ruídos de Lee & Shen adaptada ao domínio musical.
- Suíte polifônica com notas simultâneas, sobreposição parcial e silêncio.
- Implementação de HPSS + pYIN por corda + fusão de segmentos.
- Correção de oitava por energia espectral em torno de `f0/2`.
- Filtro passa-banda SOS estável para bandas baixas.
- Estrutura de métrica para comparação com gabarito sintético.
- Mapeamento nota → posições do braço e DP de menor caminho.
- Renderizador ASCII da tablatura.
- Comparador de estratégias polifônicas com heatmaps e gráficos.

## 10. GLOSSÁRIO DE TERMOS, SÍMBOLOS E SIGLAS
- **pYIN**: Probabilistic YIN, algoritmo probabilístico para estimativa de pitch fundamental.
- **YIN**: método clássico de detecção de pitch baseado em função de diferença.
- **HPSS**: Harmonic-Percussive Source Separation.
- **HPSS margin**: parâmetro do `librosa.effects.hpss()` que controla a agressividade da separação.
- **f0**: frequência fundamental estimada.
- **voiced_flag**: indicador booleano de que o frame foi considerado vozeado.
- **voiced_probs**: probabilidade associada ao voicing por frame.
- **MIDI**: representação numérica de altura musical; aqui usada para quantização de notas.
- **cent**: centésimo de semitom; 100 cents = 1 semitom.
- **SNR**: signal-to-noise ratio.
- **RMS**: root mean square, energia média quadrática por frame.
- **HPF/LPF/BPF**: passa-alta, passa-baixa, passa-banda; neste repositório há passa-banda Butterworth.
- **STFT**: short-time Fourier transform.
- **iSTFT**: inversa da STFT.
- **DP**: programação dinâmica.
- **DAG**: grafo acíclico dirigido.
- **Tab ASCII**: tablatura em texto puro.
- **corda**: string do baixo; no código, identificada por nome (`B`, `E`, `A`, `D`, `G`) e por índice numérico no braço.
- **traste / casa**: posição na corda, de `0` a `24`.
- **erro de oitava**: detecção deslocada em aproximadamente `±12 semitons`.
- **nota fantasma**: nota detectada sem suporte no gabarito temporal.
- **[NÃO IMPLEMENTADO]**: item pedido na estrutura, mas ausente no código deste workspace.
- **[NÃO MEDIDO]**: item cuja execução ou benchmark não foi realizado nesta sessão.
- **[VERIFICAR]**: afirmação que exigiria inspeção adicional ou execução externa para validação.
