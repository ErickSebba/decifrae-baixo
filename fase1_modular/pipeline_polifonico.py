"""
PIPELINE — Integração Completa dos 4 Módulos (versão polifônica)
=================================================================
Fluxo (modo polifônico):
    Áudio
      ↓  [Módulo 1b] analisar_baixo_polifonico()   ← novo
    segmentos  (nota_midi, inicio, fim, corda)       ← campo "corda" a mais
      ↓  [Bridge 1b→2] bridge_pyin_para_notas_poli()
    sequencia  (nome, inicio, fim, corda)
      ↓  [Módulo 2] calcular_casas_baixo()
    posicoes_dict  {nota → [[corda, traste], ...]}
      ↓  [Bridge 2→3] bridge_notas_para_dp()
    notas_dp  [{"nome", "posicoes", "inicio", "fim"}, ...]
      ↓  [Módulo 3] menor_caminho_DP()
    caminho  [{"nota", "posicao", "inicio", "fim"}, ...]
      ↓  [Bridge 3→4] bridge_caminho_para_tab()
    tab_input  [[corda, traste, tempo_idx], ...]
      ↓  [Módulo 4] print_tab_multiplas_notas()
    Tablatura impressa no console

Uso rápido:
    from pipeline import pipeline_completo
    pipeline_completo("seu_audio.mp3", polifonico=True)   # novo modo
    pipeline_completo("seu_audio.mp3", polifonico=False)  # modo original
"""

# ── Módulo 1 (monofônico original) ───────────────────────────────────────────
from modulo_pyin              import analisar_baixo, bridge_pyin_para_notas

# ── Módulo 1b (polifônico novo) ───────────────────────────────────────────────
from modulo_pyin_polifonico   import (analisar_baixo_polifonico,
                                      bridge_pyin_para_notas_poli)

# ── Módulos 2, 3, 4 (inalterados) ────────────────────────────────────────────
from modulo_notas             import calcular_casas_baixo, bridge_notas_para_dp
from modulo_caminho           import menor_caminho_DP, bridge_caminho_para_tab
from modulo_tablatura         import print_tab_multiplas_notas


def pipeline_completo(
    audio_path: str,
    plotar: bool = True,
    duracao_minima: float = 0.05,
    num_cordas: int = 5,
    polifonico: bool = True,           # <-- novo parâmetro
    tolerancia_overlap: float = 0.03,  # só usado no modo polifônico
) -> dict:
    """
    Executa o pipeline completo: áudio → tablatura.

    Parâmetros
    ----------
    audio_path          : str   Caminho do arquivo de áudio.
    plotar              : bool  Se True, exibe o gráfico do Módulo 1.
    duracao_minima      : float Notas menores que este valor (s) são descartadas.
    num_cordas          : int   4 ou 5 cordas.
    polifonico          : bool  True  → usa pYIN por corda (detecta acorde/intervalo)
                                False → usa pYIN monofônico original
    tolerancia_overlap  : float (só modo polifônico) limiar para fundir duplicatas.

    Retorna
    -------
    dict com as saídas intermediárias de cada módulo.
    """
    modo_str = "POLIFÔNICO (pYIN por corda)" if polifonico else "MONOFÔNICO (pYIN global)"

    print("\n" + "="*60)
    print(f"  PIPELINE: Áudio → Tablatura de Baixo [{modo_str}]")
    print("="*60)

    # ── ETAPA 1: pYIN ────────────────────────────────────────────────────────
    if polifonico:
        print("\n[1/4] Detecção de pitch POLIFÔNICA (pYIN por corda)...")
        segmentos = analisar_baixo_polifonico(
            audio_path,
            num_cordas=num_cordas,
            plotar=plotar,
            duracao_minima=duracao_minima,
            tolerancia_overlap=tolerancia_overlap,
        )
        sequencia = bridge_pyin_para_notas_poli(segmentos,
                                                 duracao_minima=duracao_minima)
    else:
        print("\n[1/4] Detecção de pitch monofônica (pYIN global)...")
        segmentos = analisar_baixo(audio_path, plotar=plotar)
        sequencia = bridge_pyin_para_notas(segmentos,
                                            duracao_minima=duracao_minima)

    print(f"      → {len(sequencia)} notas detectadas após filtragem.")
    if not sequencia:
        print("[ERRO] Nenhuma nota detectada. Verifique o áudio.")
        return {}

    # ── ETAPA 2: Posições no braço ────────────────────────────────────────────
    print("\n[2/4] Calculando posições no braço...")
    nomes_unicos  = list(dict.fromkeys(n["nome"] for n in sequencia))
    posicoes_dict = calcular_casas_baixo(nomes_unicos)
    for nota, pos in posicoes_dict.items():
        print(f"      {nota}: {pos}")

    # ── BRIDGE 2 → 3 ─────────────────────────────────────────────────────────
    notas_dp = bridge_notas_para_dp(sequencia, posicoes_dict)
    if not notas_dp:
        print("[ERRO] Nenhuma nota com posição válida para o DP.")
        return {}

    # ── ETAPA 3: Menor caminho ────────────────────────────────────────────────
    print(f"\n[3/4] Calculando menor caminho entre {len(notas_dp)} notas (DP)...")
    custo_total, caminho = menor_caminho_DP(notas_dp)
    print(f"      → Custo total de movimento: {custo_total:.1f}")
    for passo in caminho:
        corda_info = (f"  [corda detectada: {passo.get('corda','?')}]"
                      if "corda" in passo else "")
        print(f"         {passo['nota']}: corda {passo['posicao'][0]}, "
              f"traste {passo['posicao'][1]}{corda_info}")

    # ── BRIDGE 3 → 4 ─────────────────────────────────────────────────────────
    tab_input = bridge_caminho_para_tab(caminho)

    # ── ETAPA 4: Tablatura ────────────────────────────────────────────────────
    print(f"\n[4/4] Gerando tablatura ({num_cordas} cordas):\n")
    print_tab_multiplas_notas(tab_input, num_cordas=num_cordas)

    print("\n" + "="*60)
    print("  Pipeline concluído.")
    print("="*60 + "\n")

    return {
        "segmentos":     segmentos,
        "sequencia":     sequencia,
        "posicoes_dict": posicoes_dict,
        "notas_dp":      notas_dp,
        "custo_total":   custo_total,
        "caminho":       caminho,
        "tab_input":     tab_input,
    }


# ── Execução direta ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    ARQUIVO   = "Money_PinkFloyd.mp3"
    ARQUIVO_2 = "Como_tudo_Deve_Ser.mp3"

    if not ARQUIVO:
        print("[ERRO] Especifique o caminho do arquivo de áudio no código.")
    else:
        resultado   = pipeline_completo(ARQUIVO,   plotar=True,
                                        num_cordas=5, polifonico=True)
        resultado_2 = pipeline_completo(ARQUIVO_2, plotar=True,
                                        num_cordas=5, polifonico=True)
