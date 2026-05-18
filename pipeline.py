"""
PIPELINE — Integração Completa dos 4 Módulos
=============================================
Fluxo:
    Áudio
      ↓  [Módulo 1] analisar_baixo()
    segmentos  (nota_midi, inicio, fim)
      ↓  [Bridge 1→2] bridge_pyin_para_notas()
    sequencia  (nome, inicio, fim)
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
    -Usar a venv linux
      source venv/bin/activate
    -Usar a venv windows
      .venv\\Scripts\\activate
    -Executar o pipeline com um arquivo de áudio:

    from pipeline import pipeline_completo
    pipeline_completo("seu_audio.mp3")
"""

from modulo_pyin      import analisar_baixo, bridge_pyin_para_notas
from modulo_notas     import calcular_casas_baixo, bridge_notas_para_dp
from modulo_caminho   import menor_caminho_DP, bridge_caminho_para_tab
from modulo_tablatura import print_tab_multiplas_notas


def pipeline_completo(
    audio_path: str,
    plotar: bool = True,
    duracao_minima: float = 0.05,
    num_cordas: int = 5,
) -> dict:
    """
    Executa o pipeline completo: áudio → tablatura.

    Parâmetros
    ----------
    audio_path     : str    Caminho do arquivo de áudio.
    plotar         : bool   Se True, exibe o gráfico do Módulo 1.
    duracao_minima : float  Notas menores que este valor (s) são descartadas.
    num_cordas     : int    4 ou 5 cordas.

    Retorna
    -------
    dict com as saídas intermediárias de cada módulo:
        "segmentos"     → saída bruta do pYIN
        "sequencia"     → notas filtradas com timing
        "posicoes_dict" → posições no braço por nota
        "notas_dp"      → input do DP
        "custo_total"   → custo da solução DP
        "caminho"       → sequência de posições escolhida
        "tab_input"     → input da tablatura
    """
    print("\n" + "="*60)
    print("  PIPELINE: Áudio → Tablatura de Baixo")
    print("="*60)

    # ── ETAPA 1: pYIN ────────────────────────────────────────────────
    print("\n[1/4] Detecção de pitch (pYIN)...")
    segmentos = analisar_baixo(audio_path, plotar=plotar)

    # ── BRIDGE 1 → 2 ─────────────────────────────────────────────────
    sequencia = bridge_pyin_para_notas(segmentos, duracao_minima=duracao_minima)
    print(f"      → {len(sequencia)} notas detectadas após filtragem.")
    if not sequencia:
        print("[ERRO] Nenhuma nota detectada. Verifique o áudio.")
        return {}

    # ── ETAPA 2: Posições no braço ────────────────────────────────────
    print("\n[2/4] Calculando posições no braço...")
    nomes_unicos  = list(dict.fromkeys(n["nome"] for n in sequencia))
    posicoes_dict = calcular_casas_baixo(nomes_unicos)
    for nota, pos in posicoes_dict.items():
        print(f"      {nota}: {pos}")

    # ── BRIDGE 2 → 3 ─────────────────────────────────────────────────
    notas_dp = bridge_notas_para_dp(sequencia, posicoes_dict)
    if not notas_dp:
        print("[ERRO] Nenhuma nota com posição válida para o DP.")
        return {}

    # ── ETAPA 3: Menor caminho ────────────────────────────────────────
    print(f"\n[3/4] Calculando menor caminho entre {len(notas_dp)} notas (DP)...")
    custo_total, caminho = menor_caminho_DP(notas_dp)
    print(f"      → Custo total de movimento: {custo_total:.1f}")
    for passo in caminho:
        print(f"         {passo['nota']}: corda {passo['posicao'][0]}, "
              f"traste {passo['posicao'][1]}")

    # ── BRIDGE 3 → 4 ─────────────────────────────────────────────────
    tab_input = bridge_caminho_para_tab(caminho)

    # ── ETAPA 4: Tablatura ────────────────────────────────────────────
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
    
    ARQUIVO = "Money_PinkFloyd.mp3"
    ARQUIVO_2 = "Como_tudo_Deve_Ser.mp3"


    if not ARQUIVO:
        print("[ERRO] Especifique o caminho do arquivo de áudio no código.")
    else:
      resultado = pipeline_completo(ARQUIVO, plotar=True, num_cordas=5)
      resultado_2 = pipeline_completo(ARQUIVO_2, plotar=True, num_cordas=5)
    #usando try except para capturar erros
    # try:
