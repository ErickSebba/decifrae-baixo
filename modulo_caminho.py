"""
MÓDULO 3 — Menor Caminho (Programação Dinâmica)
================================================
Entrada : list[dict] no formato:
    [
        {"nome": "A#1", "posicoes": [[3,1],[4,6],[5,11]]},
        {"nome": "B1",  "posicoes": [[3,2],[4,7],[5,12]]},
        ...
    ]
    Produzida por bridge_notas_para_dp() do Módulo 2.

Saída   : (custo_total: int, caminho: list[dict])
    caminho = [
        {"nota": "A#1", "posicao": [3, 1]},
        {"nota": "B1",  "posicao": [4, 7]},
        ...
    ]

Como integrar com o Módulo 4:
    from modulo_caminho import menor_caminho_DP, bridge_caminho_para_tab
    custo, caminho = menor_caminho_DP(notas_dp)
    tab_input = bridge_caminho_para_tab(caminho)
    # → passe tab_input para print_tab_multiplas_notas()

Notas sobre métricas de custo disponíveis (descomentadas abaixo):
    Manhattan  : |Δtraste| + |Δcorda|          ← padrão atual
    Euclidiana : sqrt(Δtraste² + Δcorda²)
    Ponderada  : |Δtraste|*peso + |Δcorda|     ← prioriza minimizar mudança de traste
"""

import math


# ── Métricas de custo ─────────────────────────────────────────────────────────
# Escolha UMA como ativa (ou passe por parâmetro em menor_caminho_DP).

def custo_manhattan(pos1: list[int], pos2: list[int]) -> float:
    """|Δtraste| + |Δcorda|  — simétrico, penaliza ambas as dimensões igual."""
    return abs(pos2[1] - pos1[1]) + abs(pos2[0] - pos1[0])


def custo_euclidiano(pos1: list[int], pos2: list[int]) -> float:
    """sqrt(Δtraste² + Δcorda²)  — distância real no espaço 2D."""
    return math.sqrt((pos2[1] - pos1[1])**2 + (pos2[0] - pos1[0])**2)


def custo_ponderado(pos1: list[int], pos2: list[int], peso_traste: float = 2.0) -> float:
    """
    |Δtraste|*peso + |Δcorda|
    Útil quando movimento de traste é mais custoso que troca de corda.
    """
    return abs(pos2[1] - pos1[1]) * peso_traste + abs(pos2[0] - pos1[0])


# Padrão usado pelo pipeline
_custo_padrao = custo_manhattan


# ── Algoritmo principal ───────────────────────────────────────────────────────

def menor_caminho_DP(
    notas: list[dict],
    fn_custo=None,
) -> tuple[float, list[dict]]:
    """
    Encontra a sequência de posições de mão esquerda com menor custo total
    usando Programação Dinâmica (equivalente ao DAG shortest path).

    Parâmetros
    ----------
    notas    : list[dict]
        Cada dict: {"nome": str, "posicoes": list[[corda, traste]], ...}
        Produzido por bridge_notas_para_dp().
    fn_custo : callable, opcional
        Função de custo (pos1, pos2) → float.
        Padrão: custo_manhattan.

    Retorna
    -------
    (custo_total: float, caminho: list[dict])
        caminho[i] = {"nota": str, "posicao": [corda, traste]}
        Preserva quaisquer campos extras (inicio/fim) dos dicts de entrada.

    Levanta
    -------
    ValueError  se `notas` estiver vazia ou alguma nota não tiver posições.
    """
    if not notas:
        raise ValueError("Lista de notas está vazia.")
    for i, n in enumerate(notas):
        if not n.get("posicoes"):
            raise ValueError(f"Nota no índice {i} ('{n.get('nome')}') não tem posições.")

    fn = fn_custo or _custo_padrao
    n  = len(notas)

    # dp[i][j] = (melhor custo acumulado até aqui, índice predecessor em i-1)
    dp = [[None] * len(notas[i]["posicoes"]) for i in range(n)]

    # Caso base
    for j in range(len(notas[0]["posicoes"])):
        dp[0][j] = (0.0, None)

    # Preenchimento
    for i in range(1, n):
        for j, pos_atual in enumerate(notas[i]["posicoes"]):
            melhor  = float("inf")
            de_onde = None
            for k, pos_ant in enumerate(notas[i - 1]["posicoes"]):
                c = dp[i - 1][k][0] + fn(pos_ant, pos_atual)
                if c < melhor:
                    melhor, de_onde = c, k
            dp[i][j] = (melhor, de_onde)

    # Mínimo na última coluna
    melhor_j    = min(range(len(notas[-1]["posicoes"])),
                      key=lambda j: dp[-1][j][0])
    custo_total = dp[-1][melhor_j][0]

    # Backtracking
    caminho, j = [], melhor_j
    for i in range(n - 1, -1, -1):
        entrada = {
            "nota":    notas[i]["nome"],
            "posicao": notas[i]["posicoes"][j],
        }
        # Propaga timing se disponível
        if "inicio" in notas[i]:
            entrada["inicio"] = notas[i]["inicio"]
        if "fim" in notas[i]:
            entrada["fim"] = notas[i]["fim"]
        caminho.append(entrada)
        if i > 0:
            j = dp[i][j][1]

    return custo_total, list(reversed(caminho))


# ── Bridge para o Módulo 4 ────────────────────────────────────────────────────

def bridge_caminho_para_tab(caminho: list[dict]) -> list[list]:
    """
    Converte a saída de menor_caminho_DP() para o formato de entrada
    de print_tab_multiplas_notas().

    Parâmetros
    ----------
    caminho : list[dict]
        Saída direta de menor_caminho_DP().

    Retorna
    -------
    list[list[int, int, int]]
        Cada elemento: [corda, traste, tempo_idx]
        tempo_idx é o índice da nota na sequência (0, 1, 2, ...).
    """
    return [
        [passo["posicao"][0], passo["posicao"][1], i]
        for i, passo in enumerate(caminho)
    ]
