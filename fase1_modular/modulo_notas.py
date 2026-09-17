"""
MÓDULO 2 — Posições no Braço do Baixo
=======================================
Entrada : lista de nomes de notas (list[str])  ex.: ["A#1", "B1", "C2"]
Saída   : dict  nota → lista de posições [corda, traste]

Exemplo de saída:
    {
        "A#1": [[3, 1], [4, 6], [5, 11]],
        "B1":  [[3, 2], [4, 7], [5, 12]],
    }

    Corda 1 = G2 (mais aguda)
    Corda 5 = B0 (mais grave)
    Traste  = 0 (corda solta) … 24

Como integrar com o Módulo 3:
    from modulo_notas import calcular_casas_baixo, bridge_notas_para_dp
    posicoes_dict = calcular_casas_baixo(lista_de_nomes)
    notas_dp = bridge_notas_para_dp(sequencia_de_nomes, posicoes_dict)
    # → passe notas_dp para menor_caminho_DP()
"""


# ── Conversão de nome → índice numérico interno ───────────────────────────────
# O índice NÃO é MIDI. É um sistema próprio onde B0 = 0, E1 = 5, A1 = 10,
# D2 = 15, G2 = 20. Serve apenas para calcular distância de traste.

_MAPA_NOTAS = {
    "C": 0, "C#": 1, "C♯": 1, "Db": 1, "D": 2, "D#": 3, "D♯": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "F♯": 6, "Gb": 6, "G": 7, "G#": 8, "G♯": 8,
    "Ab": 8, "A": 9, "A#": 10, "A♯": 10, "Bb": 10, "B": 11,
}

# Corda aberta → índice numérico interno
_CORDAS_INFO = {
    20: 1,  # G2
    15: 2,  # D2
    10: 3,  # A1
    5:  4,  # E1
    0:  5,  # B0
}

_MAX_CASAS = 24  # limite de trastes de um baixo padrão


def _nota_para_indice(nota: str) -> int:
    """Converte "A#1" → índice interno (ex.: 10)."""
    nome   = nota[:-1]
    oitava = int(nota[-1])
    return (oitava * 12) + _MAPA_NOTAS[nome] - 11


def converter_notas_para_indices(vetor_notas: list[str]) -> list[int]:
    """
    Converte uma lista de nomes de notas para índices internos.

    Parâmetros
    ----------
    vetor_notas : list[str]  Nomes de notas (ex.: ["A#1", "B1"]).

    Retorna
    -------
    list[int]  Índices no sistema interno (B0=0, E1=5, A1=10, D2=15, G2=20).
    """
    return [_nota_para_indice(n) for n in vetor_notas]


def calcular_casas_baixo(vetor_notas: list[str]) -> dict[str, list[list[int]]]:
    """
    Para cada nota, calcula todas as posições válidas no braço do baixo.

    Parâmetros
    ----------
    vetor_notas : list[str]
        Lista de nomes de notas, podendo conter duplicatas
        (apenas notas únicas são processadas, mas o dict inclui todas).

    Retorna
    -------
    dict[str, list[list[int]]]
        Mapeamento nota → lista de [corda, traste].
        Corda 1 = G2 (aguda) … Corda 5 = B0 (grave).
        Traste  = 0 (corda solta) … 24.
    """
    notas_unicas = list(dict.fromkeys(vetor_notas))  # preserva ordem, sem dup.
    notas_num    = converter_notas_para_indices(notas_unicas)

    matriz_resultados = {}
    for i, nota_nome in enumerate(notas_unicas):
        valor_nota    = notas_num[i]
        possibilidades = []
        for valor_corda, num_corda in _CORDAS_INFO.items():
            casa_nota = valor_nota - valor_corda
            if 0 <= casa_nota <= _MAX_CASAS:
                possibilidades.append([num_corda, casa_nota])
        matriz_resultados[nota_nome] = possibilidades

    return matriz_resultados


# ── Bridge para o Módulo 3 ───────────────────────────────────────────────────

def bridge_notas_para_dp(
    sequencia_notas: list[dict],
    posicoes_dict: dict[str, list[list[int]]],
) -> list[dict]:
    """
    Converte a saída do Módulo 2 para o formato de entrada do Módulo 3 (DP).

    Parâmetros
    ----------
    sequencia_notas : list[dict]
        Lista ordenada de notas (saída de bridge_pyin_para_notas).
        Cada dict tem pelo menos "nome" (str).
    posicoes_dict : dict
        Saída de calcular_casas_baixo().

    Retorna
    -------
    list[dict]
        Formato: [{"nome": "A#1", "posicoes": [[3,1],[4,6],[5,11]]}, ...]
        Uma entrada por nota na sequência, incluindo repetições.

    Nota: notas sem posição válida no braço são ignoradas com aviso.
    """
    notas_dp = []
    for nota in sequencia_notas:
        nome = nota["nome"]
        if nome not in posicoes_dict or not posicoes_dict[nome]:
            print(f"[AVISO] Nota '{nome}' não tem posição válida no braço — ignorada.")
            continue
        entrada = {"nome": nome, "posicoes": posicoes_dict[nome]}
        # Propaga campos opcionais de timing se existirem
        if "inicio" in nota:
            entrada["inicio"] = nota["inicio"]
        if "fim" in nota:
            entrada["fim"] = nota["fim"]
        notas_dp.append(entrada)
    return notas_dp
