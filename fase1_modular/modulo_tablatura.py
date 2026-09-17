"""
MÓDULO 4 — Renderização de Tablatura em ASCII
==============================================
Entrada : list[[corda, traste, tempo_idx]]
    Produzida por bridge_caminho_para_tab() do Módulo 3.

    Convenção especial: corda == 0 → comando de "pular tempo"
    (adiciona espaço visual extra na tablatura).

Saída   : tablatura impressa no console (str também retornada).

Afinação do baixo de 5 cordas:
    Corda 1 = G2 (mais aguda)
    Corda 2 = D2
    Corda 3 = A1
    Corda 4 = E1
    Corda 5 = B0 (mais grave)

Como usar standalone:
    from modulo_tablatura import print_tab_multiplas_notas
    print_tab_multiplas_notas([[3,1,0],[4,7,1],[3,2,2]])
"""


_NOMES_CORDAS_5 = {1: "G", 2: "D", 3: "A", 4: "E", 5: "B"}
_NOMES_CORDAS_4 = {1: "G", 2: "D", 3: "A", 4: "E"}


def print_tab_multiplas_notas(
    sequencia_escolhida: list[list[int]],
    num_cordas: int = 5,
) -> str:
    """
    Imprime e retorna a tablatura ASCII para uma sequência de notas.

    Parâmetros
    ----------
    sequencia_escolhida : list[[corda, traste, tempo_idx]]
        Lista de notas no formato [corda, traste, tempo_idx].
        Produzida por bridge_caminho_para_tab().
        Notas com mesmo tempo_idx aparecem na mesma coluna (acorde/harmônico).
        tempo_idx == 0 no elemento com corda == 0 ativa o "pulo de tempo".
    num_cordas : int
        5 para baixo de 5 cordas (padrão), 4 para baixo de 4 cordas.

    Retorna
    -------
    str
        String da tablatura (também impressa no console).
    """
    nomes_cordas = _NOMES_CORDAS_5 if num_cordas == 5 else _NOMES_CORDAS_4
    tablatura    = {i: f"{nomes_cordas[i]}|--" for i in range(1, num_cordas + 1)}

    if not sequencia_escolhida:
        resultado = "\n".join(tablatura[i] for i in range(1, num_cordas + 1))
        print(resultado)
        return resultado

    # Agrupa notas por tempo_idx
    blocos_tempo = []
    bloco_atual  = []
    tempo_ref    = sequencia_escolhida[0][2]

    for nota in sequencia_escolhida:
        corda, casa, tempo = nota
        if tempo == tempo_ref:
            bloco_atual.append((corda, casa))
        else:
            blocos_tempo.append(bloco_atual)
            bloco_atual = [(corda, casa)]
            tempo_ref   = tempo
    blocos_tempo.append(bloco_atual)

    # Renderiza cada bloco
    for bloco in blocos_tempo:
        pular_tempo = any(n[0] == 0 for n in bloco)

        if pular_tempo:
            for c in range(1, num_cordas + 1):
                tablatura[c] += "---"
            continue

        largura       = max(len(str(n[1])) for n in bloco) if bloco else 1
        notas_no_bloco = {n[0]: n[1] for n in bloco}

        for c in range(1, num_cordas + 1):
            if c in notas_no_bloco:
                casa      = str(notas_no_bloco[c])
                tablatura[c] += casa + "-" * (largura - len(casa) + 1)
            else:
                tablatura[c] += "-" * (largura + 1)

        for c in range(1, num_cordas + 1):
            tablatura[c] += "-"

    linhas    = [tablatura[i] for i in range(1, num_cordas + 1)]
    resultado = "\n".join(linhas)
    print(resultado)
    return resultado
