"""
COMPARADOR DE ESTRATÉGIAS POLIFÔNICAS
======================================
Executa todas as 4 estratégias nas mesmas 5 suites de teste e produz:
    1. Tabela detalhada no console (por suite e geral)
    2. Figura 1 — Heatmaps: P / R / F1 / Fantasmas (estratégias × suites)
    3. Figura 2 — Barras agrupadas: P, R, F1 por suite (todas estratégias)
    4. Figura 3 — Barras agrupadas: Fantasmas por suite (todas estratégias)
    5. Figura 4 — Radar: comparação global das 4 estratégias

Uso:
    python comparador_estrategias.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # ou "Qt5Agg" se TkAgg não estiver disponível
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass

# ── Suites e métricas do módulo de teste ─────────────────────────────────────
from modulo_teste_polifonico import (
    SR,
    suite_nota_unica_por_corda,
    suite_intervalos_simultaneos,
    suite_sobreposicao_parcial,
    suite_erro_oitava_critico,
    suite_fantasmas_e_silencio,
    calcular_metricas_polifonicas,
)

# ── Estratégias ───────────────────────────────────────────────────────────────
from estrategias_pyin import ESTRATEGIAS, CORES_ESTRATEGIAS


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUTURA DE RESULTADO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResumoPorSuite:
    nome:            str
    precision:       float
    recall:          float
    f1:              float
    fantasmas:       int
    oitavas_erradas: int
    n_casos:         int


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def rodar_estrategia_suite(fn, casos, nome_suite, num_cordas=5):
    """Executa fn em todos os casos de uma suite e devolve ResumoPorSuite."""
    resultados = []
    for audio, gab, nome in casos:
        segs = fn(audio.astype(np.float32), SR, num_cordas=num_cordas)
        res  = calcular_metricas_polifonicas(segs, gab, nome)
        resultados.append(res)

    # Para suites com gabarito vazio (silêncio/ruído), P=R=0 é
    # matematicamente correto (0/0→0); não distorce a média da suite
    return ResumoPorSuite(
        nome            = nome_suite,
        precision       = float(np.mean([r.precision for r in resultados])),
        recall          = float(np.mean([r.recall    for r in resultados])),
        f1              = float(np.mean([r.f1        for r in resultados])),
        fantasmas       = int(sum(r.notas_fantasma   for r in resultados)),
        oitavas_erradas = int(sum(r.oitavas_erradas  for r in resultados)),
        n_casos         = len(resultados),
    )


def rodar_comparacao(num_cordas=5, verbose=True):
    """
    Executa todas as estratégias em todas as suites.
    Retorna dict: nome_estrategia → list[ResumoPorSuite]
    (5 suites + 1 entrada 'GERAL' no final)
    """
    suites = [
        ("Nota Única",   suite_nota_unica_por_corda()),
        ("Intervalos",   suite_intervalos_simultaneos()),
        ("Sobreposição", suite_sobreposicao_parcial()),
        ("Erro Oitava",  suite_erro_oitava_critico()),
        ("Fantasmas",    suite_fantasmas_e_silencio()),
    ]

    todos_resultados = {}

    for nome_est, fn in ESTRATEGIAS.items():
        if verbose:
            print(f"\n{'═'*64}")
            print(f"  {nome_est}")
            print(f"{'═'*64}")

        resumos = []
        for nome_suite, casos in suites:
            if verbose:
                print(f"  [{nome_suite:<14}] {len(casos)} casos ...", end=" ", flush=True)
            r = rodar_estrategia_suite(fn, casos, nome_suite, num_cordas)
            resumos.append(r)
            if verbose:
                print(f"P={r.precision*100:5.1f}%  R={r.recall*100:5.1f}%  "
                      f"F1={r.f1*100:5.1f}%  fantasmas={r.fantasmas}")

        # Agrega "GERAL" — média ponderada pelo número de casos
        total_casos = sum(r.n_casos for r in resumos)
        geral = ResumoPorSuite(
            nome            = "GERAL",
            precision       = float(np.mean([r.precision for r in resumos])),
            recall          = float(np.mean([r.recall    for r in resumos])),
            f1              = float(np.mean([r.f1        for r in resumos])),
            fantasmas       = int(sum(r.fantasmas       for r in resumos)),
            oitavas_erradas = int(sum(r.oitavas_erradas for r in resumos)),
            n_casos         = total_casos,
        )
        resumos.append(geral)
        todos_resultados[nome_est] = resumos

        if verbose:
            print(f"  {'─'*58}")
            print(f"  {'GERAL':<16}  "
                  f"P={geral.precision*100:5.1f}%  R={geral.recall*100:5.1f}%  "
                  f"F1={geral.f1*100:5.1f}%  fantasmas={geral.fantasmas}")

    return todos_resultados


# ══════════════════════════════════════════════════════════════════════════════
# TABELA NO CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_tabela(resultados: dict):
    nomes_est   = list(resultados.keys())
    nomes_suite = [r.nome for r in list(resultados.values())[0]]

    largura_est = max(len(n) for n in nomes_est) + 2

    cabecalho = (f"  {'Estratégia':<{largura_est}}"
                 f"{'Precision':>11}{'Recall':>10}{'F1':>10}{'Fantasmas':>12}{'8vaErro':>9}")
    sep       = "  " + "─" * (len(cabecalho) - 2)

    for nome_suite in nomes_suite:
        eh_geral = nome_suite == "GERAL"
        titulo   = f"  ══ {nome_suite} " + ("═" * (len(cabecalho) - len(nome_suite) - 6))
        if eh_geral:
            print(f"\n{'═'*len(cabecalho)}")
        else:
            print(f"\n{titulo}")
        print(cabecalho)
        print(sep)

        for nome_est in nomes_est:
            resumo = next(r for r in resultados[nome_est] if r.nome == nome_suite)
            marcador = "▶" if eh_geral else " "
            print(f"{marcador} {nome_est:<{largura_est}}"
                  f"{resumo.precision*100:>10.1f}%"
                  f"{resumo.recall*100:>9.1f}%"
                  f"{resumo.f1*100:>9.1f}%"
                  f"{resumo.fantasmas:>11}"
                  f"{resumo.oitavas_erradas:>9}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — HEATMAPS (estratégias × suites)
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap(ax, mat, nomes_est, nomes_suite, titulo, fmt="{:.0f}%",
             vmin=0, vmax=100, cmap="RdYlGn", separar_ultima_col=True):
    """Plota heatmap com valores em cada célula. Última coluna = GERAL (borda mais grossa)."""
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(nomes_suite)))
    ax.set_xticklabels(nomes_suite, fontsize=10, rotation=25, ha="right")
    ax.set_yticks(range(len(nomes_est)))
    ax.set_yticklabels(nomes_est, fontsize=10)
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=10)

    # Valores em cada célula
    for i in range(len(nomes_est)):
        for j in range(len(nomes_suite)):
            val  = mat[i, j]
            brilho = (val - vmin) / max(vmax - vmin, 1e-6)
            cor  = "white" if brilho < 0.38 else "black"
            ax.text(j, i, fmt.format(val),
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=cor)

    # Linha vertical destacando a coluna GERAL (última)
    if separar_ultima_col and len(nomes_suite) > 1:
        n = len(nomes_suite)
        ax.axvline(x=n - 1.5, color="white", linewidth=3.0)
        ax.axvline(x=n - 1.5, color="#333333", linewidth=1.5, linestyle="--")

    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    return im


def plotar_heatmaps(resultados: dict):
    nomes_est   = list(resultados.keys())
    nomes_suite = [r.nome for r in list(resultados.values())[0]]
    n_est, n_s  = len(nomes_est), len(nomes_suite)

    def _mat(attr, escala=100.0):
        m = np.zeros((n_est, n_s))
        for i, nome in enumerate(nomes_est):
            for j, res in enumerate(resultados[nome]):
                m[i, j] = getattr(res, attr) * escala
        return m

    fig, axes = plt.subplots(2, 2, figsize=(22, 11))
    fig.suptitle("Comparação de Estratégias — Métricas por Suite\n"
                 "(última coluna = média geral de todas as suites)",
                 fontsize=14, fontweight="bold", y=1.01)

    _heatmap(axes[0, 0], _mat("precision"), nomes_est, nomes_suite,
             "Precision (%)", vmin=0, vmax=100)
    _heatmap(axes[0, 1], _mat("recall"),    nomes_est, nomes_suite,
             "Recall (%)",    vmin=0, vmax=100)
    _heatmap(axes[1, 0], _mat("f1"),        nomes_est, nomes_suite,
             "F1 (%)",        vmin=0, vmax=100)

    mat_fant = _mat("fantasmas", escala=1.0)
    vmax_f   = max(mat_fant.max(), 1.0)
    _heatmap(axes[1, 1], mat_fant, nomes_est, nomes_suite,
             "Notas Fantasma (total)", fmt="{:.0f}",
             vmin=0, vmax=vmax_f, cmap="RdYlGn_r")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("comparacao_heatmap.png", dpi=130, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — BARRAS AGRUPADAS: P / R / F1 por suite
# ══════════════════════════════════════════════════════════════════════════════

def plotar_barras_metricas(resultados: dict):
    nomes_est   = list(resultados.keys())
    nomes_suite = [r.nome for r in list(resultados.values())[0]]
    n_est, n_s  = len(nomes_est), len(nomes_suite)

    cores  = [CORES_ESTRATEGIAS[n] for n in nomes_est]
    x      = np.arange(n_s)
    width  = 0.76 / n_est                    # largura de cada barra
    offset = [(i - n_est / 2.0 + 0.5) * width for i in range(n_est)]

    metricas = [
        ("precision", "Precision (%)"),
        ("recall",    "Recall (%)"),
        ("f1",        "F1 (%)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(22, 15),
                              gridspec_kw={"hspace": 0.52})
    fig.suptitle("Precision / Recall / F1 por Suite e Estratégia\n"
                 "(barra mais à direita = média geral)",
                 fontsize=14, fontweight="bold")

    for ax, (attr, ylabel) in zip(axes, metricas):
        for i, (nome, cor) in enumerate(zip(nomes_est, cores)):
            vals = [getattr(r, attr) * 100 for r in resultados[nome]]
            bars = ax.bar(x + offset[i], vals, width * 0.90,
                          label=nome, color=cor, alpha=0.88,
                          edgecolor="white", linewidth=0.6)
            # Rótulo sobre cada barra (omite se < 3%)
            for bar, val in zip(bars, vals):
                if val >= 3:
                    ax.text(bar.get_x() + bar.get_width() / 2.0,
                            bar.get_height() + 0.8,
                            f"{val:.0f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold", color="#222222")

        # Linha separadora antes da coluna GERAL
        ax.axvline(x=n_s - 1.5, color="#777777", linewidth=1.5,
                   linestyle="--", alpha=0.6)
        ax.text(n_s - 1.5, 105, " GERAL →", fontsize=8, color="#555555",
                va="top", ha="left")

        ax.set_xticks(x)
        ax.set_xticklabels(nomes_suite, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92,
                  ncol=2, columnspacing=1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    plt.savefig("comparacao_barras_metricas.png", dpi=130, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — BARRAS: FANTASMAS por suite
# ══════════════════════════════════════════════════════════════════════════════

def plotar_barras_fantasmas(resultados: dict):
    nomes_est   = list(resultados.keys())
    nomes_suite = [r.nome for r in list(resultados.values())[0]]
    n_est, n_s  = len(nomes_est), len(nomes_suite)

    cores  = [CORES_ESTRATEGIAS[n] for n in nomes_est]
    x      = np.arange(n_s)
    width  = 0.76 / n_est
    offset = [(i - n_est / 2.0 + 0.5) * width for i in range(n_est)]

    fig, ax = plt.subplots(figsize=(22, 7))
    fig.suptitle("Total de Notas Fantasma por Suite e Estratégia\n"
                 "(menos é melhor — barra mais à direita = total geral)",
                 fontsize=14, fontweight="bold")

    for i, (nome, cor) in enumerate(zip(nomes_est, cores)):
        vals = [r.fantasmas for r in resultados[nome]]
        bars = ax.bar(x + offset[i], vals, width * 0.90,
                      label=nome, color=cor, alpha=0.88,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.15,
                        str(val), ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#222222")

    ax.axvline(x=n_s - 1.5, color="#777777", linewidth=1.5,
               linestyle="--", alpha=0.6)
    ax.text(n_s - 1.5, ax.get_ylim()[1] * 0.97,
            " TOTAL →", fontsize=8, color="#555555", va="top", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(nomes_suite, fontsize=11)
    ax.set_ylabel("Notas fantasma (contagem total da suite)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    plt.savefig("comparacao_barras_fantasmas.png", dpi=130, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — RADAR: comparação global
# ══════════════════════════════════════════════════════════════════════════════

def plotar_radar_global(resultados: dict):
    """
    Radar/aranha com 7 eixos, um por suite (P, R, F1 médios) + geral F1.
    Cada estratégia é uma linha colorida.
    """
    nomes_est = list(resultados.keys())

    # Eixos: F1 de cada suite + fantasmas invertidos (100-fant normalizado)
    nomes_suite = [r.nome for r in list(resultados.values())[0]]
    suites_sem_geral = [n for n in nomes_suite if n != "GERAL"]

    eixos = suites_sem_geral + ["Anti-Fantasmas"]
    n_eixos = len(eixos)

    # Ângulos igualmente espaçados, fecha o polígono
    angulos = np.linspace(0, 2 * np.pi, n_eixos, endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(11, 11),
                            subplot_kw={"projection": "polar"})
    fig.suptitle("Comparação Global das Estratégias — Radar\n"
                 "(F1 por suite + Anti-Fantasmas = 100 – fantasmas normalizados)",
                 fontsize=13, fontweight="bold", y=1.01)

    max_fant = max(
        resultados[n][-1].fantasmas   # índice -1 = GERAL
        for n in nomes_est
    )
    max_fant = max(max_fant, 1)

    for nome, cor in CORES_ESTRATEGIAS.items():
        if nome not in resultados:
            continue
        resumos = resultados[nome]
        f1_por_suite = [
            next(r for r in resumos if r.nome == s).f1 * 100
            for s in suites_sem_geral
        ]
        fant_geral   = resumos[-1].fantasmas
        anti_fant    = max(0.0, 100.0 - (fant_geral / max_fant) * 100.0)
        vals = f1_por_suite + [anti_fant]
        vals += vals[:1]    # fecha o polígono

        ax.plot(angulos, vals, "o-", linewidth=2.2, color=cor,
                label=nome, markersize=6, alpha=0.9)
        ax.fill(angulos, vals, alpha=0.10, color=cor)

    # Grid e rótulos
    ax.set_thetagrids(np.degrees(angulos[:-1]), eixos, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12),
              fontsize=10, framealpha=0.92)

    plt.tight_layout()
    plt.savefig("comparacao_radar.png", dpi=130, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 68)
    print("  COMPARADOR DE ESTRATÉGIAS — pYIN POLIFÔNICO")
    print("  4 estratégias × 5 suites × sinais sintéticos")
    print("█" * 68)

    resultados = rodar_comparacao(num_cordas=5, verbose=True)

    print("\n\n" + "█" * 68)
    print("  TABELA COMPLETA DE RESULTADOS")
    print("█" * 68)
    imprimir_tabela(resultados)

    print("\n\nGerando visualizações ...\n")
    plotar_heatmaps(resultados)
    plotar_barras_metricas(resultados)
    plotar_barras_fantasmas(resultados)
    plotar_radar_global(resultados)

    print("\n[✓] Comparação concluída.")
    print("    Gráficos salvos: comparacao_heatmap.png, "
          "comparacao_barras_metricas.png,")
    print("    comparacao_barras_fantasmas.png, comparacao_radar.png")


if __name__ == "__main__":
    np.random.seed(42)
    main()
