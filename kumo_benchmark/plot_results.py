#Plot paired test scores using the recorded training draws##

from pathlib import Path


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.ticker import FuncFormatter

import pandas as pd


OUT = Path(__file__).resolve().parents[1] / "results"


def paired_chart(frame, folder, model, variant, filename, english=False):
    scores = frame.pivot(index="seed", columns=["model", "labels"], values="roc_auc")

    kumo = scores["Kumo Small", 200]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), facecolor="#fffaf0")

    fig.subplots_adjust(left=0.21, right=0.91, top=0.765, bottom=0.145, hspace=0.78)

    def local(pt, en):
        return en if english else pt

    fig.text(
        0.07,
        0.948,
        local(f"Risco de crédito   /   Kumo Small e XGBoost {variant}", f"Credit risk   /   Kumo Small and {variant} XGBoost"),
        size=15,
        color="#424b43",
    )

    wins = int((kumo > scores[model, 200]).sum())

    rounds = (
        "nas três rodadas"
        if wins == len(kumo) == 3
        else f"em {wins} de {len(kumo)} rodadas"
    )

    fig.text(
        0.07,
        0.83,
        local(f"Com 200 clientes, o Kumo\nliderou {rounds}.", f"With 200 clients, Kumo led\nin {wins} of {len(kumo)} draws."),
        size=31,
        family="DejaVu Serif",
        linespacing=1.08,
        color="#172e28",
    )

    fig.text(
        0.07, 0.787, local("ROC-AUC: mais à direita = melhor ordenação do risco.", "ROC-AUC: farther right means better risk ranking."), size=15
    )

    fig.add_artist(
        Line2D(
            [0.07, 0.93],
            [0.474, 0.474],
            transform=fig.transFigure,
            color="#aaa99c",
            linewidth=0.9,
        )
    )

    fig.text(
        0.07,
        0.431,
        local("Kumo com 200 × XGBoost com 1.000", "Kumo with 200 vs. XGBoost with 1,000"),
        size=21,
        family="DejaVu Serif",
        color="#172e28",
    )

    wins = int((scores[model, 1000] > kumo).sum())
    summary = f"O XGBoost liderou em {wins} de {len(kumo)} rodadas."
    if len(kumo) == 3 and wins == 2:
        summary = "O XGBoost liderou em duas das três rodadas."
    elif (kumo > scores[model, 1000]).all() and len(kumo) == 3:
        summary = "O Kumo liderou nas três rodadas."
    summary = local(summary, f"XGBoost led in {wins} of {len(kumo)} draws.")
    fig.text(0.07, 0.395, summary, size=15, color="#283b35")
    for ax, budget in zip(axes, (200, 1000)):
        other = scores[model, budget]

        ax.set_facecolor("#fffaf0")

        for row, seed in enumerate(kumo.index):
            ax.plot(
                [kumo[seed], other[seed]], [row - 0.1, row + 0.1], color="#8a8898", lw=2
            )

        rows = pd.Series(range(len(kumo)))

        ax.scatter(
            kumo, rows - 0.1, marker="o", color="#16745d", s=132, zorder=3, linewidths=1
        )

        count = f"{budget:,}".replace(",", local(".", ","))

        ax.scatter(
            other,
            rows + 0.1,
            marker="D",
            color="#6842a3",
            s=105,
            zorder=3,
            linewidths=1,
        )

        ax.set(
            yticks=rows,
            yticklabels=[local(f"Rodada {i + 1}", f"Draw {i + 1}") for i in rows],
            ylim=(2.45, -0.82),
            xlim=(0.655, 0.792),
        )

        ax.annotate(
            "Kumo · 200",
            (kumo.iloc[0], -0.1),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color="#105c49",
            weight="bold",
            fontsize=16,
        )

        ax.annotate(
            f"XGBoost · {count}",
            (other.iloc[0], 0.1),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color="#613792",
            weight="bold",
            fontsize=16,
        )

        if budget == 1000:
            gap = kumo - other

            seed = gap.abs().idxmin()

            row = kumo.index.get_loc(seed)

            winner = "Kumo" if gap.loc[seed] > 0 else "XGBoost"

            margin = f"{abs(gap.loc[seed]):.4f}".replace(".", local(",", "."))

            note = (
                f"{winner} à frente por {margin}" if gap.loc[seed] else "Mesma ROC-AUC"
            )

            note = local(note, f"{winner} led by {margin}" if gap.loc[seed] else "Equal ROC-AUC")
            ax.annotate(
                note,
                xy=(min(kumo.loc[seed], other.loc[seed]), row),
                xytext=(0.657, row + 0.27),
                ha="left",
                va="center",
                fontsize=14,
                color="#283b35",
                bbox=dict(facecolor="#fffaf0", edgecolor="none", pad=2),
                arrowprops=dict(arrowstyle="-", color="#68716b", shrinkB=9),
            )

        ax.grid(axis="x", linestyle=":", color="#bfb7ab")

        ax.spines[:].set_visible(False)

        ax.tick_params(length=0, labelsize=13, labelbottom=True)

        ax.set_xticks([0.66, 0.70, 0.74, 0.78])

        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", local(",", ".")))
        )

    fig.text(0.07, 0.085, local("A cada rodada, muda a amostra de treino.", "Each draw uses a different training sample."), size=14)
    fig.text(0.07, 0.060, local("Os 6.000 clientes do teste são os mesmos.", "The same 6,000 clients form the test set."), size=14)
    fig.text(
        0.07,
        0.027,
        local("Fonte: UCI · Default of Credit Card Clients · Taiwan, 2005", "Source: UCI · Default of Credit Card Clients · Taiwan, 2005"),
        size=12,
    )
    fig.text(0.07, 0.006, "doi.org/10.24432/C55S3H", size=11, color="#356452")
    temp = folder / (filename + ".tmp")

    fig.savefig(temp, format="png", dpi=150)

    temp.replace(folder / filename)

    plt.close(fig)


def render(frame, folder=OUT):
    paired_chart(frame, folder, "XGBoost", "padrão", "kumo-vs-xgboost-padrao.png")


    paired_chart(frame, folder, "XGBoost", "default", "kumo-vs-xgboost-default-en.png", english=True)


def render_followup(frame, folder):
    paired_chart(
        frame,
        folder,
        "XGBoost regularized",
        "regularizado",
        "kumo-vs-xgboost-regularizado.png",
    )


    paired_chart(frame, folder, "XGBoost regularized", "regularized", "kumo-vs-xgboost-regularized-en.png", english=True)


def redraw():
    folder = OUT / "baseline"
    render(pd.read_csv(folder / "metrics.csv"), folder)

    folder = OUT / "regularization"

    render_followup(pd.read_csv(folder / "metrics.csv"), folder)
