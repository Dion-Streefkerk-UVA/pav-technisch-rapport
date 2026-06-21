# verkennende analyse (eda): figuren en tabellen die samenvatten wie overleefde.
# staaf-assen beginnen op 0 en alles wordt als vector-pdf opgeslagen.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config, plotting, reporting

SURVIVAL_LABELS = {0: "Niet overleefd", 1: "Overleefd"}
SEX_LABELS = {"male": "Man", "female": "Vrouw"}


def plot_missing_data(missing_summary):
    # staafdiagram van ontbrekende waarden per kolom; lege kolommen weglaten
    data = missing_summary[missing_summary["percentage_missend"] > 0]
    fig, ax = plotting.new_axes(figsize=(8, 5))
    if data.empty:
        # niks ontbreekt -> tekstje ipv lege grafiek
        ax.text(0.5, 0.5, "Geen ontbrekende waarden", ha="center", va="center")
    else:
        sns.barplot(
            data=data, x="percentage_missend", y="kolom",
            color=config.COLORBLIND_PALETTE[0], ax=ax,
        )
        for container in ax.containers:
            ax.bar_label(
                container,
                labels=[f"{v:.1f}%" for v in data["percentage_missend"]],
                padding=3, fontsize=9,
            )
        ax.set_xlim(0, max(100, data["percentage_missend"].max() * 1.15))
    ax.set_title("Percentage ontbrekende waarden per kolom")
    ax.set_xlabel("Ontbrekend (%)")
    ax.set_ylabel("Kolom")
    config.save_figure(fig, "01_ontbrekende_waarden")


def plot_survival_by_sex(df):
    # overlevingspercentage per geslacht (gemiddelde van 0/1 = percentage)
    rates = df.groupby("sex")["survived"].mean().mul(100).reset_index()
    rates["geslacht"] = rates["sex"].map(SEX_LABELS)
    fig, ax = plotting.new_axes(figsize=(6, 5))
    sns.barplot(
        data=rates, x="geslacht", y="survived",
        hue="geslacht", palette=config.COLORBLIND_PALETTE[:len(rates)], legend=False, ax=ax,
    )
    plotting.annotate_bars(ax, as_percentage=True)
    ax.set_title("Overlevingspercentage per geslacht")
    ax.set_xlabel("Geslacht")
    ax.set_ylabel("Overlevingspercentage (%)")
    plotting.finalize_bar_axis(ax, ymax=100)
    config.save_figure(fig, "02_overleving_per_geslacht")


def plot_survival_by_class(df):
    # overlevingspercentage per passagiersklasse (1e/2e/3e)
    rates = df.groupby("pclass")["survived"].mean().mul(100).reset_index()
    rates["klasse"] = rates["pclass"].map({1: "1e klasse", 2: "2e klasse", 3: "3e klasse"})
    fig, ax = plotting.new_axes(figsize=(6, 5))
    sns.barplot(
        data=rates, x="klasse", y="survived",
        hue="klasse", palette=config.COLORBLIND_PALETTE[:len(rates)], legend=False, ax=ax,
    )
    plotting.annotate_bars(ax, as_percentage=True)
    ax.set_title("Overlevingspercentage per passagiersklasse")
    ax.set_xlabel("Passagiersklasse")
    ax.set_ylabel("Overlevingspercentage (%)")
    plotting.finalize_bar_axis(ax, ymax=100)
    config.save_figure(fig, "03_overleving_per_klasse")


def plot_age_distribution(df):
    # histogram van leeftijd, gesplitst op wel/niet overleefd (met kde-lijn)
    fig, ax = plotting.new_axes(figsize=(8, 5))
    plot_df = df.copy()
    plot_df["Overleving"] = plot_df["survived"].map(SURVIVAL_LABELS)
    sns.histplot(
        data=plot_df, x="age", hue="Overleving", kde=True, bins=30,
        palette=config.COLORBLIND_PALETTE[:2], element="step", stat="count", ax=ax,
    )
    ax.set_title("Leeftijdsverdeling gesplitst op overleving")
    ax.set_xlabel("Leeftijd (jaren)")
    ax.set_ylabel("Aantal passagiers")
    config.save_figure(fig, "04_leeftijdsverdeling_overleving")


def plot_survival_by_family_size(df):
    # overlevingspercentage per gezinsgrootte (lijngrafiek, family_size loopt op)
    rates = df.groupby("family_size")["survived"].mean().mul(100).reset_index()
    fig, ax = plotting.new_axes(figsize=(8, 5))
    sns.lineplot(
        data=rates, x="family_size", y="survived",
        marker="o", color=config.COLORBLIND_PALETTE[0], ax=ax,
    )
    ax.set_title("Overlevingspercentage per gezinsgrootte")
    ax.set_xlabel("Gezinsgrootte (aantal personen)")
    ax.set_ylabel("Overlevingspercentage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(rates["family_size"].unique()))
    config.save_figure(fig, "05_overleving_per_gezinsgrootte")


def plot_correlation_heatmap(df):
    # correlatie-heatmap van numerieke kolommen (1 = samen op, -1 = tegengesteld)
    numeric = df.select_dtypes(include="number")
    # one-hot embarked_-kolommen eruit, anders wordt de heatmap onleesbaar
    drop_cols = [c for c in numeric.columns if c.startswith("embarked_")]
    numeric = numeric.drop(columns=drop_cols, errors="ignore")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        vmin=-1, vmax=1, square=True, cbar_kws={"label": "Correlatie (r)"}, ax=ax,
    )
    ax.set_title("Correlatie-heatmap van numerieke kenmerken")
    config.save_figure(fig, "06_correlatie_heatmap")


def summary_statistics_table(df):
    # samenvattende statistiek per numerieke kolom (gemiddelde, std, kwartielen, min/max)
    cols = [c for c in ["age", "fare", "family_size", "sibsp", "parch"] if c in df.columns]
    desc = df[cols].describe().T
    desc = desc.rename(
        columns={
            "count": "n", "mean": "gemiddelde", "std": "std", "min": "min",
            "25%": "p25", "50%": "mediaan", "75%": "p75", "max": "max",
        }
    )
    desc.index.name = "variabele"
    return desc.reset_index()


def survival_rates_table(df):
    # overlevingspercentage per geslacht én klasse, met het aantal per groep
    grouped = (
        df.groupby(["sex", "pclass"])["survived"]
        .agg(aantal="count", overlevingspercentage="mean")
        .reset_index()
    )
    grouped["overlevingspercentage"] = (grouped["overlevingspercentage"] * 100).round(1)
    grouped["geslacht"] = grouped["sex"].map(SEX_LABELS)
    grouped = grouped[["geslacht", "pclass", "aantal", "overlevingspercentage"]]
    return grouped.rename(columns={"pclass": "klasse"})


def run_eda(df_clean, missing_summary_raw):
    # draait alle eda-plots en -tabellen en schrijft de tabellen weg
    print("eda-figuren maken...")
    plot_missing_data(missing_summary_raw)
    plot_survival_by_sex(df_clean)
    plot_survival_by_class(df_clean)
    plot_age_distribution(df_clean)
    plot_survival_by_family_size(df_clean)
    plot_correlation_heatmap(df_clean)

    print("eda-tabellen wegschrijven...")
    desc = summary_statistics_table(df_clean)
    rates = survival_rates_table(df_clean)

    reporting.save_table(
        desc, "samenvattende_statistiek",
        caption="Samenvattende statistiek van de numerieke kenmerken.",
        label="tab:beschrijvend",
    )
    reporting.save_table(
        rates, "overlevingspercentages_per_groep",
        caption="Overlevingspercentages per geslacht en passagiersklasse.",
        label="tab:overlevingsgroepen",
    )
    return {"summary": desc, "survival_rates": rates}
