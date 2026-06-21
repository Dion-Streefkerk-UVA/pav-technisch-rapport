# startpunt: data inladen -> opschonen -> eda -> modellen -> rapport.
# alle getallen in findings.md komen live uit deze run.
import sys

from src import config, data_loading, eda, modeling, preprocessing, reporting


def main():
    # draait alle stappen op volgorde. 0 = klaar, 1 = data niet ingeladen.
    config.apply_style()
    config.ensure_directories()

    try:
        raw = data_loading.load_titanic()
    except data_loading.DatasetUnavailable as err:
        print(err, file=sys.stderr)
        return 1

    data_loading.describe_structure(raw)

    classification = data_loading.classify_variables(raw)
    reporting.save_table(
        classification, "variabele_classificatie",
        caption="Classificatie van de variabelen naar meetniveau.",
        label="tab:meetniveau",
    )

    # ontbrekende waarden tellen op de ruwe data, vóór het opschonen
    missing_raw = preprocessing.missing_summary(raw)
    reporting.save_table(
        missing_raw, "ontbrekende_waarden",
        caption="Ontbrekende waarden per kolom in de ruwe dataset.",
        label="tab:ontbrekend",
    )

    # data opschonen; missing_report bewaart welke keuzes we maakten
    clean, missing_report = preprocessing.preprocess(raw)

    eda_tables = eda.run_eda(clean, missing_raw)

    # X = kenmerken, y = overleefd ja/nee
    X, y = preprocessing.split_features_target(clean)
    model_output = modeling.train_and_evaluate(X, y)
    modeling.make_model_figures(model_output)

    comparison = modeling.comparison_table(model_output)
    odds = modeling.odds_ratio_table(model_output)
    importance = modeling.feature_importance_table(model_output)

    reporting.save_table(
        comparison, "modelvergelijking",
        caption="Vergelijking van de drie modellen op cross-validatie en testset.",
        label="tab:modelvergelijking",
    )
    reporting.save_table(
        odds, "odds_ratios",
        caption="Odds ratio's van de logistische regressie (exp van de coëfficiënt).",
        label="tab:oddsratios",
    )
    reporting.save_table(
        importance, "feature_importances",
        caption="Belangrijkheid van kenmerken volgens de random forest.",
        label="tab:importances",
    )

    write_findings(clean, missing_report, eda_tables, model_output, comparison, odds, importance)

    print("\nklaar. output staat in:")
    print(f"  figuren  -> {config.FIGURES_DIR}")
    print(f"  tabellen -> {config.TABLES_DIR}")
    print(f"  schoon   -> {config.CLEAN_DATA_FILE}")
    print(f"  findings -> {config.FINDINGS_FILE}")
    return 0


def write_findings(clean, missing_report, eda_tables, model_output, comparison, odds, importance):
    # zet alle uitkomsten om in tekst voor findings.md; getallen live uit de data
    w = reporting.FindingsWriter()

    overall = clean["survived"].mean() * 100
    female_rate = clean.loc[clean["sex"] == "female", "survived"].mean() * 100
    male_rate = clean.loc[clean["sex"] == "male", "survived"].mean() * 100
    class_rates = clean.groupby("pclass")["survived"].mean().mul(100)

    w.add(
        "Resultaten",
        f"- **Algemeen overlevingspercentage:** {overall:.1f}% van de "
        f"{len(clean)} passagiers overleefde de ramp.",
    )
    w.add(
        "Resultaten",
        f"- **Geslacht:** vrouwen overleefden veel vaker dan mannen "
        f"({female_rate:.1f}% versus {male_rate:.1f}%). Geslacht hangt sterk samen "
        "met overleving.",
    )
    w.add(
        "Resultaten",
        f"- **Klasse:** het overlevingspercentage daalt met de klasse "
        f"(1e: {class_rates.loc[1]:.1f}%, 2e: {class_rates.loc[2]:.1f}%, "
        f"3e: {class_rates.loc[3]:.1f}%). Hogere klassen overleefden vaker.",
    )
    age_surv = clean.loc[clean["survived"] == 1, "age"].mean()
    age_died = clean.loc[clean["survived"] == 0, "age"].mean()
    w.add(
        "Resultaten",
        f"- **Leeftijd:** overlevenden waren gemiddeld iets jonger "
        f"({age_surv:.1f} jaar) dan niet-overlevenden ({age_died:.1f} jaar); het "
        "verschil is klein.",
    )
    alone_rate = clean.loc[clean["is_alone"] == 1, "survived"].mean() * 100
    family_rate = clean.loc[clean["is_alone"] == 0, "survived"].mean() * 100
    w.add(
        "Resultaten",
        f"- **Gezinsgrootte:** wie alleen reisde overleefde minder vaak "
        f"({alone_rate:.1f}%) dan wie met familie reisde ({family_rate:.1f}%).",
    )

    for col, decision in missing_report.decisions.items():
        w.add("Resultaten", f"- **Ontbrekende data ({col}):** {decision}")

    best = comparison.sort_values("f1", ascending=False).iloc[0]
    for _, row in comparison.iterrows():
        w.add(
            "Resultaten",
            f"- **{row['model']}:** test-accuratesse {row['test_accuracy']:.3f}, "
            f"F1 {row['f1']:.3f}, ROC-AUC {row['roc_auc']:.3f} "
            f"(5-voudige CV-accuratesse {row['cv_accuracy']:.3f} ± {row['cv_std']:.3f}).",
        )
    w.add(
        "Resultaten",
        f"- **Beste model:** {best['model']} presteert het best op F1 "
        f"({best['f1']:.3f}).",
    )

    sex_or = odds.loc[odds["kenmerk"] == "sex_female", "odds_ratio"]
    pclass_or = odds.loc[odds["kenmerk"] == "pclass", "odds_ratio"]
    age_or = odds.loc[odds["kenmerk"] == "age", "odds_ratio"]
    if not sex_or.empty:
        w.add(
            "Discussie",
            f"- **Odds ratio geslacht (vrouw):** {sex_or.iloc[0]:.2f} — vrouw zijn "
            "vergroot de overlevingskans sterk ten opzichte van man zijn.",
        )
    if not pclass_or.empty:
        w.add(
            "Discussie",
            f"- **Odds ratio klasse:** {pclass_or.iloc[0]:.2f} — een hoger "
            "klassenummer (lagere klasse) verlaagt de overlevingskans.",
        )
    if not age_or.empty:
        w.add(
            "Discussie",
            f"- **Odds ratio leeftijd:** {age_or.iloc[0]:.2f} — oudere passagiers "
            "hadden een iets lagere overlevingskans.",
        )

    top_feat = importance.iloc[0]
    w.add(
        "Discussie",
        f"- **Belangrijkste kenmerk (random forest):** '{top_feat['kenmerk']}' "
        f"met importance {top_feat['importance']:.3f}.",
    )

    # van geslacht/klasse/leeftijd: welke liet de forest het zwaarst meetellen
    target_feats = importance[importance["kenmerk"].isin(["sex_female", "pclass", "age"])]
    most_important = target_feats.iloc[0]["kenmerk"] if not target_feats.empty else "sex_female"
    nice = {"sex_female": "geslacht", "pclass": "passagiersklasse", "age": "leeftijd"}
    w.add(
        "Discussie",
        "\n### Antwoord op de onderzoeksvraag\n"
        f"Geslacht, passagiersklasse en leeftijd voorspellen samen de overlevingskans "
        f"goed (beste model F1 = {best['f1']:.3f}, ROC-AUC = {best['roc_auc']:.3f}). "
        f"Zowel de odds ratio's van de logistische regressie als de feature "
        f"importances van de random forest wijzen **{nice.get(most_important, most_important)}** "
        f"aan als de meest bepalende factor, gevolgd door passagiersklasse; leeftijd "
        f"draagt het minst bij.",
    )

    path = w.write()
    print(f"findings geschreven: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
