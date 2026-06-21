# drie modellen trainen, testen en uitleggen.
# eerst 5-fold cv, dan scores op de testset. logreg via odds ratios,
# forest via feature importances.
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config, plotting

MODEL_NAMES = {
    "logreg": "Logistische regressie",
    "tree": "Beslisboom",
    "forest": "Random forest",
}


@dataclass
class ModelResult:
    # het getrainde model + al z'n scores
    key: str
    name: str
    estimator: object
    cv_accuracy_mean: float
    cv_accuracy_std: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    roc_curve: tuple


@dataclass
class ModelingOutput:
    # alle modellen + de train/test-splits, om door te geven aan plots en tabellen
    results: dict
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list


def train_and_evaluate(X, y):
    # 80/20 train/test-split (stratify houdt de overleefd-verhouding gelijk),
    # dan de drie modellen trainen en scoren
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED,
    )

    # logreg krijgt een scaler ervoor (alle kolommen op dezelfde schaal),
    # zodat de optimalisatie netjes convergeert en coëfficiënten vergelijkbaar zijn
    models = {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)),
            ]
        ),
        "tree": DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_SEED),
        "forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}

    for key, model in models.items():
        # eerst 5-fold cv op de trainset, daarna 1x trainen op de hele trainset
        print(f"{MODEL_NAMES[key]} trainen...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)

        # voorspellingen + kansen (kansen nodig voor roc-auc en roc-curve)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results[key] = ModelResult(
            key=key,
            name=MODEL_NAMES[key],
            estimator=model,
            cv_accuracy_mean=float(cv_scores.mean()),
            cv_accuracy_std=float(cv_scores.std()),
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_proba),
            confusion=confusion_matrix(y_test, y_pred),
            roc_curve=(fpr, tpr),
        )

    return ModelingOutput(
        results=results,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )


def comparison_table(output):
    # scores van alle modellen naast elkaar in 1 tabel
    rows = []
    for res in output.results.values():
        rows.append(
            {
                "model": res.name,
                "cv_accuracy": round(res.cv_accuracy_mean, 3),
                "cv_std": round(res.cv_accuracy_std, 3),
                "test_accuracy": round(res.accuracy, 3),
                "precision": round(res.precision, 3),
                "recall": round(res.recall, 3),
                "f1": round(res.f1, 3),
                "roc_auc": round(res.roc_auc, 3),
            }
        )
    return pd.DataFrame(rows)


def odds_ratio_table(output):
    # logreg-coëfficiënten omzetten naar odds ratios (e^coef). >1 = meer kans, <1 = minder
    clf = output.results["logreg"].estimator.named_steps["clf"]
    coefs = clf.coef_[0]
    table = pd.DataFrame(
        {
            "kenmerk": output.feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    table = table.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    table["coefficient"] = table["coefficient"].round(3)
    table["odds_ratio"] = table["odds_ratio"].round(3)
    return table


def feature_importance_table(output):
    # feature importances van de forest, belangrijkste bovenaan
    forest = output.results["forest"].estimator
    table = pd.DataFrame(
        {"kenmerk": output.feature_names, "importance": forest.feature_importances_}
    )
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["importance"] = table["importance"].round(4)
    return table


def plot_confusion_matrices(output):
    # confusiematrix per model (goed vs fout voorspeld)
    n = len(output.results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    labels = ["Niet overleefd", "Overleefd"]
    for ax, res in zip(axes, output.results.values()):
        sns.heatmap(
            res.confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_title(res.name)
        ax.set_xlabel("Voorspeld")
        ax.set_ylabel("Werkelijk")
    fig.suptitle("Confusiematrices per model (testset)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    config.save_figure(fig, "07_confusiematrices")


def plot_feature_importance(output):
    # staafdiagram van de feature importances
    table = feature_importance_table(output)
    fig, ax = plotting.new_axes(figsize=(8, 5))
    sns.barplot(data=table, x="importance", y="kenmerk", color=config.COLORBLIND_PALETTE[2], ax=ax)
    ax.set_title("Belangrijkheid van kenmerken (random forest)")
    ax.set_xlabel("Relatieve belangrijkheid")
    ax.set_ylabel("Kenmerk")
    ax.set_xlim(0, table["importance"].max() * 1.15)
    config.save_figure(fig, "08_feature_importance")


def plot_odds_ratios(output):
    # staafdiagram van de odds ratios; stippellijn op 1 = geen effect
    table = odds_ratio_table(output)
    fig, ax = plotting.new_axes(figsize=(8, 5))
    sns.barplot(data=table, x="odds_ratio", y="kenmerk", color=config.COLORBLIND_PALETTE[0], ax=ax)
    # staven onder de 1 een andere kleur, leest wat fijner
    for bar, value in zip(ax.patches, table["odds_ratio"]):
        bar.set_color(config.COLORBLIND_PALETTE[0] if value >= 1 else config.COLORBLIND_PALETTE[5])
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Odds ratio's per kenmerk (logistische regressie)")
    ax.set_xlabel("Odds ratio (exp(coëfficiënt)); stippellijn = geen effect (OR=1)")
    ax.set_ylabel("Kenmerk")
    config.save_figure(fig, "09_odds_ratios")


def plot_model_comparison(output):
    # accuratesse en f1 van alle modellen naast elkaar
    table = comparison_table(output)
    long = table.melt(
        id_vars="model", value_vars=["test_accuracy", "f1"],
        var_name="metriek", value_name="score",
    )
    long["metriek"] = long["metriek"].map({"test_accuracy": "Accuratesse", "f1": "F1-score"})
    fig, ax = plotting.new_axes(figsize=(8, 5))
    sns.barplot(
        data=long, x="model", y="score", hue="metriek",
        palette=config.COLORBLIND_PALETTE[:2], ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    ax.set_title("Modelvergelijking op de testset")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Metriek")
    config.save_figure(fig, "10_modelvergelijking")


def make_model_figures(output):
    # maakt alle modelfiguren (confusie, importance, odds, vergelijking)
    print("modelfiguren maken...")
    plot_confusion_matrices(output)
    plot_feature_importance(output)
    plot_odds_ratios(output)
    plot_model_comparison(output)
