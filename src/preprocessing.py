# ruwe data klaarmaken: gaten opvullen, features maken, tekst -> getallen.
# missende waarden:
#   deck  ~77% leeg -> weggooien
#   age   ~20% leeg -> mediaan binnen klasse x geslacht
#   embarked  2 leeg -> modus (meest voorkomende haven)
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class MissingReport:
    # summary = de telling, decisions = onze keuzes per kolom in tekst
    summary: pd.DataFrame
    decisions: dict = field(default_factory=dict)


def missing_summary(df):
    # aantal + percentage missend per kolom, ergste bovenaan
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    return (
        pd.DataFrame(
            {
                "kolom": counts.index,
                "aantal_missend": counts.values,
                "percentage_missend": pct.values,
            }
        )
        .sort_values("percentage_missend", ascending=False)
        .reset_index(drop=True)
    )


def _impute_age(df):
    # lege leeftijden opvullen met de mediaan binnen klasse x geslacht
    df = df.copy()
    df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda s: s.fillna(s.median()))
    # vangnet: leeg groepje -> mediaan van de hele dataset
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def preprocess(df):
    # hele opschoon-pipeline: kolommen filteren, gaten opvullen, features maken,
    # tekst -> getallen. geeft de schone tabel + de keuzes per kolom terug.
    config.ensure_directories()
    decisions = {}

    # alleen de kolommen die we nodig hebben; de dubbele seaborn-kolommen eruit
    base_columns = [
        "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "deck",
    ]
    df = df[[c for c in base_columns if c in df.columns]].copy()

    # deck: ~77% leeg, dus weggooien (te weinig om op te vullen)
    if "deck" in df.columns:
        deck_pct = df["deck"].isna().mean() * 100
        df = df.drop(columns=["deck"])
        decisions["deck"] = (
            f"Verwijderd: {deck_pct:.0f}% van de waarden ontbreekt, te weinig "
            "informatie voor betrouwbare imputatie."
        )

    # embarked: maar 2 leeg, opvullen met de modus (meest voorkomende haven)
    if "embarked" in df.columns and df["embarked"].isna().any():
        n_missing = int(df["embarked"].isna().sum())
        mode_value = df["embarked"].mode(dropna=True).iloc[0]
        df["embarked"] = df["embarked"].fillna(mode_value)
        decisions["embarked"] = (
            f"{n_missing} ontbrekende waarden geïmputeerd met de modus "
            f"('{mode_value}'), de meest voorkomende inschepingshaven."
        )

    # age: ~20% leeg, opvullen per klasse x geslacht (zie _impute_age)
    if "age" in df.columns and df["age"].isna().any():
        n_missing = int(df["age"].isna().sum())
        df = _impute_age(df)
        decisions["age"] = (
            f"{n_missing} ontbrekende waarden geïmputeerd met de mediaan binnen "
            "groepen van klasse x geslacht (MAR-aanname: leeftijd hangt samen "
            "met klasse en geslacht)."
        )

    # extra features: family_size (familie + jezelf) en is_alone (1 = alleen)
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    # tekst -> getallen: sex_female (1 = vrouw) en one-hot voor embarked.
    # drop_first laat 1 haven weg als referentie (voorkomt collineariteit).
    df["sex_female"] = (df["sex"] == "female").astype(int)
    embarked_dummies = pd.get_dummies(df["embarked"], prefix="embarked", drop_first=True).astype(int)
    df = pd.concat([df, embarked_dummies], axis=1)

    # schone tabel opslaan; sex/embarked blijven hier tekst voor de eda
    df.to_csv(config.CLEAN_DATA_FILE, index=False)
    print(f"schone data opgeslagen: {config.CLEAN_DATA_FILE}")

    return df, MissingReport(summary=missing_summary(df), decisions=decisions)


# de kolommen die als input naar de modellen gaan (alleen getallen)
FEATURE_COLUMNS = [
    "pclass", "age", "fare", "sibsp", "parch",
    "family_size", "is_alone", "sex_female", "embarked_Q", "embarked_S",
]

TARGET_COLUMN = "survived"


def split_features_target(df):
    # splitst in X (kenmerken) en y (overleefd ja/nee)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[features].copy(), df[TARGET_COLUMN].copy()
