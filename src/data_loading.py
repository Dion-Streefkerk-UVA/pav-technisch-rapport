# titanic-data inladen en per kolom het meetniveau bepalen.
# 1x via seaborn ophalen en lokaal opslaan, daarna offline verder.
import sys

import pandas as pd

from . import config


class DatasetUnavailable(RuntimeError):
    # gegooid als de data niet lokaal staat én niet te downloaden is
    pass


def load_titanic():
    # lokale kopie eerst, anders downloaden via seaborn
    config.ensure_directories()

    if config.RAW_DATA_FILE.exists():
        print(f"lokale kopie inlezen: {config.RAW_DATA_FILE}")
        return pd.read_csv(config.RAW_DATA_FILE)

    print("geen lokale kopie, proberen te downloaden via seaborn...")
    try:
        import seaborn as sns

        df = sns.load_dataset("titanic")
    except Exception as exc:
        # download mislukt -> nette fout met uitleg hoe je het zelf oplost
        raise DatasetUnavailable(
            "Kon de titanic-dataset niet downloaden via seaborn.\n"
            f"Reden: {exc}\n\n"
            f"Zet zelf een 'titanic.csv' (891 rijen, standaard schema) in: {config.RAW_DIR}\n"
            "en draai daarna opnieuw 'python main.py'."
        ) from exc

    # lokaal opslaan zodat de download maar 1x nodig is
    df.to_csv(config.RAW_DATA_FILE, index=False)
    print(f"gedownload en opgeslagen: {config.RAW_DATA_FILE}")
    return df


def describe_structure(df):
    # snelle check: aantal rijen/kolommen en het type per kolom
    print(f"vorm: {df.shape[0]} rijen x {df.shape[1]} kolommen")
    print("dtypes:")
    print(df.dtypes.to_string())


# meetniveau + korte beschrijving per kolom (nominaal/ordinaal/continu enz.)
VARIABLE_CLASSIFICATION = {
    "survived": {"meetniveau": "binair", "beschrijving": "Overleefd (0=nee, 1=ja)"},
    "pclass": {"meetniveau": "ordinaal", "beschrijving": "Passagiersklasse (1, 2, 3)"},
    "sex": {"meetniveau": "nominaal", "beschrijving": "Geslacht"},
    "age": {"meetniveau": "continu", "beschrijving": "Leeftijd in jaren"},
    "sibsp": {"meetniveau": "discreet", "beschrijving": "Aantal broers/zussen/partners aan boord"},
    "parch": {"meetniveau": "discreet", "beschrijving": "Aantal ouders/kinderen aan boord"},
    "fare": {"meetniveau": "continu", "beschrijving": "Ticketprijs"},
    "embarked": {"meetniveau": "nominaal", "beschrijving": "Haven van inscheping (C, Q, S)"},
    "class": {"meetniveau": "ordinaal", "beschrijving": "Passagiersklasse (tekstueel)"},
    "who": {"meetniveau": "nominaal", "beschrijving": "Categorie (man/vrouw/kind)"},
    "adult_male": {"meetniveau": "binair", "beschrijving": "Volwassen man (waar/onwaar)"},
    "deck": {"meetniveau": "nominaal", "beschrijving": "Dek (grotendeels ontbrekend)"},
    "embark_town": {"meetniveau": "nominaal", "beschrijving": "Stad van inscheping"},
    "alive": {"meetniveau": "binair", "beschrijving": "Overleefd (tekstueel ja/nee)"},
    "alone": {"meetniveau": "binair", "beschrijving": "Reisde alleen (waar/onwaar)"},
}


def classify_variables(df):
    # tabel met per kolom: naam, dtype, meetniveau en beschrijving
    rows = []
    for column in df.columns:
        # onbekende kolom -> "onbekend" ipv crashen
        meta = VARIABLE_CLASSIFICATION.get(column, {"meetniveau": "onbekend", "beschrijving": ""})
        rows.append(
            {
                "variabele": column,
                "dtype": str(df[column].dtype),
                "meetniveau": meta["meetniveau"],
                "beschrijving": meta["beschrijving"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    try:
        data = load_titanic()
    except DatasetUnavailable as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    describe_structure(data)
    print(classify_variables(data).to_string(index=False))
