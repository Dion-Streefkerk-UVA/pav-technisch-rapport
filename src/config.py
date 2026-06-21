# alle vaste instellingen op 1 plek: paden, seed en plot-stijl
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# vaste seed zodat elke run dezelfde uitkomst geeft
RANDOM_SEED = 42

# paden relatief aan dit bestand, zodat het overal werkt
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

RAW_DATA_FILE = RAW_DIR / "titanic.csv"
CLEAN_DATA_FILE = PROCESSED_DIR / "titanic_clean.csv"

FINDINGS_FILE = PROJECT_ROOT / "findings.md"


def ensure_directories():
    # maakt de output-mappen aan als ze nog niet bestaan
    for directory in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# kleurenblind-vriendelijk palet (Wong, 2011)
COLORBLIND_PALETTE = [
    "#0072B2",  # blauw
    "#E69F00",  # oranje
    "#009E73",  # groen
    "#CC79A7",  # roze
    "#56B4E9",  # lichtblauw
    "#D55E00",  # rood
    "#F0E442",  # geel
    "#999999",  # grijs
]

# figuren opslaan als pdf (vector) en strak bijgesneden
SAVE_KWARGS = {
    "format": "pdf",
    "dpi": 300,
    "bbox_inches": "tight",
}


def apply_style():
    # zet 1x de huisstijl voor alle plots (palet, thema, opmaak)
    sns.set_theme(context="notebook", style="whitegrid", palette=COLORBLIND_PALETTE)
    sns.set_palette(COLORBLIND_PALETTE)
    matplotlib.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "savefig.format": "pdf",
            "pdf.fonttype": 42,  # tekst blijft selecteerbaar ipv plaatjes
            "font.size": 10,
        }
    )


def save_figure(fig, name):
    # figuur opslaan, sluiten (geheugen vrij) en het pad teruggeven
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    return path
