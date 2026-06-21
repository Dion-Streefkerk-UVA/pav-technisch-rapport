# Titanic survival analysis (PAV — UvA AI jaar 1)

Reproduceerbare data-analyse-pipeline rond de onderzoeksvraag:

> *"In hoeverre kunnen geslacht, passagiersklasse en leeftijd de overlevingskans
> op de Titanic voorspellen, en welke van deze factoren is het meest bepalend?"*

De code is in het Engels; alle output die in het verslag belandt (figuren,
tabellen, `findings.md`, `pseudocode.md`, LaTeX) is in het Nederlands.

## Installatie

```bash
pip install -r requirements.txt
```

## Uitvoeren

```bash
python main.py
```

Dit draait de volledige pipeline end-to-end en regenereert alle output. De
dataset wordt eenmalig via `seaborn` opgehaald en lokaal opgeslagen in
`data/raw/titanic.csv`; daarna draait alles offline. Lukt de download niet, dan
print het script een instructie om zelf een `titanic.csv` in `data/raw/` te
plaatsen en stopt netjes.

Vaste random seed = 42 voor alle splitsingen en modellen (reproduceerbaar).

## Wat wordt er gegenereerd

| Pad | Inhoud |
|-----|--------|
| `data/raw/titanic.csv` | Ruwe dataset (lokale kopie, 891 rijen) |
| `data/processed/titanic_clean.csv` | Geïmputeerde, gecodeerde dataset |
| `outputs/figures/*.pdf` | Alle figuren (vector-PDF, kleurenblind-veilig) |
| `outputs/tables/*.csv` + `*.tex` | Alle tabellen (CSV én LaTeX) |
| `findings.md` | Bevindingen met de werkelijk berekende getallen |
| `pseudocode.md` | Pseudocode van de pipeline (Methode-sectie) |

### Figuren

1. `01_ontbrekende_waarden` — % ontbrekend per kolom
2. `02_overleving_per_geslacht`
3. `03_overleving_per_klasse`
4. `04_leeftijdsverdeling_overleving`
5. `05_overleving_per_gezinsgrootte`
6. `06_correlatie_heatmap`
7. `07_confusiematrices` — per model
8. `08_feature_importance` — random forest
9. `09_odds_ratios` — logistische regressie
10. `10_modelvergelijking`

## Structuur

```
src/
  config.py         paden, seed=42, kleurenblind-veilige stijl, opslaginstellingen
  data_loading.py   robuust inladen + variabele-classificatie
  eda.py            EDA-figuren en -tabellen
  preprocessing.py  ontbrekende data, codering, feature engineering
  modeling.py       3 modellen trainen, evalueren, interpreteren
  plotting.py       gedeelde stijl-helpers (consistente figuren)
  reporting.py      tabellen wegschrijven (CSV + LaTeX), findings.md opbouwen
main.py             één entry point voor de hele pipeline
```

## Methode (kort)

- **Ontbrekende data:** `deck` (~77%) verwijderd; `age` (~20%) geïmputeerd met de
  mediaan binnen klasse × geslacht (MAR-aanname); `embarked` (2) met de modus.
- **Feature engineering:** `family_size = sibsp + parch + 1`, `is_alone`.
- **Modellen:** logistische regressie, beslisboom, random forest. Gestratificeerde
  train/test-split (80/20), 5-voudige cross-validatie op de trainset, evaluatie op
  de testset (accuratesse, precisie, recall, F1, ROC-AUC, confusiematrix).
- **Interpretatie:** odds ratio's (logistische regressie) en feature importances
  (random forest).

Concrete getallen staan in `findings.md` na een run.
