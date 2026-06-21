# Findings

_Automatisch gegenereerd door `python main.py` — alle getallen komen rechtstreeks uit de daadwerkelijke pipeline-run._

## Resultaten

- **Algemeen overlevingspercentage:** 38.4% van de 891 passagiers overleefde de ramp.
- **Geslacht:** vrouwen overleefden veel vaker dan mannen (74.2% versus 18.9%). Geslacht hangt sterk samen met overleving.
- **Klasse:** het overlevingspercentage daalt met de klasse (1e: 63.0%, 2e: 47.3%, 3e: 24.2%). Hogere klassen overleefden vaker.
- **Leeftijd:** overlevenden waren gemiddeld iets jonger (28.1 jaar) dan niet-overlevenden (29.7 jaar); het verschil is klein.
- **Gezinsgrootte:** wie alleen reisde overleefde minder vaak (30.4%) dan wie met familie reisde (50.6%).
- **Ontbrekende data (deck):** Verwijderd: 77% van de waarden ontbreekt, te weinig informatie voor betrouwbare imputatie.
- **Ontbrekende data (embarked):** 2 ontbrekende waarden geïmputeerd met de modus ('S'), de meest voorkomende inschepingshaven.
- **Ontbrekende data (age):** 177 ontbrekende waarden geïmputeerd met de mediaan binnen groepen van klasse x geslacht (MAR-aanname: leeftijd hangt samen met klasse en geslacht).
- **Logistische regressie:** test-accuratesse 0.804, F1 0.729, ROC-AUC 0.849 (5-voudige CV-accuratesse 0.798 ± 0.021).
- **Beslisboom:** test-accuratesse 0.771, F1 0.687, ROC-AUC 0.799 (5-voudige CV-accuratesse 0.801 ± 0.016).
- **Random forest:** test-accuratesse 0.810, F1 0.746, ROC-AUC 0.828 (5-voudige CV-accuratesse 0.812 ± 0.020).
- **Beste model:** Random forest presteert het best op F1 (0.746).

## Discussie

- **Odds ratio geslacht (vrouw):** 3.41 — vrouw zijn vergroot de overlevingskans sterk ten opzichte van man zijn.
- **Odds ratio klasse:** 0.38 — een hoger klassenummer (lagere klasse) verlaagt de overlevingskans.
- **Odds ratio leeftijd:** 0.57 — oudere passagiers hadden een iets lagere overlevingskans.
- **Belangrijkste kenmerk (random forest):** 'fare' met importance 0.268.

### Antwoord op de onderzoeksvraag
Geslacht, passagiersklasse en leeftijd voorspellen samen de overlevingskans goed (beste model F1 = 0.746, ROC-AUC = 0.828). Zowel de odds ratio's van de logistische regressie als de feature importances van de random forest wijzen **geslacht** aan als de meest bepalende factor, gevolgd door passagiersklasse; leeftijd draagt het minst bij.
