# Pseudocode pipeline

_Taalneutrale pseudocode van de volledige analyse-pipeline. Sleutelwoorden:
ALS / DAN / ANDERS / HERHAAL / STOP._

```
START pipeline

  // 1. Inladen
  ALS lokale ruwe dataset bestaat DAN
      lees dataset uit lokaal bestand
  ANDERS
      probeer dataset te downloaden
      ALS download mislukt DAN
          toon instructie om dataset handmatig te plaatsen
          STOP
      sla gedownloade dataset lokaal op

  // 2. Inspecteren
  bepaal aantal rijen en kolommen
  bepaal per variabele het meetniveau (nominaal / ordinaal / discreet / continu / binair)
  sla classificatietabel op

  // 3. Ontbrekende waarden in kaart brengen
  bereken per kolom het aantal en percentage ontbrekende waarden
  sla overzicht ontbrekende waarden op

  // 4. Ontbrekende waarden behandelen
  verwijder kolom 'dek'                         // circa 77% ontbrekend
  ALS haven van inscheping ontbreekt DAN
      vul aan met de modus (meest voorkomende haven)
  HERHAAL voor elke groep van klasse x geslacht
      vul ontbrekende leeftijd aan met de mediaan van die groep

  // 5. Kenmerken afleiden
  gezinsgrootte = aantal broers/zussen/partners + aantal ouders/kinderen + 1
  ALS gezinsgrootte gelijk aan 1 DAN reist_alleen = waar ANDERS reist_alleen = onwaar

  // 6. Coderen
  zet geslacht om naar binaire indicator (vrouw = 1)
  zet haven van inscheping om naar indicatorkolommen
  sla schone dataset op

  // 7. Verkennende analyse
  maak figuur: ontbrekende waarden per kolom
  maak figuur: overlevingspercentage per geslacht
  maak figuur: overlevingspercentage per klasse
  maak figuur: leeftijdsverdeling gesplitst op overleving
  maak figuur: overlevingspercentage per gezinsgrootte
  maak figuur: correlatie-heatmap
  sla samenvattende statistiek en overlevingspercentages per groep op

  // 8. Splitsen
  splits gegevens in kenmerken (X) en doel (overleving)
  splits in trainset en testset (80 / 20, gestratificeerd op overleving, vaste seed)

  // 9. Modellen trainen en evalueren
  HERHAAL voor elk model in {logistische regressie, beslisboom, random forest}
      bereken 5-voudige cross-validatie-accuratesse op de trainset
      train model op de volledige trainset
      voorspel op de testset
      bereken accuratesse, precisie, recall, F1, ROC-AUC
      bewaar confusiematrix
  maak figuren: confusiematrices, modelvergelijking

  // 10. Interpreteren
  bereken odds ratio's uit de logistische regressie (exp van de coefficient)
  bereken belangrijkheid van kenmerken uit de random forest
  maak figuren: odds ratio's, belangrijkheid van kenmerken
  sla modelvergelijking, odds ratio's en belangrijkheid op

  // 11. Rapporteren
  schrijf bevindingen weg met de werkelijk berekende getallen
  bepaal meest bepalende factor uit odds ratio's en belangrijkheid

STOP
```
