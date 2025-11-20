# **Model Card — Game Review Ratio (AutoGluon Production Model)**

## **1. Problem & Intended Use**

Model służy do **predykcji odsetka pozytywnych recenzji gier (`pct_pos_total`)** na podstawie publicznych metadanych gier z platformy Steam.

### Docelowe zastosowania:

* szybka ocena jakości gry bez konieczności analizy tysięcy recenzji,
* wsparcie twórców gier przy analizie rynku i benchmarkowaniu,
* analizy danych: jakie elementy gry korelują z pozytywnym odbiorem,
* eksploracja trendów wśród gatunków, tagów, platform i innych metadanych.

### Niedozwolone / niezalecane zastosowania:

* decyzje finansowe, biznesowe lub inwestycyjne bez nadzoru człowieka,
* wykorzystywanie modelu do oceny indywidualnych użytkowników (dataset nie dotyczy użytkowników).

Model ma charakter **analityczny / wspierający**, nie decyzyjny.

---

## **2. Data (source, license, size, PII=no)**

### Źródło danych

Dane: [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

### Licencja

MIT (zgodnie z opisem datasetu).

### Rozmiar danych

* Wersja produkcyjna modelu trenowana na próbce ~100 rekordów,
* Docelowy pełny zbiór: ok. 5000 gier.

### Zawartość danych

* gatunki, tagi, kategorie,
* języki obsługiwane przez grę,
* platformy (Windows/Mac/Linux),
* wydawca, developer,
* data premiery,
* liczba recenzji i procent pozytywnych,
* opisy i dodatkowe metadane.

### PII

Brak danych osobowych.
Zbiór nie zawiera recenzentów ani ich identyfikatorów – wyłącznie dane o produktach.

---

## **3. Metrics**

### Główna metryka:

**RMSE (Root Mean Squared Error)**  
Powód: bardziej karze duże błędy predykcji i dobrze działa w regresji.

### Walidacja:

* podział danych: **80% train / 20% test**, losowy
* powtarzalność: `random_state = 42`

---

## **4. Limitations & Risks**

### Ograniczenia:

* mała próbka treningowa w wersji dev (ok. 100 gier) - możliwe niestabilności,
* model korzysta **wyłącznie z metadanych**, nie analizuje treści recenzji,
* różnorodność gier (AAA vs indie) może prowadzić do biasów,
* brak pełnej reprezentacji gier niszowych.

### Ryzyka:

* niewłaściwe decyzje biznesowe, jeśli model stosowany bez walidacji eksperckiej,
* zbyt optymistyczne wyniki na małym zbiorze treningowym,
* błędna interpretacja.

### Mitigacje:

* trenowanie na pełnym zbiorze (5000+ rekordów),
* regularny retraining i monitoring w W&B,
* zbieranie dodatkowych cech,
* analiza błędów,
* walidacja w rzeczywistych warunkach.

---

## **5. Versioning**

### W&B Run (Production Model):

[W&B Dashboard — GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

### Model Artifact:

`gamereviewratio/ag_model:production`

### Code version:

Commit: `---`
(`kedro run` wykonywany z tego commitu)

### Data version:

`clean_data`
Źródło: [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_full.csv)

### Environment:

* Python 3.11
* AutoGluon 1.x
* scikit-learn 1.5
* Kedro 1.0
* wandb
* pandas, numpy, pyarrow

---
