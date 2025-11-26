# **Model Card — Game Review Ratio (Production Model)**

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

## **2. Data (source, license, size, PII)**

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
Zbiór nie zawiera recenzentów ani ich identyfikatorów. Wyłącznie dane o produktach.

---

## **3. Metrics**

### Główna metryka:

**RMSE (Root Mean Squared Error)**
Powód: karze duże błędy predykcji i jest standardem dla regresji.

### Walidacja:

* podział danych: **80% train / 20% test**, losowy,
* powtarzalność: `random_state = 42`.

### Wyniki (model produkcyjny – baseline):

* **RMSE (test): ~7.05**
* **MAE:** –
* **R²:** –
  *(Baseline loguje tylko RMSE, zgodnie z implementacją).*

### Wyniki modeli porównawczych (AutoGluon):

| Eksperyment    | Parametry                                                  | RMSE  | MAE   | R²    |
| -------------- | ---------------------------------------------------------- | ----- | ----- | ----- |
| **AG – Exp 1** | `time_limit=30`, `medium_quality_faster_train`             | ~7.43 | ~5.69 | ~0.20 |
| **AG – Exp 2** | `time_limit=60`, `medium_quality`                          | ~7.43 | ~5.69 | ~0.20 |
| **AG – Exp 3** | `time_limit=120`, `high_quality_fast_inference_only_refit` | ~9.07 | ~7.61 | −0.19 |

### Wnioski:

* **Baseline jest najlepszy (najniższy RMSE)** i został wybrany jako model produkcyjny.
* AutoGluon nie uzyskał lepszych wyników na małej próbce danych (~100 gier).
* Eksperymenty AG 1 i 2 dają identyczne wyniki — dłuższy czas treningu nie poprawia jakości.
* Preset `high_quality_fast_inference_only_refit` (Exp 3) prowadzi do przeuczenia.

---

## **4. Limitations & Risks**

### Ograniczenia:

* mała próbka treningowa (ok. 100 gier) -> wysoka wariancja i ograniczona generalizacja,
* model korzysta wyłącznie z metadanych,
* różnorodność gier (AAA vs indie) może wprowadzać bias,
* mała reprezentacja gier niszowych i starych tytułów.

### Ryzyka:

* błędna interpretacja predykcji bez kontekstu dziedzinowego,
* fałszywe poczucie pewności co do jakości gry,
* ryzyko użycia modelu w celach decyzyjnych, do których nie został zaprojektowany.

### Mitigacje:

* trenowanie na pełnym zbiorze (5000+ gier),
* regularny retraining i monitoring w W&B,
* wzbogacenie cech o embeddingi tekstowe (opinie graczy),
* analiza błędów i odrzucanie przypadków o wysokiej niepewności.

---

## **5. Versioning**

### W&B Run (Production Model):

[W&B Dashboard — GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

### Model Artifact:

`gamereviewratio/baseline_model:production`

*(Model AutoGluon został zachowany jako kandydat, ale nie wybrany).*

### Code version:

Commit: `575d69d`
(`kedro run` wykonano z tego commitu)

### Data version:

Plik: `data/01_raw/sample_100.csv`
Źródło: Steam Games Dataset (Kaggle)

### Environment:

* Python 3.11
* AutoGluon 1.x (modele porównawcze)
* scikit-learn 1.5
* Kedro 1.0
* wandb
* pandas, numpy, pyarrow
