# **Game Review Ratio**

Predykcja odsetka pozytywnych recenzji gier na podstawie metadanych Steam.
Projekt dostarcza gotowy pipeline ML oparty o Kedro oraz automatyczne eksperymenty AutoGluon.
Docelowo użytkownik końcowy otrzyma: **API (FastAPI)** oraz **UI (Streamlit)**.

---

# **SPIS TREŚCI**

* [Cel i zakres](#cel-i-zakres)
* [Architektura](#architektura-high-level)
* [Dane](#dane)
* [Potok (Kedro)](#potok-kedro)
* [Eksperymenty i wyniki (W&B)](#eksperymenty-i-wyniki-wb)
* [Model i Model Card](#model-i-model-card)
* [Środowisko i instalacja](#środowisko-i-instalacja)
* [Uruchomienie lokalne (API + UI)](#uruchomienie-lokalne-bez-dockera)
* [Docker i docker-compose](#docker-i-docker-compose)
* [Wdrożenie w chmurze (GCP Cloud Run)](#wdrożenie-w-chmurze-gcp-cloud-run)
* [Konfiguracja: ENV i sekrety](#konfiguracja-env-i-sekrety)
* [API (FastAPI)](#api-fastapi)
* [UI (Streamlit)](#ui-streamlit)
* [Baza danych (opcjonalnie)](#baza-danych-opcjonalnie)
* [Monitoring i diagnostyka](#monitoring-i-diagnostyka)
* [Testy i jakość](#testy-i-jakość)
* [Struktura repozytorium](#struktura-repozytorium)
* [Załączniki / linki](#załączniki--linki)

---

# **CEL I ZAKRES**

Celem projektu jest trenowanie i porównywanie modeli przewidujących **pct_pos_total** (odsetek pozytywnych recenzji gier Steam).
Projekt przeznaczony jest dla zespołów data science i developerów chcących przeprowadzać automatyczne eksperymenty ML.

**Ograniczenia:**

* dane pochodzą z jednego źródła (Kaggle – Steam Games Dataset)

---

# **ARCHITEKTURA (HIGH-LEVEL)**

Architektura logiczna projektu wygląda tak:

```
Steam Dataset
      ↓
Kedro Pipeline (load → clean → split → baseline + AutoGluon → evaluate → choose_best)
      ↓
Artefakty (model.pkl + metrics.json)
      ↓
W&B – eksperymenty, porównania, artefakty "production"
      ↓
[TODO] FastAPI (predict endpoint)
      ↓
[TODO] Streamlit UI (formularz predykcji)
      ↓
[TODO] GCP Cloud Run – hosting API i UI
```

Eksperymenty śledzone są w:
**W&B Dashboard:** [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

---

# **DANE**

**Źródło:** [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)

**Licencja:** MIT
**Data pobrania:** 07.10.2025

**Rozmiar:** próbka 100 wierszy (`data/01_raw/sample_100.csv`)

**Target:** `pct_pos_total`
**Cechy:** metadane gier: gatunki, tagi, platformy, języki, opisy, daty, wydawcy itd.

**PII:** brak (zbiór publiczny, opis produktów, nie użytkowników)

---

# **POTOK (KEDRO)**

Uruchomienie potoku:

```
kedro run
```

### Główne nody:

* `load_raw` – ładuje raw CSV
* `basic_clean` – czyszczenie danych (daty, NA, binarizacja, one-hot, MLB)
* `split_data` – podział na train/test
* `train_autogluon` – trening AutoGluon (eksperymenty)
* `evaluate_autogluon` – liczenie RMSE
* `train_baseline` – RandomForest, baseline
* `evaluate` – RMSE baseline’u
* `choose_best_model` – wybór najlepszego modelu

### Kluczowe pliki konfiguracji:

* `conf/base/catalog.yml`
* `conf/base/parameters.yml`

### Artefakty:

* `data/03_processed/` – przetworzone dane
* `data/06_models/` – baseline + autogluon
* `data/09_tracking/` – metryki JSON

Diagram (Kedro-Viz):

<p align="center">
  <img src="images/kedro-pipeline.svg" width="70%" />
</p>

---

# **EKSPERYMENTY I WYNIKI (W&B)**

Wszystkie eksperymenty widoczne są tutaj:  
[W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

Logowane są:

* RMSE
* czas treningu
* parametry (time_limit, presets, eval_metric, problem_type)
* artefakt modelu (alias: **candidate**)
* alias **production** dla wybranego modelu

**Główna metryka:** RMSE

---

# **MODEL I MODEL CARD**

Model produkcyjny (najlepszy run AutoGluon) zapisany jest jako:

```
data/06_models/ag_production.pkl
```

Model Card znajduje się w:

```
docs/model_card.md
```

Zawiera:

* opis problemu, danych, metryk,
* ograniczenia modelu,
* ryzyka i etyczne aspekty,
* informacje o wersjonowaniu (run ID, artifact alias `production`).

---

# **ŚRODOWISKO I INSTALACJA**

Wymagania:

* Python 3.11

Instalacja:

```
conda env create -f environment.yml
conda activate asi-ml
python -m ipykernel install --user --name asi-ml --display-name "Python (asi-ml)"
```

Kluczowe zależności:

* kedro
* autogluon.tabular
* scikit-learn
* pandas, numpy, pyarrow
* wandb
* ruff, black, pre-commit
* pytest

---

# **URUCHOMIENIE LOKALNE (BEZ DOCKERA)**

---

# **DOCKER I DOCKER-COMPOSE**



---

# **WDROŻENIE W CHMURZE (GCP CLOUD RUN)**

---

# **KONFIGURACJA: ENV I SEKRETY**

---

# **API (FASTAPI)**

---

# **UI (STREAMLIT)**

---

# **BAZA DANYCH (OPCJONALNIE)**

---

# **MONITORING I DIAGNOSTYKA**

---

# **TESTY I JAKOŚĆ**

Uruchamianie:

```
pytest -q
pre-commit run -a
```

Testy obejmują:

* czyszczenie danych (`basic_clean`)
* podział danych (`split_data`)
* logikę pipeline’u (podstawowe testy)

---

# **STRUKTURA REPOZYTORIUM**

```
src/
  gamereviewratio/
    pipelines/
      data_science/     # pipeline Kedro
      evaluation/       # logika czyszczenia, splitu, trenowania

conf/
  base/
    catalog.yml
    parameters.yml

data/
  01_raw/
  02_interim/
  03_processed/
  06_models/
  09_tracking/

docs/
  model_card.md

tests/
  test_clean_node.py
  test_split_node.py
  test_pipeline.py

images/
  kedro-pipeline.svg
```

---

# **ZAŁĄCZNIKI / LINKI**

* **W&B Project:** [W&B GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)
* **Artefakty modelu** — dostępne w W&B
* **Model Card:** `docs/model_card.md`
* **Diagram potoku:** `images/kedro-pipeline.svg`

---

