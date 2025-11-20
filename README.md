# Game Review Ratio

Dane: [Steam Games Dataset (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_full.csv)
**Licencja:** MIT
**Data pobrania:** 07.10.2025 r.

---

### Cel projektu
Celem projektu jest predykcja odsetka pozytywnych recenzji gier (pct_pos_total) na podstawie danych o grach dostępnych na platformie Steam.
Projekt został zrealizowany z użyciem frameworka Kedro, a eksperymenty śledzone są w Weights & Biases (W&B).

---

### Wybrana metryka: RMSE
Metryka RMSE (Root Mean Squared Error) została wybrana, ponieważ silniej karze duże błędy predykcji.
Dzięki temu lepiej uwzględnia gry o nietypowych wynikach i poprawia ocenę jakości modelu regresyjnego.

---

### Weights & Biases
Monitorowanie eksperymentów i wyników:
[W&B Dashboard – GameReviewRatio](https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio)

---

### Uruchomienie pipeline’u Kedro
Aby uruchomić cały pipeline:
```
kedro run
```

---

### Struktura pipeline’u

Poniżej znajduje się wizualizacja przepływu danych i zadań w projekcie:

<p align="center"> <img src="images/kedro-pipeline.svg" alt="Kedro pipeline" width="70%"> </p>
