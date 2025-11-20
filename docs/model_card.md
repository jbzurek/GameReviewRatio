# Model Card
## Problem & Intended Use
Przewidywanie odsetek pozytywnych recenzji gier\
Docelowe zastosowanie:
szybka ocena jakości gry bez czytania tysięcy komentarzy,\
wsparcie dla twórców gier przy analizie rynku,\
analizy porównawcze (np. jakie cechy gry podnoszą ocenę społeczności).\
Model nie jest przeznaczony do zastosowań krytycznych, 
finansowych ani automatycznego podejmowania decyzji biznesowych bez nadzoru człowieka
## Data (source, license, size, PII=no)
Source = recenzje gier z na podstawie metadanych plarformy Steam,\
License = dane ogólnodostępne (Public Domain / Steam public metadata),\
Size = sample 100 rekordów, pełny zbiór docelowy 5000 gier,\
PII: brak danych wrażliwych, użytkownicy ani recenzenci nie są w datasetach
## Metrics (main + secondary, test split)
Metryka główna: RMSE,\
Metryki Pomocnicze: MAE R²,\
podział danych testowych i treningowych 20/80 podzielone losowo
## Limitations & Risks
Ograniczenia modelu:
dane o grach mogą być niekompletne lub niespójne (różne struktury metadanych),\
model nie rozumie treści recenzji – bazuje wyłącznie na metadanych,\
dane mogą mieć bias spowodowane przez popularność gatunków lub studiów,\
brak pełnej reprezentacji gier indie – możliwe obniżenie jakości predykcji.\
Ryzyka:
błędne wnioski biznesowe, jeśli model używany jest bez nadzoru,\
zbyt mały dataset w wersji dev może dawać zbyt optymistyczne/niestabilne wyniki.\
Mitigacje:
rozszerzenie zbioru danych do pełnych 5000 rekordów,\
monitoring jakości (W&B),\
ponowne trenowanie modelu co pewien czas
## Versioning
- W&B run: <https://wandb.ai/zurek-jakub-polsko-japo-ska-akademia-technik-komputerowych/gamereviewratio>
- Model artifact: gamereviewratio/ag_model:production (v3) <link>
- Code: commit 4f7a2c9 <kedro run>
- Data: clean_data:v12 <https://www.kaggle.com/datasets/artermiloff/steam-games-dataset?resource=download&select=games_march2025_full.csv>
- Env: Python 3.11, AutoGluon 1.x, sklearn 1.5, Kedro 1.0