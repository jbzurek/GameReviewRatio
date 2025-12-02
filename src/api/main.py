import datetime as dt
import json
import os
from contextlib import asynccontextmanager
from typing import Dict

import joblib
import pandas as pd
from databases import Database
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings


# konfiguracja
class Settings(BaseSettings):
    # ścieżka do pliku modelu produkcyjnego
    MODEL_PATH: str = "data/06_models/production_model.pkl"
    # ścieżka do pliku z wymaganymi kolumnami
    REQUIRED_COLUMNS_PATH: str = "data/06_models/required_columns.json"
    # url do bazy danych
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/asi_baza"
    # wersja modelu
    MODEL_VERSION: str = "local-dev"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

database = Database(settings.DATABASE_URL)
model = None
required_columns: list[str] = []


class Features(BaseModel):
    # wszystkie feature'y są int64
    data: Dict[str, int]

    @field_validator("data")
    @classmethod
    def validate_required_columns(cls, v: Dict[str, int]) -> Dict[str, int]:
        # sprawdzenie, czy są wszystkie wymagane kolumny
        if required_columns:
            missing = set(required_columns) - set(v.keys())
            if missing:
                raise ValueError(f"brak wymaganych kolumn w payload: {sorted(missing)}")
        return v


class Prediction(BaseModel):
    # odpowiedź api
    prediction: float | str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # inicjalizacja modelu i bazy
    global model, database, required_columns

    # ładuje model, jeśli niezaładowany
    if model is None:
        if not os.path.isfile(settings.MODEL_PATH):
            raise RuntimeError(f"nie znaleziono pliku modelu w '{settings.MODEL_PATH}'")
        model = joblib.load(settings.MODEL_PATH)

    # wczytaj wymagane kolumny
    if os.path.isfile(settings.REQUIRED_COLUMNS_PATH):
        try:
            with open(settings.REQUIRED_COLUMNS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # wszystkie kolumny mają typ int64
                required_columns = (
                    data["columns"]
                    if isinstance(data, dict) and "columns" in data
                    else list(data)
                )
        except Exception:
            required_columns = []
    else:
        required_columns = []

    # połącz z bazą; fallback do sqlite, jeśli postgres nie działa
    try:
        await database.connect()
    except Exception:
        # sqlite jako fallback
        database = Database("sqlite+aiosqlite:///./predictions.db")
        await database.connect()

    await init_db()
    try:
        yield
    finally:
        await database.disconnect()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    # endpoint healthcheck
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
async def predict(payload: Features):
    # sprawdzenie, czy model jest gotowy
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="model nie został załadowany",
        )

    # budowa dataframe z danych wejściowych
    x = pd.DataFrame([payload.data])

    # dopasuj kolumny do modelu
    if required_columns:
        target_cols = list(required_columns)
        x = x.reindex(columns=target_cols, fill_value=0)

    # inferencja modelu
    try:
        pred = model.predict(x)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"inferencja nie powiodła się: {e}",
        )

    # rzutowanie predykcji
    try:
        pred_out: float | str = float(pred)
    except Exception:
        pred_out = str(pred)

    # zapis do bazy (text dla sqlite, jsonb dla postgres)
    await save_prediction(payload.data, pred_out, settings.MODEL_VERSION)

    return Prediction(
        prediction=pred_out,
        model_version=settings.MODEL_VERSION,
    )


async def init_db():
    # rozpoznaje backend i tworzy tabelę
    backend = database.url.scheme

    if backend.startswith("sqlite"):
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                payload TEXT,
                prediction TEXT,
                model_version TEXT
            )
        """
    else:
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP,
                payload JSONB,
                prediction TEXT,
                model_version TEXT
            )
        """

    await database.execute(query=query)


async def save_prediction(
    payload: dict,
    prediction: float | str,
    model_version: str,
):
    # zapis pojedynczej predykcji
    backend = database.url.scheme
    is_sqlite = backend.startswith("sqlite")

    ts_value = dt.datetime.utcnow().isoformat() if is_sqlite else dt.datetime.utcnow()
    payload_value = json.dumps(payload) if is_sqlite else payload

    query = """
        INSERT INTO predictions(ts, payload, prediction, model_version)
        VALUES (:ts, :payload, :pred, :ver)
    """

    await database.execute(
        query=query,
        values={
            "ts": ts_value,
            "payload": payload_value,
            "pred": str(prediction),
            "ver": model_version,
        },
    )
