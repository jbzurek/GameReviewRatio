import os
import json
import datetime as dt
from typing import Dict, Any

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from databases import Database

# --- Konfiguracja ---
MODEL_PATH = "data/06_models/ag_production.pkl"
REQUIRED_COLUMNS_PATH = "data/06_models/required_columns.json"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/asi_baza")
MODEL_VERSION = "local-dev"
database = Database(DATABASE_URL)

model = None
required_columns: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, database, required_columns
   
    if model is None:
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f"Nie znaleziono pliku modelu w '{MODEL_PATH}'")
        model = joblib.load(MODEL_PATH)

    if os.path.isfile(REQUIRED_COLUMNS_PATH):
        try:
            with open(REQUIRED_COLUMNS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # wszystkie maja typ int64
                required_columns = data["columns"] if isinstance(data, dict) and "columns" in data else list(data)
        except Exception:
            required_columns = []
    else:
        required_columns = []

    # Połącz DB; fallback do SQLite jeśli Postgres nie działa
    try:
        await database.connect()
    except Exception:
        # SQLite jako fallback
        database = Database("sqlite+aiosqlite:///./predictions.db")
        await database.connect()

    await init_db()
    yield
    await database.disconnect()

app = FastAPI(lifespan=lifespan)

class Features(BaseModel):
    data: Dict[str, Any]

class Prediction(BaseModel):
    prediction: float | str
    model_version: str

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
async def predict(payload: Features):
    # Sprawdź czy model jest gotowy
    if model is None:
        raise HTTPException(status_code=500, detail="Model nie został załadowany")

    X = pd.DataFrame([payload.data])


    if required_columns:
        target_cols = list(required_columns)
        X = X.reindex(columns=target_cols, fill_value=0)


    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencja nie powiodła się: {e}")

    try:
        pred_out = float(pred)
    except Exception:
        pred_out = str(pred)

    # Zapis do bazy (payload: TEXT dla SQLite, JSONB dla Postgresa)
    await save_prediction(payload.data, pred_out, MODEL_VERSION)

    return {"prediction": pred_out, "model_version": MODEL_VERSION}

async def init_db():
    # Rozpoznaj backend
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

async def save_prediction(payload: dict, prediction: float | str, model_version: str):
    backend = database.url.scheme
    is_sqlite = backend.startswith("sqlite")

    ts_value = dt.datetime.utcnow() if not is_sqlite else dt.datetime.utcnow().isoformat()
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