from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
try:
    from .pipeline import predict
except ImportError:
    from pipeline import predict

app = FastAPI(title="Predictive Maintenance Risk API", version="1.0.0")
MODEL_PATH = Path("artifacts/model.joblib")

class SensorReading(BaseModel):
    Type: str = Field(pattern="^[LMH]$")
    air_temperature_k: float = Field(alias="Air temperature [K]", ge=250, le=350)
    process_temperature_k: float = Field(alias="Process temperature [K]", ge=250, le=400)
    rotational_speed_rpm: float = Field(alias="Rotational speed [rpm]", ge=0, le=10000)
    torque_nm: float = Field(alias="Torque [Nm]", ge=0, le=500)
    tool_wear_min: float = Field(alias="Tool wear [min]", ge=0, le=1000)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.post("/predict")
def score(reading: SensorReading):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Train the model first: python src/pipeline.py")
    model = joblib.load(MODEL_PATH)
    raw = reading.model_dump(by_alias=True)
    return predict(model, raw)
