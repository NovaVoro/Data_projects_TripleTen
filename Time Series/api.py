# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_core import train_all_models, predict_next_hour

CSV_PATH = "taxi.csv"

app = FastAPI(title="Taxi Demand Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train once at startup
@app.on_event("startup")
def load_model():
    global MODEL_STATE
    MODEL_STATE = train_all_models(CSV_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    state = MODEL_STATE
    return {
        "best_model_name": state["best_model_name"],
        "best_test_rmse": state["best_test_rmse"],
        "all_results": {
            k: {
                "train_rmse": v["train_rmse"],
                "test_rmse": v["test_rmse"],
            }
            for k, v in state["all_results"].items()
        },
    }


@app.get("/predict_next")
def predict_next():
    state = MODEL_STATE
    pred = predict_next_hour(
        state["best_model"],
        state["df_features"],
        state["df_hourly"],
    )
    return pred