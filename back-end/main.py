import logging
import math
from contextlib import asynccontextmanager
from numbers import Real

from fastapi import FastAPI, HTTPException, Request
from data import HousingInput
from llm import ExplanationError, explain_prediction
from predictor import load_model, predict as run_prediction

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading housing model")
    try:
        app.state.model = load_model()
    except Exception as exc:
        logger.exception("Housing model failed to load")
        raise RuntimeError("The prediction model could not be loaded.") from exc
    logger.info("Housing model loaded")
    yield
    app.state.model = None
    logger.info("Application shutting down")


app = FastAPI(lifespan=lifespan)


def _predict(data: HousingInput, request: Request) -> dict:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        )

    try:
        prediction = run_prediction(model, data.model_dump())
    except Exception as exc:
        logger.exception("Prediction model failed")
        raise HTTPException(
            status_code=500,
            detail="The prediction model failed to process the request.",
        ) from exc

    price = prediction.get("price_usd") if isinstance(prediction, dict) else None
    model_name = prediction.get("model") if isinstance(prediction, dict) else None
    if (
        not isinstance(price, Real)
        or isinstance(price, bool)
        or not math.isfinite(float(price))
        or not isinstance(model_name, str)
        or not model_name.strip()
    ):
        logger.error("Prediction model returned an invalid result: %r", prediction)
        raise HTTPException(
            status_code=500,
            detail="The prediction model returned an invalid result.",
        )

    return prediction


@app.post("/predict")
def predict(data: HousingInput, request: Request):
    return _predict(data, request)


@app.post("/prediction_explanation")
def prediction_explanation(data: HousingInput, request: Request):
    housing_data = data.model_dump()
    prediction = _predict(data, request)
    try:
        explanation = explain_prediction(housing_data, prediction)
    except ExplanationError as exc:
        logger.warning("Explanation service failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "price_usd": prediction["price_usd"],
        "model": prediction["model"],
        "explanation": explanation
    }
