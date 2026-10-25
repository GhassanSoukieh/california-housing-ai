import math

import pytest

from predictor import predict


class StubModel:
    def predict(self, housing_data):
        return {
            "price_usd": 250000.0 + housing_data["MedInc"] * 10000,
            "model": "stub-model",
        }


VALID_INPUT = {
    "MedInc": 4.445,
    "HouseAge": 52.0,
    "AveRooms": 5.5346,
    "AveBedrms": 1.1509,
    "Population": 742.0,
    "AveOccup": 2.3333,
    "Latitude": 37.77,
    "Longitude": -122.43,
}


def test_predict_returns_reasonable_numeric_result():
    result = predict(StubModel(), VALID_INPUT)

    assert isinstance(result["price_usd"], (int, float))
    assert math.isfinite(result["price_usd"])
    assert 0 < result["price_usd"] < 1_000_000
    assert result["model"] == "stub-model"


def test_predict_passes_all_features_to_model():
    class RecordingModel:
        def predict(self, housing_data):
            assert housing_data == VALID_INPUT
            return {"price_usd": 100.0, "model": "recording-model"}

    assert predict(RecordingModel(), VALID_INPUT)["price_usd"] == 100.0


def test_predict_propagates_model_failure():
    class FailingModel:
        def predict(self, housing_data):
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        predict(FailingModel(), VALID_INPUT)
