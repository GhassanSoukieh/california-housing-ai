from unittest.mock import patch

import requests

from llm import ExplanationError


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


def test_predict_endpoint_returns_prediction(client):
    with patch("main.run_prediction", return_value={
        "price_usd": 275000.0,
        "model": "test-model",
    }) as predict_mock:
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json() == {"price_usd": 275000.0, "model": "test-model"}
    predict_mock.assert_called_once()


def test_explanation_endpoint_returns_prediction_and_text(client):
    with (
        patch("main.run_prediction", return_value={
            "price_usd": 275000.0,
            "model": "test-model",
        }),
        patch("main.explain_prediction", return_value="Income and location affect the estimate."),
    ):
        response = client.post("/prediction_explanation", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json() == {
        "price_usd": 275000.0,
        "model": "test-model",
        "explanation": "Income and location affect the estimate.",
    }


def test_endpoints_reject_invalid_data(client):
    invalid_input = {**VALID_INPUT, "MedInc": "not-a-number"}

    predict_response = client.post("/predict", json=invalid_input)
    explanation_response = client.post("/prediction_explanation", json=invalid_input)

    assert predict_response.status_code == 422
    assert explanation_response.status_code == 422
    assert "detail" in predict_response.json()


def test_predict_endpoint_reports_model_failure(client):
    with patch("main.run_prediction", side_effect=RuntimeError("broken model")):
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 500
    assert response.json()["detail"] == "The prediction model failed to process the request."


def test_explanation_endpoint_reports_offline_ollama(client):
    with (
        patch("main.run_prediction", return_value={"price_usd": 275000.0, "model": "test-model"}),
        patch("main.explain_prediction", side_effect=ExplanationError(
            "The explanation service is unavailable."
        )),
    ):
        response = client.post("/prediction_explanation", json=VALID_INPUT)

    assert response.status_code == 503
    assert "explanation service is unavailable" in response.json()["detail"].lower()


def test_explanation_endpoint_reports_ollama_model_failure(client):
    from llm import ExplanationError

    with (
        patch("main.run_prediction", return_value={"price_usd": 275000.0, "model": "test-model"}),
        patch("main.explain_prediction", side_effect=ExplanationError(
            "The explanation service rejected the request. Verify that the llama3.2 model is installed in Ollama."
        )),
    ):
        response = client.post("/prediction_explanation", json=VALID_INPUT)

    assert response.status_code == 503
    assert "model is installed" in response.json()["detail"]


def test_predict_endpoint_rejects_invalid_model_result(client):
    with patch("main.run_prediction", return_value={"price_usd": float("nan"), "model": "test-model"}):
        response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 500
    assert "invalid result" in response.json()["detail"]


def test_llm_request_failure_has_meaningful_error():
    from llm import explain_prediction

    with patch("llm.requests.post", side_effect=requests.ConnectionError("offline")):
        try:
            explain_prediction(VALID_INPUT, {"price_usd": 275000.0})
        except ExplanationError as exc:
            assert "Ollama" in str(exc)
        else:
            raise AssertionError("Expected ExplanationError")
