import math
import sys
from pathlib import Path

from test_predictor import VALID_INPUT




sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictor import load_model, predict  # pyright: ignore[reportMissingImports]


def test_real_huggingface_model_prediction():
    model = load_model()

    result = predict(model, VALID_INPUT)

    assert isinstance(result, dict)
    assert isinstance(result["price_usd"], (int, float))
    assert math.isfinite(result["price_usd"])
    assert result["price_usd"] > 0
    assert result["model"] == "xgboost"
    

## test if the model loads once. 
def test_model_loads_only_once(monkeypatch):
    import main
    from fastapi.testclient import TestClient
    from unittest.mock import Mock

    # Create a fake model
    fake_model = object()

    # Create a mock loader that returns the fake model
    mock_loader = Mock(return_value=fake_model)

    # Replace the real load_model function
    monkeypatch.setattr(main, "load_model", mock_loader)

    # Replace prediction with a fake function
    mock_predict = Mock(
        return_value={
            "price_usd": 275000.0,
            "model": "test-model"
        }
    )

    monkeypatch.setattr(main, "run_prediction", mock_predict)

    # Start FastAPI
    with TestClient(main.app) as client:

        # Send three prediction requests
        for _ in range(3):
            response = client.post("/predict", json=VALID_INPUT)
            assert response.status_code == 200

        # Verify model was loaded only once
        mock_loader.assert_called_once_with()

        # Verify three predictions were made
        assert mock_predict.call_count == 3

        # Verify every prediction used the same model
        for call in mock_predict.call_args_list:
            assert call.args[0] is fake_model
        
