from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Welcome to FraudLens API"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "XGBoost"
    assert data["model_loaded"] is True


def test_missing_transaction_features():
    response = client.post(
        "/predict",
        json={
            "Time": 100,
            "Amount": 250
        }
    )

    assert response.status_code == 422

def test_negative_amount():
    transaction = {
        "Time": 100,
        "Amount": -500  
    }
    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    response = client.post("/predict", json=transaction)
    assert response.status_code == 422


def test_valid_prediction():
    transaction = {
        "Time": 100,
        "Amount": 250
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "prediction" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert data["prediction"] in ["Fraud", "Legitimate"]


def test_nan_value():
    transaction = {
        "Time": 100,
        "Amount": 250
    }


    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    transaction["V1"] = "NaN"

    response = client.post("/predict", json=transaction)

    assert response.status_code == 422



from unittest.mock import patch


def test_fraud_threshold():
    transaction = {
        "Time": 100,
        "Amount": 250
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    with patch("backend.main.model.predict_proba") as mock_predict:
        mock_predict.return_value = [[0.75, 0.25]]

        response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert data["fraud_probability"] == 0.25
    assert data["is_fraud"] is True
    assert data["prediction"] == "Fraud"

def test_legitimate_threshold():
    transaction = {
        "Time": 100,
        "Amount": 250
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    with patch("backend.main.model.predict_proba") as mock_predict:
        mock_predict.return_value = [[0.90, 0.10]]

        response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert data["fraud_probability"] == 0.10
    assert data["is_fraud"] is False
    assert data["prediction"] == "Legitimate"

def test_prediction_error():
    transaction = {
        "Time": 100,
        "Amount": 250
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = 0.0

    with patch("backend.main.model.predict_proba") as mock_predict:
        mock_predict.side_effect = RuntimeError("Model failed")

        response = client.post("/predict", json=transaction)

    assert response.status_code == 500

    assert response.json() == {
        "detail": "An error occurred while processing the prediction."
    }

def test_prediction_history():
    response = client.get("/predictions")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)

    for prediction in data:
        assert "id" in prediction
        assert "amount" in prediction
        assert "prediction" in prediction
        assert "fraud_probability" in prediction


from fastapi.testclient import TestClient
from unittest.mock import patch

def test_prediction_history_database_error():
    error_client = TestClient(app, raise_server_exceptions=False)

    with patch(
        "backend.main.get_predictions",
        side_effect=Exception("Database connection failed")
    ):
        response = error_client.get("/predictions")

    assert response.status_code == 500