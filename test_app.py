import pytest
from fastapi.testclient import TestClient
from app import app

# Using TestClient as a context manager triggers the lifespan startup,
# which loads the real model and scaler from disk before any test runs.
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# A valid trip used across multiple tests
VALID_TRIP = {
    "pickup_hour": 14,
    "pickup_day_of_week": 2,
    "is_weekend": 0,
    "trip_distance": 3.5,
    "trip_duration_minutes": 18.0,
    "trip_speed_mph": 11.7,
    "log_trip_distance": 1.45,
    "fare_amount": 16.5,
    "fare_per_mile": 4.71,
    "fare_per_minute": 0.92,
    "passenger_count": 1,
    "pickup_borough_encoded": 3,
    "dropoff_borough_encoded": 3,
    "tolls_amount": 0.0,
    "extra": 0.5,
    "mta_tax": 0.5,
    "congestion_surcharge": 2.5,
    "Airport_fee": 0.0,
}


# Test 1 - Valid single prediction returns expected fields and a numeric tip
def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["tip_amount"], float)
    assert len(data["prediction_id"]) == 36  # UUID length


# Test 2 - Batch prediction returns one result per trip
def test_predict_batch_valid(client):
    payload = {"trips": [VALID_TRIP, VALID_TRIP, VALID_TRIP]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        assert "tip_amount" in pred
        assert "prediction_id" in pred


# Test 3 - pickup_hour outside 0-23 is rejected
def test_predict_invalid_pickup_hour(client):
    bad_trip = {**VALID_TRIP, "pickup_hour": 25}
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


# Test 4 - Negative trip distance is rejected
def test_predict_invalid_negative_distance(client):
    bad_trip = {**VALID_TRIP, "trip_distance": -1.0}
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


# Test 5 - Request missing a required field is rejected
def test_predict_missing_field(client):
    incomplete = {k: v for k, v in VALID_TRIP.items() if k != "fare_amount"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


# Test 6 - Health check confirms model is loaded
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


# Test 7 - Model info returns expected metadata fields
def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_names" in data
    assert "training_metrics" in data
    assert len(data["feature_names"]) == 18


# Test 8 - Batch endpoint rejects more than 100 records
def test_batch_max_100_enforced(client):
    payload = {"trips": [VALID_TRIP] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# Test 9 - Zero distance is rejected (field requires gt=0)
def test_predict_zero_distance(client):
    bad_trip = {**VALID_TRIP, "trip_distance": 0.0}
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422


# Test 10 - Extreme fare is accepted; model still returns a tip prediction
def test_predict_extreme_fare(client):
    extreme_trip = {**VALID_TRIP, "fare_amount": 999.99}
    response = client.post("/predict", json=extreme_trip)
    assert response.status_code == 200
    assert "tip_amount" in response.json()