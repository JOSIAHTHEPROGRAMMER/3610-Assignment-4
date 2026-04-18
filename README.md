# COMP 3610 - Assignment 4: MLOps & Model Deployment

Deploys a Random Forest model (trained on NYC Yellow Taxi data) as a containerized REST API using FastAPI, MLflow, and Docker.

---

## Prerequisites

- Python 3.11
- Docker Desktop (or Docker Engine on Linux)
- Git

---

## Project Structure

```
assignment4/
├── assignment4.ipynb    # Main notebook with experiments and demos
├── app.py               # FastAPI application
├── test_app.py          # pytest test suite
├── Dockerfile           # Container definition
├── docker-compose.yml   # Service orchestration (API + MLflow)
├── requirements.txt     # Pinned Python dependencies
├── README.md
├── .gitignore
├── .dockerignore
└── models/              # Saved model and scaler (gitignored)
```

---

## Setup & Running Locally

**1. Clone the repository**

```bash
git clone https://github.com/JOSIAHTHEPROGRAMMER/3610-Assignment-4
cd assignment4
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Train the model and generate artifacts**

Open and run `assignment4.ipynb` from top to bottom. This will:

- Download the NYC Taxi dataset
- Train and log models to MLflow
- Save `models/rf_baseline.pkl` and `models/scaler.pkl`

**5. Run the API locally**

```bash
uvicorn app:app --reload
```

API will be available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

**6. Run tests**

```bash
pytest test_app.py -v
```

---

## Running with Docker Compose

**1. Build and start all services**

```bash
docker compose up --build
```

This starts:

- `api` - prediction service at `http://localhost:8000`
- `mlflow` - tracking server at `http://localhost:5000`

**2. Make a prediction**

```bash
# Health check
curl.exe http://localhost:8000/health

# Batch prediction (PowerShell)
$body = '{"trips":[{"pickup_hour":9,...}]}'
Invoke-WebRequest -Uri http://localhost:8000/predict/batch -Method POST -ContentType "application/json" -Body $body -UseBasicParsing
```

**3. Shut down**

```bash
docker compose down
```

---

## Environment Variables

| Variable              | Default                  | Description                           |
| --------------------- | ------------------------ | ------------------------------------- |
| `MODEL_PATH`          | `models/rf_baseline.pkl` | Path to saved model inside container  |
| `SCALER_PATH`         | `models/scaler.pkl`      | Path to saved scaler inside container |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000`     | MLflow server address                 |
