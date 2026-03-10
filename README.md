# Fraud Detection SaaS - Setup Instructions

## Prerequisites
- Python 3.8+
- pip

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd app/models
python train.py
```

This will:
- Load data from `data/Fraud.csv`
- Train an XGBoost classifier
- Save artifacts to `app/models/model/`:
  - `fraud_model.pkl`
  - `scaler.pkl`
  - `features.pkl`

### 3. Initialize Database
```bash
python init_db.py
```

### 4. Seed Database (Optional)
```bash
python seed_db.py
```

### 5. Run the API
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

### 6. Run the Dashboard
```bash
streamlit run dashboard.py
```

Dashboard will be available at: `http://localhost:8501`

## API Usage

### Create Organization
```bash
curl -X POST http://localhost:8000/admin/org \
  -H "Content-Type: application/json" \
  -d '{"name": "Acme Corp"}'
```

### Create API Key
```bash
curl -X POST http://localhost:8000/admin/apikey \
  -H "Content-Type: application/json" \
  -d '{"org_id": 1, "threshold": 0.5}'
```

### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "step": 1,
    "amount": 9839.64,
    "oldbalanceOrg": 170136.0,
    "newbalanceOrig": 160296.36,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "transaction_type": "PAYMENT",
    "transaction_hour": 14,
    "transaction_day": 1,
    "merchant_category": 5411,
    "user_transaction_count_24h": 3,
    "user_avg_transaction_amount": 5000.0,
    "is_international": 0,
    "distance_from_home": 0.0,
    "device_change": 0,
    "failed_login_attempts": 0
  }'
```

## Project Structure
```
frauddetectionproject/
├── app/
│   ├── api/          # FastAPI application
│   ├── core/         # Core utilities
│   ├── features/     # Feature engineering
│   ├── models/       # ML models and training
│   ├── db.py         # Database setup
│   └── models.py     # SQLAlchemy models
├── data/             # Dataset
├── dashboard.py      # Streamlit dashboard
├── requirements.txt  # Dependencies
└── README.md         # This file
```
