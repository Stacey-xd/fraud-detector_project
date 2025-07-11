# Credit Card Fraud Detection API

This project implements a production-ready system for detecting fraudulent credit card transactions using machine learning. The model is served through a RESTful API built with FastAPI.

## Features

- Performed EDA and preprocessing
- Handled class imbalance (fraud is rare) via SMOTE and class weights
- Trained multiple ML models: Logistic Regression, RandomForest, XGBoost, CatBoost, LightGBM
- Created FastAPI-based API for predictions
- Included unit tests with `pytest` and `httpx`
- Modular structure suitable for scaling

---

## Tech Stack

| Category              | Libraries / Tools                                                |
|----------------------|------------------------------------------------------------------|
| **Language**         | Python 3.10.17                                                   |
| **Visualization**    | `matplotlib`, `seaborn` *(used during EDA)*                      |
| **ML Models**        | `logistic regression`, `XGBoost`, `LightGBM`, `CatBoost`                |
| **Imbalance Handling** | `imbalanced-learn` (`SMOTE`, `BorderlineSMOTE`), class weights |
| **API Backend**      | `FastAPI`, `Uvicorn`                                             |
| **Testing**          | `pytest`, `httpx` *(FastAPI endpoints)*                          |

---

## Example Request

```bash
POST /predict
Content-Type: application/json
```

```json
{
  "V1": -1.35,
  "V2": -0.07,
  "V3": 2.53,
  "V4": 1.37,
  "V28": -0.02,
  "Amount": 0.24,
  "Time": -1.99
  ...
}
```

```json
{
  "fraud_probability": 0.08
}
```