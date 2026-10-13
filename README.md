# FraudLens — Machine Learning Fraud Detection Platform

**An end-to-end fraud detection application powered by XGBoost, FastAPI, React, Docker, and AWS.**

FraudLens is a full-stack machine learning application that identifies potentially fraudulent credit card transactions using a trained classification model. It combines predictive analytics, a REST API, and an interactive dashboard to demonstrate how machine learning can be integrated into a real-world software system.

The project covers the complete ML development workflow, from data exploration and model evaluation to API development, containerization, and cloud deployment.

## Key Features

- **Real-time fraud prediction:** Classifies transactions and returns fraud probability scores.
- **Machine learning pipeline:** Compares Logistic Regression, Random Forest, and XGBoost models.
- **Interactive dashboard:** Built with React and Recharts to visualize prediction results.
- **REST API:** FastAPI endpoints for predictions, transaction history, and application health.
- **Prediction history:** Stores transaction predictions in a SQLite database.
- **Input validation:** Uses Pydantic to validate incoming transaction data.
- **Containerized backend:** Docker packages the application for consistent execution.
- **AWS deployment:** Backend deployed using Amazon ECS Fargate and Amazon ECR.
- **Automated testing:** Includes API tests using pytest.

## Technology Stack

| Category | Technologies |
|---|---|
| Machine Learning | Python, XGBoost, scikit-learn, pandas |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | React, JavaScript, Vite, Recharts |
| Database | SQLite |
| Cloud | AWS ECS, AWS Fargate, Amazon ECR |
| DevOps | Docker, Git, GitHub |
| Testing | pytest |

## Machine Learning Approach

FraudLens uses a credit card transaction dataset containing **284,807 transactions**, including **492 fraudulent transactions**.

Because fraudulent transactions represent only approximately **0.17%** of the dataset, class imbalance is an important challenge.

Three classification approaches were explored:

1. Logistic Regression
2. Random Forest
3. XGBoost

XGBoost was selected as the final model based on evaluation results.

### XGBoost Model Performance

| Metric | Result |
|---|---|
| Precision | ~90.9% |
| Recall | ~81.6% |
| F1-score | ~0.860 |
| Classification threshold | 0.20 |

The classification threshold was adjusted to balance fraud detection recall and precision.

**Why these metrics matter:** Accuracy alone can be misleading for heavily imbalanced fraud datasets. Precision, recall, and F1-score provide a more useful view of fraud detection performance.

These results represent offline evaluation, not guaranteed production performance.

## System Architecture

```text
                  User
                   |
                   v
           React Dashboard
                   |
              HTTP Requests
                   |
                   v
              FastAPI API
                   |
          +--------+--------+
          |                 |
          v                 v
     XGBoost Model     SQLite Database
          |                 |
          v                 v
   Fraud Probability   Prediction History
          |
          v
    Prediction Result
```

The backend runs inside a Docker container deployed to AWS ECS Fargate. The Docker image is stored in Amazon ECR.

The React frontend communicates with the backend through REST API requests.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend and model health check |
| POST | `/predict` | Generate a fraud prediction |
| GET | `/predictions` | Retrieve prediction history |

### Example Prediction Request

```json
{
  "Time": 0,
  "V1": -1.36,
  "V2": -0.07,
  "V3": 2.54,
  "V4": 1.38,
  "V5": -0.34,
  "V6": 0.46,
  "V7": 0.24,
  "V8": 0.10,
  "V9": 0.36,
  "V10": 0.09,
  "V11": -0.55,
  "V12": -0.62,
  "V13": -0.99,
  "V14": -0.31,
  "V15": 1.47,
  "V16": -0.47,
  "V17": 0.21,
  "V18": 0.03,
  "V19": 0.40,
  "V20": 0.25,
  "V21": -0.02,
  "V22": 0.28,
  "V23": -0.11,
  "V24": 0.07,
  "V25": 0.10,
  "V26": -0.19,
  "V27": 0.13,
  "V28": -0.02,
  "Amount": 149.62
}
```

The API validates transaction features, applies the trained model, and returns a classification and fraud probability.

## Running FraudLens Locally

### Prerequisites

- Python 3.13
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/username36497/FraudLens.git
cd FraudLens
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Backend address:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

### 4. Start the React frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local address displayed by Vite, typically:

`http://localhost:5173`

### 5. Run API tests

From the project root:

```bash
python -m pytest tests/ -v
```

## Cloud Deployment

The FraudLens backend is containerized with Docker and deployed using AWS.

**Deployment workflow:**

1. Build the Docker image for Linux ARM64.
2. Push the image to Amazon Elastic Container Registry.
3. Deploy the container to Amazon ECS using AWS Fargate.
4. Configure networking and container access.
5. Verify the running application through the `/health` endpoint.

The deployed backend has successfully passed its health check with the XGBoost model loaded.

The public frontend deployment and HTTPS integration are being finalized.

**Current limitation:** Prediction history uses local SQLite storage inside the ECS container. This storage is ephemeral and may be lost when the task is replaced.

## Project Structure

```text
FraudLens/
├── backend/
│   ├── main.py
│   └── database.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── xgboost.pkl
│   ├── xgboost_threshold.pkl
│   ├── random_forest.pkl
│   └── rf_threshold.pkl
├── tests/
│   └── test_api.py
├── explore_data.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Engineering Challenges

### Handling Imbalanced Data

Fraudulent transactions make up a very small proportion of the dataset. Model evaluation therefore focuses on precision, recall, and F1-score rather than accuracy alone.

### Building a Production-Style API

The model is integrated with FastAPI and Pydantic to provide structured requests, input validation, and reusable prediction endpoints.

### Containerization and Cloud Compatibility

Deploying the backend required configuring Docker images for the correct CPU architecture, handling cloud environment differences, and troubleshooting database initialization.

### Connecting Machine Learning with a User Interface

The React dashboard provides a practical interface for submitting transactions, displaying fraud predictions, and reviewing prediction history.

## Future Improvements

- Deploy the React dashboard with HTTPS.
- Configure a stable HTTPS API endpoint.
- Replace ephemeral SQLite storage with persistent cloud storage.
- Add CI/CD workflows for automated testing and deployment.
- Introduce monitoring, logging, and model performance tracking.
- Add model explainability to help interpret fraud predictions.

## Author

**Gael Ganza Gatsimbanyi**

Computer Science & Mathematics Student — University of Alberta

[GitHub Profile](https://github.com/username36497)

---

*FraudLens is an educational and portfolio project demonstrating end-to-end machine learning engineering and cloud deployment.*
