# 🌸 Iris Species Classifier

[![CI/CD Pipeline](https://github.com/juma-paul/iris-classifier-app/actions/workflows/ci.yml/badge.svg)](https://github.com/juma-paul/flower-species-classifier/actions)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-deployed-orange.svg)](https://aws.amazon.com/)

> A production-grade machine learning system for iris flower species classification with complete CI/CD pipeline, monitoring, and deployment on AWS.

[Live Demo](https://irisclassifierapp.streamlit.app/) • [API Documentation](http://18.190.86.245/docs)

## 📋 Overview

This project demonstrates a complete end-to-end machine learning system, from data processing to production deployment. The system classifies iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.

### Key Features

- ✅ **Production-Ready ML Pipeline** - Automated data processing, training, and evaluation
- ✅ **RESTful API** - FastAPI with single and batch prediction endpoints
- ✅ **Comprehensive Testing** - 20+ unit and integration tests
- ✅ **CI/CD Pipeline** - Automated testing and deployment with GitHub Actions
- ✅ **Containerized Deployment** - Docker with AWS EC2 hosting
- ✅ **Real-time Monitoring** - Prometheus + Grafana dashboards
- ✅ **Interactive UI** - Streamlit web application
- ✅ **Experiment Tracking** - MLflow for model versioning

### Model Performance

- **Accuracy:** 93.3%
- **Precision:** 94.2% (macro avg)
- **Recall:** 94.2% (macro avg)
- **F1-Score:** 94.2% (macro avg)

## 🏗️ Architecture

<img src="images/architecture.png" alt="Architecture Diagram" width="700" />

*System architecture showing the complete ML pipeline from data to deployment*

### Technology Stack

**Machine Learning & Data Processing**
- Python 3.13, scikit-learn, pandas, numpy
- MLflow for experiment tracking

**API & Web**
- FastAPI (REST API)
- Streamlit (User Interface)
- Pydantic (data validation)

**DevOps & Infrastructure**
- Docker (containerization)
- GitHub Actions (CI/CD)
- AWS EC2 (deployment)
- Elastic IP (static endpoint)

**Monitoring**
- Prometheus (metrics collection)
- Grafana (visualization dashboards)

## 📸 Screenshots

### Streamlit User Interface

<img src="images/streamlit-ui.png" alt="Streamlit UI" width="700" />

*Interactive web interface for single and batch predictions*

### MLflow Experiment Tracking

<img src="images/mlflow-dashboard.png" alt="MLflow Dashboard" width="700" />

*Model performance comparison and experiment tracking*

### Grafana Monitoring Dashboard

<img src="images/grafana-dashboard.png" alt="Grafana Dashboard" width="700" />

*Real-time API metrics: predictions per Endpoint*

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- Docker
- AWS account (for deployment)
- GitHub account (for CI/CD)

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/juma-paul/iris-classifier-app.git 
cd iris-classifier-app
```

2. **Install uv (package manager)**
```bash 
pip install uv
```

3. **Install dependencies**
```bash 
uv sync
```

4. **Prepare data and train model**
```bash
uv run python src/data/prepare_data.py`
uv run python src/models/train.py
```

5. **Run the API**
```bash
uv run uvicorn src.api.app:app --reload
```

API available at `http://localhost:8000/docs`

6. **Run Streamlit UI** (in a new terminal)
```bash
streamlit run app_streamlit.py
```

### Running Tests
```bash
pytest tests/ -v
```

## 🐳 Docker Deployment

### Build and Run Locally
Build the image
```bash
docker build -t iris-classifier .
```

Run the container
```bash
docker run -d -p 8000:8000 iris-classifier
```

### Deploy to Docker Hub
Build for multiple platforms
```bash
docker buildx build --platform linux/amd64,linux/arm64
-t YOUR_DOCKERHUB_USERNAME/iris-classifier:latest --push .
```

## ☁️ AWS EC2 Deployment

The application is deployed on AWS EC2 with the following setup:

- **Instance Type:** t2.micro (free tier)
- **OS:** Ubuntu 24.04 LTS
- **Elastic IP:** Static IP address for consistent access
- **Docker Containers:** API, Prometheus, Grafana
- **Ports:** 80 (API), 9090 (Prometheus), 3000 (Grafana)

### Monitoring Stack

Access monitoring dashboards:
- **Prometheus:** `http://YOUR_ELASTIC_IP:9090`
- **Grafana:** `http://YOUR_ELASTIC_IP:3000` (default login: admin/admin)
Replace YOUR_DOCKERHUB_USERNAME and YOUR_ELASTIC_IP with your actual values.

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for automated testing and deployment.

### Workflow Triggers
- Push to `main` branch
- Pull requests to `main`

### Pipeline Steps
1. **Test** - Run all unit and integration tests
2. **Build** - Create Docker image for linux/amd64
3. **Push** - Upload image to Docker Hub
4. **Deploy** - SSH to EC2 and update running container

View workflow: `.github/workflows/ci.yml`

## 📡 API Usage

### Single Prediction

```bash
curl -X POST "http://YOUR_ELASTIC_IP/predict"
-H "Content-Type: application/json"
-d '{ "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 }'
```

**Response:**
{"prediction": 0}

### Batch Prediction
```bash
curl -X POST "http://YOUR_ELASTIC_IP/predict-batch"
-H "Content-Type: application/json"
-d '{ "iris_measurements": [ {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}, {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3} ] }'
```
**Response:**
[0, 2]

Species mapping: `0 = Setosa, 1 = Versicolor, 2 = Virginica`


## 🧪 Testing

The project includes comprehensive testing:

- **Data Processing Tests** (8 tests) - Validation, cleaning, splitting
- **Model Tests** (5 tests) - Loading, predictions, accuracy
- **API Tests** (7 tests) - Endpoints, validation, error handling

Total: **20 tests** ensuring code quality and reliability

## 📊 Monitoring Metrics

Tracked metrics include:
- Total predictions by IP address
- Predictions per endpoint (/predict vs /predict-batch)
- Average response time by endpoint
- Unique users (IP addresses)

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@juma-paul](https://github.com/juma-paul)
- LinkedIn: [Juma R. Paul ](https://www.linkedin.com/in/juma-paul/)
- Portfolio: [Juma Paul](https://your-website.com)

## 🙏 Acknowledgments

- **Dataset**: scikit-learn's built-in Iris dataset (originally from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris))
- Built as part of a production ML systems learning project

---

⭐ If you found this project helpful, please consider giving it a star!