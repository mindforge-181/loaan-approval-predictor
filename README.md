# Loan Approval Predictor

An AI/ML-based application that predicts loan approval outcomes using supervised machine learning. The project processes applicant data and uses a trained classification model to generate predictions.

## Overview

The Loan Approval Predictor aims to simplify the initial loan assessment process by analyzing applicant information and predicting whether a loan application is likely to be approved or rejected.

## Features

- **Machine Learning Prediction:** Uses a trained supervised learning classification model.
- **Data Preprocessing:** Prepares applicant data for model inference.
- **Model Integration:** Loads a trained machine learning model for predictions.
- **Feature Scaling:** Uses a saved scaler where required by the model pipeline.
- **REST API:** Provides a backend interface for submitting applicant details and retrieving predictions.
- **Model Evaluation:** Supports evaluating model performance using appropriate classification metrics.

## Tech Stack

- **Programming Language:** Python
- **Backend Framework:** FastAPI
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **API Testing:** Swagger UI / Postman
- **Model Storage:** Pickle (`.pkl`)

## My Role — Backend Developer

As a Backend Developer, I contributed to the development of the backend API and the integration of the supervised machine learning model.

- Developed API endpoints using FastAPI.
- Integrated the trained machine learning model into the backend.
- Connected input data processing with the prediction workflow.
- Worked on request validation and prediction responses.
- Supported API testing and integration with the application workflow.

## Project Workflow

1. Collect and inspect the loan applicant dataset.
2. Clean and preprocess the data.
3. Train a supervised machine learning classification model.
4. Evaluate the model using suitable performance metrics.
5. Save the trained model and required preprocessing components.
6. Integrate the model with the FastAPI backend.
7. Submit applicant information through the API to obtain a prediction.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/nethanreddyp/loaan-approval-predictor.git
cd loaan-approval-predictor
```

Install the required dependencies if `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

### Run the Application

If `main.py` contains the FastAPI application instance named `app`, run:

```bash
python -m uvicorn main:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

**Note:** Adjust the startup command to match the actual application entry point and installed dependencies.

## Future Enhancements

- Add a web-based user interface.
- Improve model evaluation and validation.
- Add secure authentication and application logging.
- Deploy the API to a cloud platform.

## Disclaimer

This project is intended for educational and demonstration purposes. Predictions are not a substitute for a financial institution's actual loan assessment process.

## Author

**Nethan Reddy**

GitHub: [@nethanreddyp](https://github.com/nethanreddyp)

## Acknowledgements

Developed as part of an AI/ML training project, with collaborative contributions across data preprocessing, model development, backend integration, and application testing.
