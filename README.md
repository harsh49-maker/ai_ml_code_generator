# AI ML Code Generator

AI ML Code Generator is a Streamlit-based application that helps users generate machine learning Python code from an uploaded dataset and a simple description of the ML task.

The application uses an AI agent to inspect the dataset, understand its structure, and generate a suitable machine learning pipeline using Python and scikit-learn.

## Features

- Upload CSV or Excel datasets
- Preview uploaded datasets
- Display dataset size and missing values
- Automatically inspect dataset columns and data types
- Identify numerical and categorical columns
- Detect missing values and duplicate rows
- Generate machine learning code based on the user's requirement
- Supports regression and classification workflows
- Automatic preprocessing and categorical encoding
- Train-test splitting
- Model training and evaluation
- Generate useful evaluation metrics
- Generate visualizations
- Download the generated Python code
- Simple and interactive Streamlit interface

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- smolagents
- LiteLLM
- Gemini API

## Project Structure

```text
ai-ml-code-generator/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── generated/