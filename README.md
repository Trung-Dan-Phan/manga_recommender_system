# Manga Recommender System

A comprehensive system designed to provide personalized manga recommendations using advanced machine learning techniques and MLOps best practices.

## Table of Contents
- [Manga Recommender System](#manga-recommender-system)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Objective](#objective)
  - [Methodology](#methodology)
    - [Data Collection](#data-collection)
    - [Data-Driven Analysis](#data-driven-analysis)
  - [Project Structure 📁](#project-structure-)
    - [Modeling Approach](#modeling-approach)
    - [MLOps Integration](#mlops-integration)
  - [Features](#features)
  - [Installation](#installation)
    - [Option 1: Using Poetry](#option-1-using-poetry)
    - [Option 2: Using pip](#option-2-using-pip)
  - [Streamlit Demo 🚀](#streamlit-demo-)
  - [Contributors 🤝](#contributors-)
  - [Sources](#sources)

## Introduction
The Manga Recommender System leverages user preferences and reading history to suggest manga titles tailored to individual tastes. This project integrates machine learning, data engineering, and deployment strategies to build an intelligent and scalable recommendation engine.

## Objective
The main objective is to build an end-to-end intelligent recommender system tailored for manga enthusiasts. This system aims to:

- **Personalize Recommendations:** Analyze user preferences and reading histories to suggest new manga titles, helping users explore a wide variety of genres beyond just popular books.
- **Integrate MLOps Practices:** Ensure robust development, deployment, and continuous improvement of the recommendation model through automated pipelines and monitoring tools.

## Methodology
The project methodology encompasses the tools and techniques used to implement an end-to-end system with a focus on MLOps best practices:

### Data Collection
- Collect manga-related data through the **Anilist API**.
- Store and manage datasets using **Google Cloud Platform BigQuery**.

### Data-Driven Analysis
- Perform **Exploratory Data Analysis (EDA)** to extract insights and define key features for the recommendation model.
- Utilize visualization tools to understand trends and relationships in the data.

## Project Structure 📁
The project is organized as follows:
```
/manga_recommender_system
│── .github/workflows/       # CI/CD pipeline configuration
│   └── ci.yml               # GitHub Actions workflow file
│── notebooks/               # Jupyter notebooks for exploratory data analysis (EDA)
│   └── eda.ipynb            # EDA and visualization
│── src/                     # Source code for the recommendation system
│   │── app/                 # Streamlit app demo
│   │── config/              # Configuration management
│   │── data_collection/     # Manga & Users Data fetching scripts
│   │── preprocessing/        # Data preprocessing scripts
│   │── queries/        # SQL queries to fetch data from BigQuery
│   │── training/        # Training models and deploy workflows scripts
│   │── utils/        # helper scripts
│── .gitignore               # Git ignore file
│── .pre-commit-config.yaml  # Pre-commit hooks configuration
│── Makefile                 # Automation commands
│── README.md                # Project documentation
│── poetry.lock              # Poetry dependency lockfile
│── pyproject.toml           # Poetry project configuration
│── requirements.txt         # Python dependencies
│── setup.cfg                # Setup configuration
```

### Modeling Approach
There are three primary recommender system methods:
1. **Collaborative Filtering** - Recommends manga that similar users have read.
2. **Content-Based Filtering** - Recommends manga based on their characteristics.
3. **Hybrid Approach** - Combines both methods for improved accuracy.

Currently, the project focuses on **Collaborative Filtering**. We aim to research and test multiple algorithms from Scikit-Surprise while tracking performance using **MLflow** and **Prefect**.

### MLOps Integration
- **CI/CD Pipelines:** Automate testing, integration, and deployment of new models using tools like pre-commit and GitHub Actions.
- **Version Control:** Maintain code and model versioning using **Git** and tracking tools like **MLflow**.
- **Containerization:** Deploy model using **Docker** for consistency across environments.

## Features
- **Personalized Recommendations** based on user behavior and preferences.
- **Automated Data Pipeline** for collecting and updating manga data.
- **Model Performance Tracking** using MLflow.
- **Cloud-Based Storage** for scalability and efficiency.

## Installation

This project uses Python for data analysis. To set up the environment, you can use either **Poetry** or **pip**.

### Option 1: Using Poetry
Poetry is recommended for managing dependencies and virtual environments.

1. Install Poetry (if not already installed):
   ```bash
   pip install poetry
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Option 2: Using pip
Alternatively, you can use `pip` to install dependencies.

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```

2. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Streamlit Demo 🚀

1. Clone the repository:
   ```bash
   git clone https://github.com/Trung-Dan-Phan/manga_recommender_system.git
   cd path/to/project-directory
   ```

2. Install dependencies (using Poetry or pip as described above).

3. Launch the POC
   ```bash
   cd .\src\
   $env:PYTHONPATH = (Get-Location).Path
   $ streamlit run app/streamlit_demo.py
   ```

4. Access the streamlit app: [localhost:8501](http://localhost:8501)

---

## Contributors 🤝

This project was developed by Dan Phan as part of a Personal Project for the Master of Science Data Science and AI for Business at HEC Paris.

## Sources
The project leverages the following tools and datasets:
- **Manga Databases & APIs:** [Anilist API](https://docs.anilist.co/).
- **BigQuery, MLflow, Prefect** for data management and MLOps.
- [**Scikit-Surprise**](https://surprise.readthedocs.io/en/stable/) for building recommendation models.
