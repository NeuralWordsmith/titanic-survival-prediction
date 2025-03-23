# Titanic Survival Prediction

This project is a complete end-to-end machine learning pipeline built for the Titanic dataset from Kaggle. It includes data preprocessing, feature engineering, model training, and test set prediction, along with notebook outputs suitable for reproducible analysis and leaderboard submission.

---

## About the Project

The goal is to predict passenger survival using structured data from the Titanic ship manifest. The dataset includes features such as passenger class, age, sex, family size, and more.

This repository includes:

- A modular codebase (`src/`) for data loading and preprocessing
- Two Jupyter Notebooks:
  - `titanic_final_pipeline.ipynb` – clean preprocessing + logistic regression pipeline
  - `titanic_rf_highscore.ipynb` – optimized Random Forest model
- Exported `submission.csv` for Kaggle upload

---

## Directory Structure

```
P3_titanic_survival_prediction/
│
├── data/                     # Raw input CSVs (train/test from Kaggle)
│   └── raw/
├── outputs/
│   └── predictions/          # Final submission file
├── src/
│   ├── data_loading.py       # Loads raw datasets
│   └── preprocessing.py      # All preprocessing & feature engineering
├── titanic_final_pipeline.ipynb
├── titanic_rf_highscore.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Set up environment

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the raw data

Download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) and place them inside:

```
data/raw/
```

### 4. Run the Notebook

Open either notebook:

```bash
jupyter notebook
```

- Run `titanic_final_pipeline.ipynb` for a clean Logistic Regression pipeline
- Or run `titanic_rf_highscore.ipynb` for an optimized Random Forest model

This will generate `submission.csv` inside `outputs/predictions/`.

---

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it as long as proper attribution is given.

---
