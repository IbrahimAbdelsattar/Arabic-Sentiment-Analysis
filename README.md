# Arabic & Egyptian Sentiment Analysis

A text classification project for Arabic and Egyptian dialect reviews, combining notebook experiments with an Arabic Streamlit interface.

**Technology:** Python · scikit-learn · TensorFlow/Keras · Streamlit

## Features

- Classify entered text as negative, neutral, or positive using TF-IDF and Logistic Regression in the web app.
- Compare classical classifiers and an embedding-based Keras network in the training notebook.
- Keep separate Keras and TensorFlow Lite exports for the neural experiments.

## Repository guide

| Path | Purpose |
|---|---|
| [app.py](app.py) | Arabic Streamlit inference interface. |
| [arabic-egypt-sentiment-analysis.ipynb](arabic-egypt-sentiment-analysis.ipynb) | Review exploration, text preparation, model comparison, and exports. |
| [Requirements.txt](Requirements.txt) | Dependencies for the inference app. |
| [Egyptian Reviews Dataset.rar](Egyptian%20Reviews%20Dataset.rar) | Archived review dataset. |
| [sentiment_model.h5](sentiment_model.h5) | Keras model export. |

## Requirements and current limitations

The app specifically loads `logistic_model.pkl` and `tfidf_vectorizer (1).pkl`; neither is present in the current repository. Export the matching classifier and fitted vectorizer from training before launching it. The included Keras/TFLite files are separate artifacts and cannot replace these two files directly.

The notebook uses a Kaggle dataset path. Extract the dataset and adjust that path for local execution. Its training dependencies exceed the small app requirements file; install the packages imported by the notebook in a separate training environment. Preserve the app's label mapping: `0 = negative`, `1 = neutral`, `2 = positive`. The committed runtime manifest omits scikit-learn/joblib dependencies needed by the saved preprocessing or model artifacts; the supplemental install command supplies them.

## Getting started

```bash
git clone https://github.com/IbrahimAbdelsattar/Arabic-Sentiment-Analysis.git
cd Arabic-Sentiment-Analysis
```

Use a Python virtual environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -r Requirements.txt
python -m pip install scikit-learn joblib
python -m streamlit run app.py
```
