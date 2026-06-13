# Wine Dataset: Statistical Analysis, Classification & Clustering

A complete data science pipeline on the **UCI Wine Recognition Dataset** (178 samples, 13 numeric chemical features, 3 wine cultivars), covering preprocessing, exploratory statistical analysis, classification, and clustering — with all results consolidated into a 15-page PDF report.

## Project Overview

| Task | Description |
|------|-------------|
| **A. Preprocessing & EDA** | Missing-value checks, descriptive statistics, correlation analysis, standardization, distribution analysis |
| **B. Classification** | SVM (RBF kernel) vs k-Nearest Neighbours — compared on Accuracy, Cohen's Kappa, and ROC-AUC |
| **C. Clustering** | k-Means vs DBSCAN — compared on SSE and Entropy (cluster purity vs. true labels) |

## Key Results

### Classification

| Metric | SVM (RBF) | kNN (k=5) |
|---|---|---|
| Accuracy | **0.9815** | 0.9444 |
| Cohen's Kappa | **0.9719** | 0.9167 |
| ROC-AUC | **1.0000** | 0.9953 |

### Clustering

| Metric | k-Means (k=3) | DBSCAN (eps=2.2) |
|---|---|---|
| SSE | 1277.93 | **965.69** |
| Entropy (bits) | **0.1898** | 0.8057 |
| Silhouette | **0.2849** | 0.0725 |

**Takeaway:** SVM outperforms kNN on every metric, achieving near-perfect class separation. k-Means produces clusters that align far more closely with the true cultivars than DBSCAN, despite DBSCAN's lower SSE — illustrating that SSE alone is an unreliable indicator of clustering quality.

## Repository Structure

```
wine_ml_project/
├── src/
│   ├── analysis.py          # Full pipeline: EDA, classification, clustering, plot generation
│   └── generate_report.py   # Builds the PDF report from results + plots
├── plots/                    # 12 generated visualizations (heatmaps, ROC curves, PCA, etc.)
├── data/
│   ├── wine_dataset.csv      # Dataset (exported from sklearn.datasets.load_wine)
│   ├── descriptive_stats.csv
│   └── results_summary.json  # All metrics in machine-readable form
└── Wine_Dataset_ML_Report.pdf  # Full 15-page report with plots & inferences
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy reportlab

python src/analysis.py        # Runs full pipeline, generates plots + results_summary.json
python src/generate_report.py # Builds the PDF report
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `scipy` · `reportlab`

## Dataset

[UCI Wine Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Wine) — chemical analysis of wines from three cultivars grown in the same region of Italy. Accessed via `sklearn.datasets.load_wine`.


## Suggested Improvements Implemented
- Fixed data leakage issue by scaling after train/test split.
- Prepared project for cross-validation and hyperparameter tuning extensions.
- Recommended adding Logistic Regression and Random Forest baselines.
