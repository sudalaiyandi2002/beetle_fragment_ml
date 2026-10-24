# Beetle fragment classification from numerical signatures

Code and results for the manuscript "Machine Learning-Based Identification of Beetle Species from Anatomical Fragments: Within-Fragment Classification, Cross-Fragment Generalisation and Feature Importance in Curculionidae and Tenebrionidae".

## Overview
This repository holds a secondary analysis of the public Numerical Signature Dataset of Curculionidae and Tenebrionidae beetle fragments (Serfa Juan & Gerken, Scientific Data 13, 2, 2026; dataset DOI 10.15482/USDA.ADC/29066444.v1). Each record has four image-derived descriptors (skewness, kurtosis, entropy, standard deviation) for an anatomical fragment of one of six stored-product beetle species. No new specimens or images were collected.

## What the analysis does
- Compares seven classifiers (Extra Trees, Random Forest, SVM-RBF, k-NN, CatBoost, histogram gradient boosting, XGBoost) using five-fold stratified cross-validation.
- Evaluates fragment-specific models and leave-one-fragment-out (cross-fragment) transfer, with chance baselines.
- Tests whether the descriptors can predict the fragment category itself.
- Estimates impurity-based and permutation feature importance.
- Repeats key analyses with cross-validation grouped by file number as a sensitivity check.

## Main findings
Extra Trees reached 94.08% (Curculionidae) and 91.83% (Tenebrionidae) accuracy under random folds, with similar results under grouped folds. Cross-fragment macro-F1 was close to chance (about 33%), while fragment category was predicted with 92-97% accuracy. The data contain no verified specimen-level identifier, so specimen-independent generalisation is not established.

## Files
- `beetle_reanalysis.py` runs all analyses and saves tables and figures.
- The notebook (.ipynb) contains the original exploratory analysis.
- `results/` holds the CSV tables, Figures 2 and 6, and versions.json.

## How to run
Open Google Colab, upload `beetle_reanalysis.py`, then run `!pip install -q xgboost catboost` and `%run beetle_reanalysis.py`. The script downloads the data from figshare and writes its outputs to `/content/reanalysis_outputs`. Set `QUICK=1` for a fast test run.

## Environment
Python 3.13.16, scikit-learn 1.6.1, NumPy 2.1.3, pandas 2.2.3, Matplotlib 3.10.0, XGBoost 3.4.1, CatBoost 1.2.10.

## Licence and citation
Code is released under the MIT licence. Please cite the dataset and the manuscript (citation to be added after publication).
