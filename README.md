# Article-Clustering

````markdown
# SMS Spam Detection using Stacking Ensemble & Meta Features

This project implements an advanced SMS spam detection system using a **stacking ensemble** of multiple classifiers, enriched with **meta-features** and **TF-IDF-based text features**. The pipeline also incorporates feature selection and hyperparameter tuning via GridSearchCV.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Base Learners](#base-learners)
5. [Stacking Ensemble](#stacking-ensemble)
6. [Pipeline & GridSearchCV](#pipeline--gridsearchcv)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Dependencies](#dependencies)

---

## Dataset
We use the **SMS Spam Collection Dataset** from Kaggle:

- **Source:** [SMS Spam Collection Dataset](https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv)
- **Description:** Contains 5,574 SMS messages labeled as `ham` (non-spam) or `spam`.
- **Columns used:** 
  - `label` (target: ham/spam)
  - `text` (raw SMS content)

---

## Data Preprocessing
- Dropped unnecessary columns (`Unnamed: 2,3,4`).
- Renamed remaining columns to `label` and `text`.
- Text cleaning:
  - Removed non-alphabetic characters
  - Converted to lowercase
  - Removed stopwords
  - Lemmatized words using `WordNetLemmatizer`

```python
data['clean'] = data['text'].apply(clean_text)
````

---

## Feature Engineering

1. **TF-IDF Features:**

   * **Word-level:** 1-2 grams
   * **Character-level:** 2-4 grams
2. **Meta-features:** Custom features capturing characteristics common in spam messages:

   * Message length
   * Number of digits
   * Number of punctuations
   * Number of ALL CAPS words
   * Presence of URLs
   * Occurrences of the word `free`

```python
class MetaFeatures(BaseEstimator, TransformerMixin):
    ...
```

3. **Feature Union:** Combines word TF-IDF, char TF-IDF, and meta-features into a single feature matrix.

---

## Base Learners

The stacking ensemble uses the following base classifiers:

| Model                        | Notes                                                  |
| ---------------------------- | ------------------------------------------------------ |
| Multinomial Naive Bayes (NB) | Simple probabilistic classifier                        |
| XGBoost (XGB)                | Gradient boosting with handling for imbalanced classes |
| LightGBM (LGBM)              | Gradient boosting, class balanced                      |
| Logistic Regression (LR)     | Regularized linear model                               |
| Random Forest (RF)           | Ensemble of decision trees                             |

---

## Stacking Ensemble

* Base learners: NB, XGB, LGBM, LR, RF
* Meta-model: XGBoost
* 5-fold cross-validation
* Final predictions are generated using the meta-model over base learners’ outputs

```python
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=XGBClassifier(...),
    cv=5,
    n_jobs=-1,
    passthrough=False
)
```

---

## Pipeline & Hyperparameter Tuning

* **Pipeline Steps:**

  1. `FeatureUnion` for TF-IDF + meta-features
  2. `SelectKBest` for top 3,000 features based on `chi2`
  3. `StackingClassifier` for ensemble learning

* **GridSearchCV:** Tunes hyperparameters for base learners

```python
param_grid = {
    'clf__nb__alpha': [0.5, 1.0],
    'clf__lr__C': [0.1, 1.0, 10.0],
    'clf__rf__n_estimators': [100, 150],
    'clf__xgb__scale_pos_weight': [3, 5, 7],
}
```

---

## Evaluation

* Metrics reported: **F1-score (macro)**, **precision**, **recall**
* Confusion matrix displayed for model performance

```python
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

* The model handles class imbalance using `class_weight` in some base learners and `scale_pos_weight` in XGBoost.

---

## Usage

1. Clone the repository.
2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm nltk
   ```
3. Run the script:

   ```bash
   python spam_detection.py
   ```

---

## Dependencies

* Python >= 3.8
* pandas
* numpy
* scikit-learn
* xgboost
* lightgbm
* nltk

**NLTK Data:**

```python
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Notes

* The stacking ensemble with meta-features improves spam detection compared to using only TF-IDF.
* The meta-features are particularly effective at capturing patterns unique to spam messages (e.g., ALL CAPS words, URLs, "free").

---

## Author

Reihan – Machine Learning & Data Science Enthusiast

```

