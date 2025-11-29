# Breast Cancer Prediction Using Machine Learning

This project applies multiple supervised machine learning algorithms to the Breast Cancer Wisconsin Diagnostic Dataset in order to classify tumors as malignant or benign. The work includes exploratory data analysis, correlation study, feature scaling, model training, evaluation, and comparison across several classifiers such as Logistic Regression, Random Forest, XGBoost, Naive Bayes, and Gradient Boosting.

---

## Dataset Overview

The dataset contains **569 samples** and **32 columns**, including:

- `diagnosis`: Target variable (M = malignant, B = benign)
- 30 real-valued features describing cell nuclei measurements from digitized images
- `id` column identifying each sample

After preprocessing, no missing values remain in the dataset. Numerical features include:

- Radius, Texture, Perimeter, Area  
- Smoothness, Compactness, Concavity  
- Concave points, Symmetry, Fractal dimension  
- Standard error (SE) features  
- Worst-case values  

Target variable (`diagnosis`) is label encoded:
- 1 = Malignant  
- 0 = Benign  

---

## Exploratory Data Analysis

Several descriptive analyses were performed, including:

### 1. Class Distribution
Benign (0): 357 samples  
Malignant (1): 212 samples  

The dataset is moderately imbalanced but still suitable for standard ML classification.

### 2. Summary Statistics  
Descriptive statistics were generated for all 30 feature columns, showing variations in central tendency and spread.

### 3. Correlation Analysis  
Correlation matrices and heatmaps demonstrated strong relationships among:
- Radius, perimeter, and area features  
- Concavity and compactness families of features  
- Most “worst” features showed high correlation with diagnosis  

### 4. Visualizations  
- Heatmaps for subsets of the data  
- Pair plots for selected features  
- Diagnosis distribution plot  

---

## Data Preprocessing

The following preprocessing steps were applied:

1. Dropped irrelevant or empty columns (no missing values remained)
2. Label encoded `diagnosis` (M → 1, B → 0)
3. Extracted feature matrix `X` (30 numerical features)
4. Extracted target vector `y`
5. Performed an 80/20 train–test split
6. Standardized features using `StandardScaler`

---

## Machine Learning Models

The project includes multiple classifiers trained and compared:

### 1. Logistic Regression
- Training Accuracy: **0.989**
- Test Accuracy: **0.956**
- Strong precision and recall for both classes  

### 2. Random Forest Classifier
- Test Accuracy: **0.9649**
- Confusion Matrix:
  - True negatives: 65  
  - True positives: 45  

### 3. XGBoost Classifier
- Test Accuracy: **0.9649**
- Balanced performance with low misclassification  

### 4. Gaussian Naive Bayes
- Test Accuracy: **0.9386**
- Slightly lower recall for malignant tumors  

### 5. Gradient Boosting Classifier
- Test Accuracy: **0.9649**
- Very strong recall for benign class; effective overall  

### Summary of Model Accuracy

| Model                     | Accuracy |
|---------------------------|----------|
| Logistic Regression       | 0.9561   |
| Random Forest            | 0.9649   |
| XGBoost                  | 0.9649   |
| Naive Bayes              | 0.9386   |
| Gradient Boosting        | 0.9649   |

The top-performing models (Random Forest, XGBoost, Gradient Boosting) achieved similar accuracy.

---

## Model Saving

The Logistic Regression model was saved using:

```python
pickle.dump(log, open("model.pkl", "wb"))
```
## How to Run
- Place Breast_data.csv in your working directory.
- Install required libraries:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
- Run the Python script or Jupyter notebook.
- Sections should be executed in the following order:
  - Data loading and inspection
  - Statistical summary and EDA
  - Correlation and pair plots
  - Preprocessing (encoding, scaling, train–test split)
  - Model training (Logistic Regression, RF, XGBoost, etc.)
  - Evaluation (accuracy, confusion matrix, classification report)
  - Optional: save/load model with pickle

## Observations & Notes

- Most “worst” features (e.g., radius_worst, perimeter_worst) show the highest correlation with malignancy.
- Radius, perimeter, and area families contribute significantly to prediction performance.
- Tree-based models (RF, XGB, GB) outperform linear models slightly.
- Dataset is small but high-quality, allowing models to generalize well without overfitting.
- Standardization is essential for models like Logistic Regression but less important for tree-based methods.
- XGBoost and Random Forest achieve the best balance of accuracy and robustness.

## Possible Extensions

- Hyperparameter tuning using GridSearchCV or Bayesian optimization
- Apply dimensionality reduction (PCA or LDA)
- Try more advanced algorithms such as LightGBM or CatBoost
- Build a small web app for diagnosis prediction
- Use SHAP or LIME to interpret feature contribution
- Evaluate ensemble combinations of best models
