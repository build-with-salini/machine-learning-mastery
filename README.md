# 🤖 Machine Learning Mastery — 8-Week GitHub Roadmap

> **A hands-on, machine learning projects** which pairs **concept documentation** (README with LaTeX formulas) with an **end-to-end Python project** (Jupyter Notebook + real-world dataset).

---

## 📁 Repository Structure

```
ML-Roadmap/
│
├── README.md                          ← You are here (Main Roadmap)
│
├── Week-01-Linear-Regression/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-02-Advanced-Regression/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-03-Time-Series-Part1/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-04-Time-Series-Part2/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-05-Classification/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-06-Decision-Trees-and-Random-Forest/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
├── Week-07-Ensemble-Boosting/
│   ├── README.md
│   ├── Project_Notebook.ipynb
│   ├── data/
│   └── images/
│
└── Week-08-Unsupervised-and-Special-Topics/
    ├── README.md
    ├── Project_Notebook.ipynb
    ├── data/
    └── images/
```

---

## 🗺️ 8-Week Progress Tracker

> **Instructions:** Check off each item as you complete it. Update the status badges weekly.

---

### ✅ Week 1 — Linear Regression
**Theme:** *Foundations of Supervised Learning — Regression*

**Documentation**
- [ ] Simple Linear Regression (SLR) — equation `y = mx + c`, slope & intercept
- [ ] Multiple Linear Regression (MLR) — matrix form, feature coefficients
- [ ] Ordinary Least Squares (OLS) — derivation, cost minimization
- [ ] Gradient Descent — weight & bias update rules, learning rate α
- [ ] MSE Loss Function — mean squared error derivation
- [ ] 5 Assumptions of Linear Regression:
  - [ ] Linearity (scatter plot check)
  - [ ] No Multicollinearity (VIF & heatmap)
  - [ ] Normality of Residuals (KDE & Q-Q plot)
  - [ ] Homoscedasticity (residual scatter plot)
  - [ ] No Autocorrelation (ACF plot)
- [ ] Python implementation with `statsmodels` (`OLS`, `.summary()`)

**Project**
- [ ] Dataset acquired & loaded (e.g., Boston Housing / California Housing / Ames Housing)
- [ ] Data cleaning & preprocessing complete
- [ ] Exploratory Data Analysis (EDA) with Seaborn & Plotly
- [ ] OLS & Gradient Descent models built and compared
- [ ] Assumption checks visualized (Q-Q plot, VIF table, ACF plot)
- [ ] Model evaluation: MSE, RMSE, R², Adjusted R²
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 2 — Advanced Regression & Regularization
**Theme:** *Overfitting, Regularization & the Bias-Variance Tradeoff*

**Documentation**
- [ ] Polynomial Regression — degree selection, feature engineering
- [ ] Bias-Variance Tradeoff — decomposition, underfitting vs. overfitting
- [ ] Ridge Regression (L2) — penalty term λ(m²), shrinkage
- [ ] Lasso Regression (L1) — penalty term λ|m|, feature selection
- [ ] Elastic Net — combined L1 + L2, mixing parameter c
- [ ] Hyperparameter tuning — cross-validation, GridSearchCV

**Project**
- [ ] Dataset acquired & loaded (e.g., Diabetes Dataset / Insurance / Energy Efficiency)
- [ ] Data cleaning & preprocessing complete
- [ ] EDA with Seaborn & Plotly
- [ ] Polynomial, Ridge, Lasso, Elastic Net models compared
- [ ] Regularization path plots (alpha vs. coefficients)
- [ ] Bias-Variance tradeoff curve visualized
- [ ] Model evaluation: CV scores, MSE, R²
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 3 — Time Series Forecasting Part 1
**Theme:** *Decomposition, Smoothing & Exponential Models*

**Documentation**
- [ ] Time Series components: Trend, Seasonality, Noise/Residual
- [ ] Additive vs. Multiplicative decomposition
- [ ] Moving Average (MA) smoothing
- [ ] Simple Exponential Smoothing (SES) — formula & α parameter
- [ ] Double Exponential Smoothing (Holt's Method) — trend capture
- [ ] Triple Exponential Smoothing (Holt-Winters) — trend + seasonality

**Project**
- [ ] Dataset acquired & loaded (e.g., Air Passengers / Retail Sales / AQI data)
- [ ] Data cleaning, datetime indexing & resampling
- [ ] EDA — trend & seasonality visualized with Plotly
- [ ] Decomposition plot (additive & multiplicative)
- [ ] SES, Holt's, Holt-Winters models built and compared
- [ ] Forecast vs. actual plots
- [ ] Model evaluation: MSE, MAPE
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 4 — Time Series Forecasting Part 2
**Theme:** *Stationarity, ARIMA & Statistical Model-Based Forecasting*

**Documentation**
- [ ] Stationarity — definition, importance, visual checks
- [ ] ADF Test (Augmented Dickey-Fuller) — hypothesis, p-value interpretation
- [ ] Differencing — making series stationary
- [ ] AutoRegressive (AR) model — lag terms, p parameter
- [ ] Moving Average (MA) model — error terms, q parameter
- [ ] ARMA(p, q) — combined model
- [ ] ARIMA(p, d, q) — with differencing order d
- [ ] ACF & PACF plots — identifying p and q
- [ ] Model evaluation: MSE, MAPE

**Project**
- [ ] Dataset acquired & loaded (e.g., Stock prices / Weather / Electricity demand)
- [ ] Data cleaning & datetime preprocessing
- [ ] Stationarity check: ADF test + rolling mean/std plots
- [ ] ACF & PACF plots for parameter identification
- [ ] ARIMA model built, tuned, and forecasted
- [ ] Forecast confidence intervals plotted with Plotly
- [ ] Model evaluation: MSE, MAPE
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 5 — Classification Algorithms
**Theme:** *Logistic Regression, SVM & KNN*

**Documentation**
- [ ] Logistic Regression — sigmoid function `ŷ = 1/(1 + e^-z)`, decision boundary
- [ ] Binary Cross-Entropy Loss — formula & gradient
- [ ] Support Vector Machine (SVM):
  - [ ] Hyperplane, support vectors, margin maximization
  - [ ] Hard margin vs. soft margin (C parameter)
  - [ ] Kernels: linear, polynomial, RBF, sigmoid
  - [ ] Hinge Loss function
- [ ] K-Nearest Neighbors (KNN):
  - [ ] Euclidean & Manhattan distance
  - [ ] k hyperparameter selection (Elbow method)
  - [ ] Curse of dimensionality
- [ ] Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

**Project**
- [ ] Dataset acquired & loaded (e.g., Breast Cancer / Titanic / Heart Disease)
- [ ] Data cleaning, encoding, scaling
- [ ] EDA with Seaborn & Plotly (class distribution, correlations)
- [ ] Logistic Regression, SVM (multiple kernels), KNN models built
- [ ] Confusion matrices & ROC curves visualized
- [ ] Hyperparameter tuning (C for SVM, k for KNN)
- [ ] Model comparison table
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 6 — Decision Trees & Random Forest
**Theme:** *Tree-Based Learning, Bagging & Feature Importance*

**Documentation**
- [ ] Decision Tree for Classification:
  - [ ] Entropy — formula: `H = -Σ p·log₂(p)`
  - [ ] Information Gain (IG)
  - [ ] Gini Impurity
- [ ] Decision Tree for Regression:
  - [ ] Variance Reduction criterion
- [ ] Tree pruning — max_depth, min_samples_leaf (pre-pruning)
- [ ] Bagging (Bootstrap Aggregation) — sampling with replacement, variance reduction
- [ ] Random Forest:
  - [ ] Column (feature) sampling
  - [ ] √P features for classification; P/2 features for regression
  - [ ] Out-of-Bag (OOB) error
  - [ ] Feature Importance scores

**Project**
- [ ] Dataset acquired & loaded (e.g., Bank Churn / Credit Default / Employee Attrition)
- [ ] Data cleaning, encoding, preprocessing
- [ ] EDA with Seaborn & Plotly
- [ ] Decision Tree model built & visualized (graphviz / dtreeviz)
- [ ] Random Forest model built and tuned
- [ ] Feature importance bar chart (Plotly)
- [ ] OOB error curve plotted
- [ ] Model evaluation: Accuracy, Precision, Recall, F1, AUC
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 7 — Ensemble Boosting Methods
**Theme:** *AdaBoost, Gradient Boosting & XGBoost*

**Documentation**
- [ ] Voting Ensemble — hard & soft voting
- [ ] Stacking — meta-learner concept
- [ ] AdaBoost:
  - [ ] Decision stumps
  - [ ] Sample weight update rule
  - [ ] Classifier weight `e^λk` constant
- [ ] Gradient Boosting Machine (GBM):
  - [ ] Pseudo-residuals — fitting on errors
  - [ ] Learning rate (shrinkage)
  - [ ] Sequential decision trees
- [ ] XGBoost:
  - [ ] Similarity Score formula
  - [ ] Gain = Similarity(left) + Similarity(right) − Similarity(parent) − γ
  - [ ] Regularization λ (L2 on leaf weights)
  - [ ] Pruning with γ threshold
  - [ ] Comparison: GBM vs. XGBoost

**Project**
- [ ] Dataset acquired & loaded (e.g., Loan Default / Fraud Detection / Flight Delay)
- [ ] Data cleaning, encoding, class imbalance handling (SMOTE / class_weight)
- [ ] EDA with Seaborn & Plotly
- [ ] AdaBoost, GBM, XGBoost models built and compared
- [ ] Learning curves & staged prediction plots
- [ ] SHAP values for model explainability
- [ ] Hyperparameter tuning (n_estimators, max_depth, learning_rate)
- [ ] Model comparison: Accuracy, F1, AUC, training time
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

### ✅ Week 8 — Unsupervised ML & Special Topics
**Theme:** *Clustering, Association Rules, Recommendations & Dimensionality Reduction*

**Documentation**
- [ ] **K-Means Clustering:**
  - [ ] Centroid-based algorithm steps
  - [ ] WCSS (Within-Cluster Sum of Squares)
  - [ ] Elbow Method for optimal k
- [ ] **DBSCAN:**
  - [ ] eps & MinPts hyperparameters
  - [ ] Core, boundary & noise point classification
- [ ] **Hierarchical Clustering:**
  - [ ] Agglomerative (bottom-up) & Divisive (top-down)
  - [ ] Dendrogram visualization
  - [ ] Linkage methods: single, complete, average, centroid
- [ ] **Silhouette Score:** `S = (b − a) / max(a, b)`
- [ ] **Association Rule Mining (Apriori):**
  - [ ] Support = freq(X,Y) / N
  - [ ] Confidence = freq(X,Y) / freq(X)
  - [ ] Lift = Support / (Support_A × Support_B)
- [ ] **Recommendation Systems:**
  - [ ] Content-Based Filtering
  - [ ] Popularity-Based Filtering
  - [ ] Collaborative Filtering (user-based & item-based)
  - [ ] Cosine Similarity
- [ ] **PCA (Principal Component Analysis):**
  - [ ] Eigenvalues & eigenvectors
  - [ ] Covariance matrix
  - [ ] Standardization step
  - [ ] 8-step PCA algorithm
  - [ ] Explained variance ratio

**Project**
- [ ] Dataset acquired & loaded (e.g., Mall Customers / Online Retail / MovieLens)
- [ ] Data cleaning & preprocessing
- [ ] EDA with Seaborn & Plotly
- [ ] K-Means clustering + Elbow Method plot
- [ ] DBSCAN clustering (noise point visualization)
- [ ] Dendrogram for Hierarchical Clustering
- [ ] Silhouette Score comparison across methods
- [ ] Apriori association rules mined & visualized
- [ ] Collaborative Filtering recommendation system built
- [ ] PCA applied: explained variance plot, 2D/3D projection
- [ ] Interactive dashboard / final visualization complete
- [ ] `Project_Notebook.ipynb` committed

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Jupyter Notebook | Interactive development |
| Pandas & NumPy | Data wrangling |
| Scikit-learn | ML algorithms |
| Statsmodels | OLS, ARIMA |
| XGBoost | Gradient Boosting |
| Seaborn & Matplotlib | Static visualization |
| Plotly & Dash | Interactive visualization & dashboards |
| SHAP | Model explainability |

---

## 📌 Notes
- Each week's folder is self-contained — you can work through them independently.
- LaTeX-rendered formulas in READMEs require a Markdown renderer that supports math (e.g., GitHub with MathJax, Typora, or VS Code with extensions).
- Real-world datasets are sourced from Kaggle, UCI ML Repository, or `sklearn.datasets`.
