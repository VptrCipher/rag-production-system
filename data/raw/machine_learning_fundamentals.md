# Machine Learning Fundamentals

## What is Machine Learning?

Machine learning (ML) is a subfield of artificial intelligence that gives systems the ability to learn and improve from experience without being explicitly programmed. Instead of writing rules, you provide data and the algorithm figures out the rules.

### Three Main Types of Machine Learning

**Supervised Learning**: The model learns from labeled training data. Each example has an input and a desired output.
- Classification: predicting a category (e.g., spam vs not spam)
- Regression: predicting a continuous value (e.g., house prices)
- Examples: Linear Regression, Logistic Regression, Decision Trees, SVMs, Neural Networks

**Unsupervised Learning**: The model finds patterns in unlabeled data.
- Clustering: grouping similar data together (K-Means, DBSCAN, Hierarchical)
- Dimensionality Reduction: compressing data while retaining structure (PCA, t-SNE, UMAP)
- Anomaly Detection: finding unusual data points
- Generative Models: learning to generate new data (GANs, VAEs)

**Reinforcement Learning**: An agent learns by interacting with an environment and receiving rewards or penalties.
- Used in robotics, games (AlphaGo), trading, and now LLM fine-tuning (RLHF)
- Key concepts: agent, environment, state, action, reward, policy

---

## The Machine Learning Pipeline

### 1. Problem Definition
- Define the task (classification, regression, etc.)
- Identify the success metric (accuracy, F1, RMSE, AUC)
- Understand business constraints (latency, interpretability, fairness)

### 2. Data Collection and Exploration
- **EDA (Exploratory Data Analysis)**: understand distributions, correlations, outliers
- Check for class imbalance in classification tasks
- Identify missing values and their patterns (MCAR, MAR, MNAR)
- Tools: pandas, matplotlib, seaborn, plotly

### 3. Feature Engineering
- **Numerical features**: normalization (min-max), standardization (z-score), log transforms
- **Categorical features**: one-hot encoding, label encoding, target encoding
- **Text features**: TF-IDF, word embeddings, BERT embeddings
- **Time series**: lag features, rolling windows, Fourier transforms
- Feature selection: correlation analysis, mutual information, SHAP values

### 4. Model Selection
- Start simple (linear models) then add complexity
- Consider the bias-variance tradeoff
- Use cross-validation for reliable estimates
- Ensemble methods often outperform single models

### 5. Training
- Gradient descent and its variants (SGD, Adam, RMSprop)
- Batch size: mini-batch is typical (32–256 samples)
- Learning rate scheduling: warm-up, cosine annealing, reduce-on-plateau
- Regularization: L1 (Lasso), L2 (Ridge), Dropout, Early Stopping

### 6. Evaluation
- **Classification**: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
- **Regression**: MAE, MSE, RMSE, R², MAPE
- **Ranking**: NDCG, MRR, MAP
- Always evaluate on a holdout test set, never the training set

### 7. Deployment and Monitoring
- Package the model (pickle, ONNX, TorchScript)
- Serve via REST API (FastAPI, Flask, TorchServe)
- Monitor for data drift, concept drift, and performance degradation

---

## Key Algorithms

### Linear Regression
Models the relationship between features and a continuous target as a linear equation.
- `y = w₀ + w₁x₁ + w₂x₂ + ...`
- Loss: Mean Squared Error (MSE)
- Assumptions: linearity, homoscedasticity, no multicollinearity

### Logistic Regression
Despite the name, it's a classification algorithm. Applies a sigmoid function to output probabilities.
- Uses log-loss (binary cross-entropy)
- Highly interpretable, fast, and a good baseline

### Decision Trees
Splits data based on feature thresholds to create a tree of decisions.
- Prone to overfitting — mitigated by pruning, max_depth, min_samples_leaf
- Interpretable but unstable

### Random Forests
An ensemble of decision trees, each trained on a random subset of data and features.
- Reduces variance via averaging (bagging)
- Feature importance via mean decrease in impurity
- Robust to overfitting, handles missing data well

### Gradient Boosting (XGBoost, LightGBM, CatBoost)
Builds trees sequentially, each correcting the errors of the previous.
- More powerful than Random Forests on tabular data
- XGBoost won countless Kaggle competitions
- Key hyperparameters: n_estimators, learning_rate, max_depth, subsample

### Support Vector Machines (SVM)
Finds the hyperplane that maximally separates classes.
- Effective in high dimensions, works well with small datasets
- Kernel trick allows non-linear boundaries (RBF, polynomial kernels)
- Does not scale well to very large datasets

---

## The Bias-Variance Tradeoff

- **High Bias** (underfitting): model is too simple to capture the pattern
  - Symptoms: high training error, high test error
  - Fix: add more features, use a more complex model

- **High Variance** (overfitting): model memorizes training data but fails to generalize
  - Symptoms: low training error, high test error
  - Fix: regularize, get more data, reduce model complexity, use cross-validation

- **Sweet spot**: balance between bias and variance that minimizes total error

---

## Cross-Validation

Technique to estimate model performance on unseen data:

- **k-Fold**: split data into k folds, train on k-1, validate on 1, rotate
- **Stratified k-Fold**: preserves class distribution in each fold (important for imbalanced data)
- **Leave-One-Out (LOO)**: k = n, computationally expensive
- **Time-series split**: respects temporal order — no lookahead leakage

---

## Handling Imbalanced Data

Common in fraud detection, medical diagnosis, and rare event prediction:
- **Oversampling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Undersampling**: randomly reduce majority class
- **Class weights**: penalize misclassification of minority class more
- **Threshold tuning**: adjust decision threshold (default 0.5)
- **Evaluation**: use F1, AUC-PR instead of accuracy

---

## Feature Importance Methods

### Permutation Importance
Randomly shuffle a feature and measure the drop in model performance. Large drops = important feature.

### SHAP Values (SHapley Additive exPlanations)
Game-theory-based approach that assigns each feature a contribution to each individual prediction. Provides both global and local explanations.

### Partial Dependence Plots (PDP)
Show the marginal effect of a feature on the predicted outcome, averaging over all other features.

---

## Model Interpretability vs. Performance

| Model | Interpretability | Performance |
|---|---|---|
| Linear Regression | ★★★★★ | ★★ |
| Decision Tree | ★★★★ | ★★★ |
| Random Forest | ★★★ | ★★★★ |
| XGBoost | ★★ | ★★★★★ |
| Neural Network | ★ | ★★★★★ |

Use interpretable models in regulated industries (healthcare, finance). Use SHAP + explainability tools with complex models.

---

## Hyperparameter Tuning

- **Grid Search**: exhaustive search over a defined parameter grid
- **Random Search**: randomly sample combinations — often better than Grid Search
- **Bayesian Optimization**: uses a probabilistic model to guide search (Optuna, Hyperopt)
- **AutoML**: automated pipelines (H2O AutoML, Auto-sklearn, Google AutoML)

---

## Common ML Pitfalls

1. **Data leakage**: test data information leaks into training (e.g., normalizing before splitting)
2. **Not having a baseline**: always compare to a simple baseline (mean predictor, random classifier)
3. **Metric mismatch**: optimizing for accuracy when AUC or F1 is what matters
4. **Ignoring class imbalance**: accuracy of 99% means nothing if 99% of data is one class
5. **Not checking train/test distributions**: covariate shift between environments
6. **Overfitting to the validation set**: too many rounds of hyperparameter tuning on the same validation set
7. **Wrong cross-validation for time series**: using random splits instead of time-based splits
