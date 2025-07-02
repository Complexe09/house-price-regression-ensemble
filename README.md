# House Price Prediction — Ensemble Regression

This project uses the popular **Kaggle House Prices: Advanced Regression Techniques** dataset to predict housing prices using a **stacked ensemble model**.  
It includes thorough **feature engineering**, **log transformation**, and **model tuning**.

<br>

## Dataset

- [Kaggle Dataset Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Rows: 1460 (training set)
- Task: Regression (predict `SalePrice`)
- Target is **log-transformed** to handle skew and improve performance.

<br>

## Techniques Used

- **Log transformation** of target (`log(SalePrice)`)
- **Feature engineering**:
  - `TotalSF` = basement + 1st + 2nd floor (log-transformed)
  - `Age²` = squared age of the house
  - `TotalBath`, `Qual×Area`, `HasPool`, etc.
- 🔍 Handling skewed numerical features (`LotFrontage`, `GrLivArea`)
- 🧹 Preprocessing pipelines (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
- 🧠 Models:
  - `RidgeCV` (regularized linear model)
  - `RandomForestRegressor`
  - `XGBRegressor`
  - Combined using `StackingRegressor`

<br>

## ⚙️ Models Compared

| Model       | RMSE (log target) | R² Score |
|-------------|-------------------|----------|
| RidgeCV     | ~0.15             | ~0.88    |
| RandomForest| ~0.14             | ~0.89    |
| XGBoost     | ~0.13             | ~0.90    |
| **Stacked** | **~0.14**         | **~0.89** |

> 💡 *Results may vary slightly depending on train-test split and random seed.*


