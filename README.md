# Practical Application II — Module 11  
## What Drives the Price of a Used Car?

**Author:** Siddarth R Mannem  
**Program:** AI/ML Certificate — Practical Application II (Module 11)  
**Dataset:** `vehicles.csv` (Kaggle Used Cars subset, ~426,000 records)  
**Audience:** Used Car Dealership Executives & Data Strategy Teams  

---

## Business Understanding
The goal of this analysis is to help a **used car dealership** understand which factors drive **car resale prices**.  
Dealers need to know which attributes — such as **year**, **mileage**, **condition**, and **title status** — have the strongest influence on value so they can refine **pricing and inventory acquisition** strategies.

### Key Objectives
1. Identify the strongest predictors of used car price.  
2. Build regression models to estimate price based on vehicle attributes.  
3. Provide actionable recommendations for pricing and inventory decisions.  

**Primary Metric:**  
- **RMSE (Root Mean Squared Error):** measures average prediction error magnitude.  
- **MAE (Mean Absolute Error):** interpretable error in dollar terms.  

---

## Data Understanding
The dataset includes 426,000 used car listings from a Kaggle dataset containing features such as:
- `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`,  
  `odometer`, `title_status`, `transmission`, `drive`, `type`, etc.  
- Target variable: **`price`**

### Data Exploration Summary
- Price distribution was right-skewed with outliers beyond \$150,000.  
- Cars newer than 2015 and with <100,000 miles commanded higher prices.  
- Missing values appeared in categorical columns like `model`, `drive`, and `type`.  

---

## Data Preparation
To ensure modeling accuracy and reproducibility:

1. **Removed missing prices and extreme outliers**  
   - Clipped to the 1st–99th percentile (bounded \$500–\$150,000).  
2. **Created a clean modeling dataset (`df_model`)**  
   - Retained a copy of the raw `df` for reference.  
3. **Handled missing values and scaling using pipelines:**  
   - Numeric pipeline → `SimpleImputer(median)` + `StandardScaler()`  
   - Categorical pipeline → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`  
4. **Split dataset** into 80% training / 20% test.  
5. Combined transformations using a unified `ColumnTransformer`.

---

## Modeling
Models were developed and validated using **cross-validation (CV)** and **grid search** for hyperparameter tuning.

| Model                                 | Description | CV/Hyperparameters | RMSE | MAE | Notes |
|---------------------------------------|--------------|--------------------|------|-----|-------|
| **Baseline**                          | Mean predictor | — | High | High | Benchmark only |
| **Linear Regression**                 | OLS, full dataset | 3-fold CV | ↓↓ | ↓↓ | Strong linear fit |
| **Ridge Regression (L2)**             | Regularized model | α ∈ [0.01, 100] | ↓ | ↓ | Best stability |
| **Lasso Regression (L1)**<br/>(NOT-Active) | Trained on 10% sample | α ∈ [0.01, 100] | Slightly ↑ | Slight bias | Demonstrates feature selection |

**Reasoning:**  
- Ridge chosen as the main model due to its stability on large datasets and efficient convergence.  
- Lasso performed on a 10% training sample to demonstrate L1 regularization efficiently (reduces runtime without losing interpretability).  

---

## Evaluation
### Model Performance
- **Ridge Regression** achieved the best balance between RMSE and MAE on the test set.  
- **Residual analysis** showed no major bias — errors centered around zero.  
- **Top predictive features** (by absolute coefficient magnitude):
  | Feature | Effect | Description |
  |----------|---------|-------------|
  | `year` | ↑ | Newer cars have higher prices |
  | `odometer` | ↓ | Higher mileage reduces value |
  | `condition_excellent` | ↑ | Adds strong price premium |
  | `title_status_clean` | ↑ | Clean titles increase resale value |
  | `drive_4wd` / `drive_awd` | ↑ | Improves resale due to performance |
  | `transmission_manual` | ↓ | Slightly lowers value vs automatic |

**Metrics for Final Ridge Model**
| Metric | Value |
|---------|--------|
| RMSE | ~\$5,000–6,000 |
| MAE | ~\$3,000–4,000 |

---

## Deployment & Recommendations
The model delivers data-driven insights to guide dealership strategy.

### Business Insights
1. **Newer vehicles** with **lower mileage** retain the highest resale value.  
2. **Excellent condition** cars can command significant premiums.  
3. **Clean titles** and **4WD/AWD drivetrains** increase buyer willingness to pay.  
4. **Manual transmissions** and **salvage titles** reduce perceived value.  

### Recommendations for Dealership
- **Acquisition Strategy:** Focus on late-model, clean-title cars with moderate mileage.  
- **Pricing Strategy:**  
  - Use model predictions as a starting point for listing prices.  
  - Apply condition-based markup tiers (+5–15% for “excellent”).  
- **Inventory Optimization:**  
  - Reduce high-mileage, salvage, or manual transmission stock.  
- **Future Enhancements:**  
  - Incorporate **regional trends** and **temporal (year-over-year)** effects.  
  - Build a web dashboard for dynamic pricing.

---

## Key Takeaways
- Ridge Regression provides stable, interpretable, and scalable modeling for large datasets.  
- Regularization improves generalization and reduces overfitting.  
- Data-driven insights align strongly with business intuition — newer, cleaner cars cost more.  

---

## 📂 Project Structure
