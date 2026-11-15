# GreenBasket Delivery Failure Prediction

An end-to-end machine-learning project that predicts whether a GreenBasket grocery order is likely to fail at the time it is placed. The project combines exploratory analysis, time-safe preprocessing, classification modeling, explainability, and operational recommendations.

## Business Context

GreenBasket Grocers is an online grocery-delivery company operating across 14 U.S. metropolitan markets with approximately 141 dark stores and 3,200 gig delivery partners. It fulfills roughly 85,000 orders per day through scheduled two-hour windows.

Delivery failures—late deliveries, refusals, incorrect addresses, unavailable customers, damaged perishables, and payment failures—affect approximately 8–10% of orders. Each failure costs an estimated $12–$28 in delivery, reverse-logistics, perishables, and support costs, in addition to customer-experience damage.

The current dispatch process treats orders uniformly. A reliable delivery-risk score enables operations to target actions such as confirmation calls, prepaid-payment nudges, priority routing, and slot reassignment only where they are likely to help.

## Objective

Build a binary classification model for `delivery_failed`:

- `1`: delivery failed
- `0`: delivery succeeded

The model scores risk at order placement using customer history, address, order composition, payment, scheduling, operational-load, and weather information. The intended outcome is fewer failed deliveries, less perishable waste, more efficient operations, and a better customer experience.

## Data Dictionary

The source dataset, `data/raw/greenbasket_deliveries.csv`, contains 45,000 order-level observations from a 45-day operational period.

### Identifiers and time

- `order_id`: unique order identifier
- `order_timestamp`: time at which the order was placed
- `customer_id`: unique customer identifier

### Customer history

- `customer_tenure_days`, `past_order_count`, `past_failure_count`, `past_failure_rate`
- `avg_past_order_value_usd`, `days_since_last_order`

### Delivery address and location

- `delivery_city`, `delivery_pin_code`, `is_new_address`, `address_type`, `floor_number`
- `distance_from_darkstore_km`, `pin_code_failure_rate`

### Order and payment details

- `order_value_usd`, `num_items`, `num_perishable_items`, `num_frozen_items`
- `has_alcohol`, `payment_mode`, `discount_applied_pct`

### Schedule and operations

- `delivery_slot`, `day_of_week`, `is_weekend`, `lead_time_hours`
- `partner_current_load`, `darkstore_utilization_pct`
- `weather_condition`, `temperature_f`

### Target

- `delivery_failed`: binary delivery-failure outcome

## Notebook Workflow

The complete workflow is documented in [GreenBasket_Delivery_Failure_Full.ipynb](notebooks/GreenBasket_Delivery_Failure_Full.ipynb).

### Data overview and quality audit

- Loaded the raw dataset and inspected first/last observations, shape, types, descriptive statistics, missingness, and duplicates.
- Exported a reusable data-quality report to `outputs/raw_data_quality_report.txt`.
- Found 45,000 records, 32 raw fields, no duplicate rows, and concentrated missingness in customer-history value and recency fields.

### Exploratory data analysis

- Performed univariate, bivariate, multivariate, and interaction analysis.
- Examined target imbalance, distributions, skewness, outliers, operational patterns, weather, payment mode, location, customer history, and missingness.
- Generated and saved EDA figures in `outputs/figures/`.
- Identified meaningful drivers including lead time, distance from dark store, payment mode, delivery timing, weather, operational load, historical customer risk, and pin-code risk.

### Preprocessing and feature engineering

- Converted timestamps and engineered day/hour plus cyclic time features.
- Used chronological splits: 70% training, 15% validation, and 15% locked test data.
- Created time-safe customer and pin-code historical failure features to reduce leakage risk.
- Applied training-only median imputation, missingness indicators, rare pin-code grouping, frequency encoding, one-hot encoding, and standard scaling where appropriate.
- Saved cleaned and processed data under `data/interim/` and `data/processed/`.

### Model building and comparison

Baseline candidates included:

- Dummy classifier
- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting
- XGBoost
- Artificial Neural Networks using Adam and SGD with momentum

F1 was selected as the primary comparison metric because delivery failures are a minority class and the business needs a balance between catching failures and avoiding excessive interventions. Recall, precision, PR-AUC, ROC-AUC, calibration, and confusion matrices were retained as supporting measures.

### Hyperparameter tuning

- Tuned XGBoost with time-aware randomized search and expanding-window validation.
- Tuned ANN architecture, layer widths, dropout, L2 regularization, optimizer, learning rate, batch size, and class weight.
- Compared the strongest tuned ANN with tuned XGBoost on the same chronological validation and locked test partitions.

## Model Performance and Final Selection

Tuned XGBoost was selected as the final model based on locked test performance:

- F1: 0.445
- Precision: 0.408
- Recall: 0.490
- PR-AUC: 0.425
- ROC-AUC: 0.829

The tuned ANN remained a strong challenger, with better test precision (0.414) but lower F1 (0.425) and recall (0.436). XGBoost was selected because it captured more failed deliveries while maintaining useful precision and stronger ranking metrics.

Model artifacts and performance files are stored in `models/`. Final charts, confusion matrices, feature-importance plots, ROC/PR comparisons, EDA charts, and SHAP visualizations are stored in `outputs/figures/`.

## Explainability and Reviews

The selected XGBoost model was interpreted with built-in feature importance and SHAP analysis.

Key SHAP drivers include:

- `lead_time_hours`
- `temperature_f`
- `distance_from_darkstore_km`
- `pin_code_failure_rate`
- order-time features
- cash-on-delivery payment mode
- new-address status
- historical customer failure rate

SHAP global bar, beeswarm, dependence, and waterfall plots support both global governance and individual high-risk-order review. Feature importance is predictive rather than causal; operational interventions should therefore be tested through controlled pilots.

## Actionable Insights and Recommendations

- Deploy Tuned XGBoost to score the target `delivery_failed = 1`.
- Start from the validation-selected threshold of approximately 0.560 and adjust it to daily operations-review capacity.
- At the observed test profile, the model alerts on about 139 of every 1,000 orders, identifying roughly 57 failed-delivery risks while producing about 82 false alerts.
- Use tiered actions: human review for high-risk orders, automated payment/address checks for medium-risk orders, and standard handling for low-risk orders.
- Prioritize routing or slot actions for long lead times and long distances; use weather-aware monitoring for temperature risk; use confirmation workflows for cash-on-delivery and new-address orders.
- Monitor weekly rolling performance and trigger review when F1, recall, precision, or PR-AUC declines materially from locked-test benchmarks; also monitor feature drift, missingness, category drift, alert volume, and calibration.
- Preserve strict time safety for historical risk features and validate intervention impact with pilots before scaling.

## Project Structure

```text
data/
  raw/          Source delivery dataset
  interim/      Cleaned dataset
  processed/    Chronological train, validation, and test datasets
models/         Fitted models, tuning artifacts, metrics, and model reports
notebooks/      Complete analytical notebook and HTML export
outputs/
  figures/      EDA, final-model, feature-importance, and SHAP figures
```

## Learning

This exercise demonstrates how an AI/ML project becomes useful only when technical rigor is connected to a business decision.

As a student, you learn to begin with the operational problem rather than a model. Delivery failure is not simply a binary label: it has intervention costs, customer-experience consequences, capacity limits, and time-sensitive information. That context determines the target, the split strategy, the evaluation metric, and the deployment threshold.

You also learn that good modeling requires disciplined data work. Missingness can be informative, high-cardinality variables need deliberate encoding, and historical aggregates can leak future information unless they are constructed from strictly prior records. A chronological split is often more realistic than a random split when the production task predicts future events.

The project shows why multiple models should be compared. A neural network may lead on one metric, while a tree-based model may rank risk more effectively or capture more failures. Final selection should reflect the business objective, not just a leaderboard score.

Finally, explainability and monitoring are part of responsible ML practice. SHAP plots help translate model behavior into operational playbooks, while drift and performance monitoring keep a once-good model from silently becoming unreliable. These habits—framing the problem, preventing leakage, evaluating thoughtfully, explaining results, and monitoring deployment—transfer directly to real AI/ML work.
