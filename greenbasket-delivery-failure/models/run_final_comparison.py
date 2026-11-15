import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, RocCurveDisplay,
                             PrecisionRecallDisplay)

root = Path(__file__).resolve().parent.parent
models = root / 'models'
figures_dir = root / 'outputs' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
processed = root / 'data' / 'processed'
X_test = pd.read_csv(processed / 'X_test_preprocessed.csv')
y_test = pd.read_csv(processed / 'y_test.csv').iloc[:, 0]
X_valid = pd.read_csv(processed / 'X_validation_preprocessed.csv')
y_valid = pd.read_csv(processed / 'y_validation.csv').iloc[:, 0]

nn_results = pd.read_csv(models / 'nn_tuning_validation_results.csv')
ann_row = nn_results.sort_values(['f1', 'pr_auc'], ascending=False).iloc[0]
xgb_results = pd.read_csv(models / 'validation_model_comparison_with_tuned_xgboost.csv')
xgb_row = xgb_results.loc[xgb_results['model'].eq('Tuned XGBoost')].iloc[0]

ann = tf.keras.models.load_model(models / 'ann_tuned_01.keras')
xgb = joblib.load(models / 'tuned_xgboost.joblib')
candidates = [
    ('ANN Tuned 01', ann, float(ann_row.threshold), True),
    ('Tuned XGBoost', xgb, float(xgb_row.threshold), False),
]
rows = []
scores_by_model = {}
for name, model, threshold, is_ann in candidates:
    scores = (model.predict(X_test.astype('float32'), verbose=0).ravel()
              if is_ann else model.predict_proba(X_test)[:, 1])
    preds = (scores >= threshold).astype(int)
    rows.append({'model': name, 'threshold': threshold,
                 'accuracy': accuracy_score(y_test, preds),
                 'precision': precision_score(y_test, preds, zero_division=0),
                 'recall': recall_score(y_test, preds, zero_division=0),
                 'f1': f1_score(y_test, preds, zero_division=0),
                 'pr_auc': average_precision_score(y_test, scores),
                 'roc_auc': roc_auc_score(y_test, scores)})
    scores_by_model[name] = (scores, preds)
    ConfusionMatrixDisplay.from_predictions(y_test, preds, display_labels=['Successful', 'Failed'], cmap='Blues', colorbar=False)
    plt.title(f'Final Test Confusion Matrix: {name}')
    plt.tight_layout(); plt.savefig(figures_dir / f'{name.lower().replace(" ", "_")}_final_test_confusion_matrix.png', dpi=300); plt.close()

comparison = pd.DataFrame(rows).sort_values(['f1', 'pr_auc'], ascending=False).reset_index(drop=True)
comparison.to_csv(models / 'final_model_test_comparison.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, (scores, _) in scores_by_model.items():
    RocCurveDisplay.from_predictions(y_test, scores, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, scores, name=name, ax=axes[1])
axes[0].set_title('Final Test ROC Comparison'); axes[1].set_title('Final Test Precision-Recall Comparison')
plt.tight_layout(); plt.savefig(figures_dir / 'final_model_test_roc_pr_comparison.png', dpi=300); plt.close()

# XGBoost built-in feature importance.
xgb_importance = pd.DataFrame({'feature': X_test.columns, 'importance': xgb.feature_importances_}).nlargest(20, 'importance').sort_values('importance')
ax = xgb_importance.plot.barh(x='feature', y='importance', legend=False, color='teal', figsize=(10, 7), title='Tuned XGBoost Feature Importance')
ax.set_xlabel('Relative importance'); plt.tight_layout(); plt.savefig(figures_dir / 'tuned_xgboost_feature_importance.png', dpi=300); plt.close()

# ANN permutation importance on validation data, preserving the validation-selected threshold.
baseline = f1_score(y_valid, (ann.predict(X_valid.astype('float32'), verbose=0).ravel() >= float(ann_row.threshold)).astype(int))
rng = np.random.default_rng(53); drops = []
for feature in X_valid.columns:
    shuffled = X_valid.copy(); shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
    score = f1_score(y_valid, (ann.predict(shuffled.astype('float32'), verbose=0).ravel() >= float(ann_row.threshold)).astype(int))
    drops.append(baseline - score)
ann_importance = pd.DataFrame({'feature': X_valid.columns, 'f1_drop': drops}).nlargest(20, 'f1_drop').sort_values('f1_drop')
ax = ann_importance.plot.barh(x='feature', y='f1_drop', legend=False, color='violet', figsize=(10, 7), title='Tuned ANN Permutation Importance')
ax.set_xlabel('Validation F1 decrease when shuffled'); plt.tight_layout(); plt.savefig(figures_dir / 'ann_tuned_01_feature_importance.png', dpi=300); plt.close()

best = comparison.iloc[0].to_dict()
with open(models / 'final_model_recommendation.json', 'w', encoding='utf-8') as f:
    json.dump({'recommended_model': best['model'], 'selection_metric': 'final test F1', 'metrics': best}, f, indent=2)
print(comparison.to_string(index=False))
print(f"Recommended final model by test F1: {best['model']}")
