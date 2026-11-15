from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

root = Path(__file__).resolve().parent.parent
models_dir = root / 'models'
figures_dir = root / 'outputs' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)

X_validation = pd.read_csv(root / 'data' / 'processed' / 'X_validation_preprocessed.csv')
sample = X_validation.sample(n=min(1000, len(X_validation)), random_state=53)
model = joblib.load(models_dir / 'tuned_xgboost.joblib')
explainer = shap.TreeExplainer(model)
values = explainer.shap_values(sample)
base_value = explainer.expected_value

plt.figure()
shap.summary_plot(values, sample, plot_type='bar', max_display=20, show=False)
plt.title('SHAP Global Feature Importance: Tuned XGBoost')
plt.tight_layout(); plt.savefig(figures_dir / 'shap_global_bar.png', dpi=300, bbox_inches='tight'); plt.close()

plt.figure()
shap.summary_plot(values, sample, max_display=20, show=False)
plt.title('SHAP Beeswarm: Feature Impact on Delivery-Failure Risk')
plt.tight_layout(); plt.savefig(figures_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight'); plt.close()

plt.figure()
shap.dependence_plot('lead_time_hours', values, sample, interaction_index='distance_from_darkstore_km', show=False)
plt.title('SHAP Dependence: Lead Time and Distance Interaction')
plt.tight_layout(); plt.savefig(figures_dir / 'shap_lead_time_distance_scatter.png', dpi=300, bbox_inches='tight'); plt.close()

explanation = shap.Explanation(values=values[0], base_values=base_value, data=sample.iloc[0], feature_names=sample.columns)
plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.tight_layout(); plt.savefig(figures_dir / 'shap_waterfall_example_order.png', dpi=300, bbox_inches='tight'); plt.close()

mean_abs = pd.Series(abs(values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
mean_abs.to_csv(models_dir / 'shap_global_feature_importance.csv', header=['mean_abs_shap'])
print(mean_abs.head(10).to_string())
print(f'SHAP figures saved to: {figures_dir}')
