import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(model_performance):
    model_names = list(model_performance.keys())
    r2_scores = list(model_performance.values())

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, r2_scores, color=['blue', 'green', 'orange'])
    plt.xlabel('Models')
    plt.ylabel('R-squared Score')
    plt.title('Model Comparison: R-squared Scores')
    plt.ylim(0, 1)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()
    print("Model comparison visualization complete.")

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(feature_names)), importances[indices], align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()
        print("Feature importance visualization complete.")
    else:
        print("The model does not support feature importance.")

def plot_predictions_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.show()
    print("Predictions vs. actual visualization complete.")

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals of Predictions')
    plt.show()
    print("Residuals visualization complete.")
