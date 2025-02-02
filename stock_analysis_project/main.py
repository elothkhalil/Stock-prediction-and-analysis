from Scripts.preprocess import load_and_preprocess
from Scripts.feature_engineering import add_features
from Scripts.model_training import train_models
from Scripts.hyperparameter_tuning import tune_model
from Scripts.visualization_tools import (
    plot_model_comparison,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals
)
from sklearn.model_selection import train_test_split
import joblib

if __name__ == "__main__":
    # Load and preprocess data
    data = load_and_preprocess('data/DELL_daily_data.csv')
    data = add_features(data)

    # Prepare data for training
    X = data[['MA_50', 'MA_200', 'Daily_Return']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and tune the best one
    trained_models, model_performance = train_models(X_train, y_train)
    best_model = tune_model(X_train, y_train)

    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')
    print("Best model saved successfully.")

    # Visualize R-squared scores
    plot_model_comparison(model_performance)

    # Visualize feature importance for the best model
    plot_feature_importance(best_model, ['MA_50', 'MA_200', 'Daily_Return'])

    # Make predictions and visualize predictions vs. actual and residuals
    y_pred = best_model.predict(X_test)
    plot_predictions_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
