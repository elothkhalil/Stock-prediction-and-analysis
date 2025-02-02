from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_models(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    model_performance = {}

    for name, model in models.items():
        model.fit(X, y)  # Correctly use X and y passed as parameters
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        model_performance[name] = r2
        print(f"{name} - Mean Squared Error: {mse:.2f}, R-squared Score: {r2:.2f}")

    return models, model_performance



if __name__ == "__main__":
    from feature_engineering import add_features
    from preprocess import load_and_preprocess

    data = load_and_preprocess('data/DELL_daily_data.csv')
    data = add_features(data)
    
    # Prepare data for training
    X = data[['MA_50', 'MA_200', 'Daily_Return']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models, performance = train_models(X_train, y_train)
