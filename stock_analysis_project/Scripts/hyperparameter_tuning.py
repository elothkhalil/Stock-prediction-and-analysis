from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    return grid_search

if __name__ == "__main__":
    from model_training import X_train, y_train
    best_model = tune_model(X_train, y_train)
    print(f"Best parameters: {best_model.best_params_}")
    print(f"Best R-squared score: {best_model.best_score_:.2f}")
