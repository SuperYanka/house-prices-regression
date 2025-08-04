import os
import sys
import joblib
import json
from preprocess import load_and_preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, '..', 'data', 'train.csv')
test_path = os.path.join(BASE_DIR, '..', 'data', 'test.csv')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

X_train, X_test, y_train, preprocessor, test_ids = load_and_preprocess_data(train_path, test_path)

metrics = {}

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

params = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)

model = grid.best_estimator_

model.fit(X_train, y_train)
y_pred = model.predict(X_train)

mae = round(mean_absolute_error(y_train, y_pred), 3)
mse = round(mean_squared_error(y_train, y_pred), 3)
r2 = round(r2_score(y_train, y_pred), 3)

metrics = {

    "mae": mae,
    "mse": mse,
    "r2": r2
}

print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")


# Сохраняем модель
model_path = os.path.join(BASE_DIR, '..', 'models', 'random_forest.pkl')
joblib.dump(model, model_path)

# Сохраняем метрики
metrics_path = os.path.join(BASE_DIR, '..', 'models', 'random_forest_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nМодель сохранена: {model_path}")
print(f"Метрики сохранены: {metrics_path}")


pipeline_rf_path = os.path.join(BASE_DIR, '..', 'models', 'final_model.pkl')
joblib.dump(model, pipeline_rf_path)
print("Pipeline сохранен как final_model.pkl")