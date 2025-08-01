import os
import sys
import joblib
import json
from preprocess import load_and_preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, '..', 'data', 'train.csv')
test_path = os.path.join(BASE_DIR, '..', 'data', 'test.csv')
sys.path.append(os.path.join(BASE_DIR, '..', 'src'))

X_train, X_test, y_train, preprocessor, test_ids = load_and_preprocess_data(train_path, test_path)

metrics = {}

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

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
model_path = os.path.join(BASE_DIR, '..', 'models', 'linear_regression.pkl')
joblib.dump(model, model_path)

# Сохраняем метрики
metrics_path = os.path.join(BASE_DIR, '..', 'models', 'linear_regression_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"\nМодель сохранена: {model_path}")
print(f"Метрики сохранены: {metrics_path}")