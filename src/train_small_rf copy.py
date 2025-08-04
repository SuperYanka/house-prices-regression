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

# Only 4 features for small model
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

X_train, X_test, y_train, preprocessor, test_ids = load_and_preprocess_data(train_path, test_path, selected_features)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=2))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_train)

mae = round(mean_absolute_error(y_train, y_pred), 3)
mse = round(mean_squared_error(y_train, y_pred), 3)
r2 = round(r2_score(y_train, y_pred), 3)

metrics = {    "mae": mae,    "mse": mse,    "r2": r2}
print(f"MAE: {mae}, MSE: {mse}, R2: {r2}")


print("Модель ожидает признаков:", model.named_steps['preprocessor'].get_feature_names_out())

output_path = os.path.join(BASE_DIR, '..', 'models', 'rf_small.pkl')
joblib.dump(model, output_path)
app_model_path = os.path.join(BASE_DIR, '..', 'deployment', 'app', 'rf_small.pkl')
joblib.dump(model, app_model_path)


# Сохраняем список признаков (для FastAPI)
with open(os.path.join(BASE_DIR, '..', 'models', 'features_small.json'), 'w') as f:
    json.dump(selected_features, f)

with open(os.path.join(BASE_DIR, '..', 'models', 'rf_small_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)