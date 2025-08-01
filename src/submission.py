import os
import pandas as pd
import numpy as np
import joblib
from preprocess import load_and_preprocess_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(BASE_DIR, '..', 'data', 'train.csv')
test_path = os.path.join(BASE_DIR, '..', 'data', 'test.csv')

# Пути к моделям
l_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'lasso.pkl'))
lr_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'linear_regression.pkl'))
rf_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'random_forest.pkl'))
r_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'ridge.pkl'))

# Пути для сабмишнов
l_submission_path = os.path.join(BASE_DIR, '..', 'submissions', 'lasso_submission.csv')
lr_submission_path = os.path.join(BASE_DIR, '..', 'submissions', 'linear_regression_submission.csv')
rf_submission_path = os.path.join(BASE_DIR, '..', 'submissions', 'random_forest_submission.csv')
r_submission_path = os.path.join(BASE_DIR, '..', 'submissions', 'ridge_submission.csv')

# Загрузка и подготовка
_, X_test, y_train, _, test_ids = load_and_preprocess_data(train_path, test_path)

# Предсказания
l_final_preds = np.expm1(l_model.predict(X_test))
lr_final_preds = np.expm1(lr_model.predict(X_test))
rf_final_preds = np.expm1(rf_model.predict(X_test))
r_final_preds = np.expm1(r_model.predict(X_test))

# Сохраняем отдельно каждую сабмишн-таблицу
pd.DataFrame({'Id': test_ids, 'SalePrice': l_final_preds}).to_csv(l_submission_path, index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': lr_final_preds}).to_csv(lr_submission_path, index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': rf_final_preds}).to_csv(rf_submission_path, index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': r_final_preds}).to_csv(r_submission_path, index=False)

print("Все сабмишны успешно сохранены:")
print(f" - Lasso:           {l_submission_path}")
print(f" - LinearRegression:{lr_submission_path}")
print(f" - RandomForest:    {rf_submission_path}")
print(f" - Ridge:           {r_submission_path}")
