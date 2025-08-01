import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(train_path, test_path):
 
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.drop(columns=['Id'])
    test_ids = test_df['Id']
    test_df = test_df.drop(columns=['Id'])
    print("Файл загружен. Колонки:", train_df.columns)

    print("Логарифмируем целевой признак SalePrice → LogSalePrice...")
    train_df['LogSalePrice'] = np.log1p(train_df['SalePrice'])
    print("Колонка LogSalePrice успешно создана")

    # Общий датафрейм для обработки признаков
    full_df = pd.concat([train_df.drop(columns=['SalePrice', 'LogSalePrice']), test_df])

    # Определяем числовые и категориальные признаки
    numeric_features = full_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = full_df.select_dtypes(include=['object']).columns.tolist()

    # Пайплайны
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Разделяем
    X_train = train_df.drop(columns=['SalePrice', 'LogSalePrice'])
    y_train = train_df['LogSalePrice']
    X_test = test_df

    return X_train, X_test, y_train, preprocessor, test_ids