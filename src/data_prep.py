import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path):
    """
    Загружает данные из CSV или Excel, в зависимости от расширения имени файла.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return df

def build_preprocessor(num_features, cat_features):
    """
    Возвращает sklearn ColumnTransformer для числовых и категориальных признаков.
    """
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        # в новых sklearn sparse_output вместо sparse
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_features)
    ])
    return preprocessor

def prepare_datasets(
    df,
    num_features,
    cat_features,
    target="price",
    test_size=0.2,
    random_state=42
):
    """
    Разбивает df на train/test, подготавливает и применяет preprocessor.
    Возвращает X_train_p, X_test_p, y_train, y_test, preprocessor.
    """
    X = df[num_features + cat_features]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(num_features, cat_features)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    return X_train_p, X_test_p, y_train, y_test, preprocessor
