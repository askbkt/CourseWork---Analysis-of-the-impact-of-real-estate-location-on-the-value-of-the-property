import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    df_cleaned = pd.read_csv(os.path.join(ROOT, "cleaned_data.csv"))
    df_data = pd.read_csv(os.path.join(ROOT, "data", "data.csv"))
    df = pd.concat([df_cleaned, df_data], axis=1)

    df["building_age"] = 2025 - df["construction_year"]
    df["floor_ratio"] = df["floor"] / df["number_of_floors"].replace(0, np.nan)
    df["age_x_area"] = df["building_age"] * df["total_area"]

    df["floor_ratio"] = df["floor_ratio"].fillna(0)

    features = [
        'pred_price', 'error', 'min_to_metro', 'total_area', 'living_area',
        'floor', 'number_of_floors', 'ceiling_height', 'number_of_rooms',
        'building_age', 'floor_ratio', 'age_x_area',
        'is_new', 'is_apartments'
    ]

    X = df[features].copy()
    y = df["true_price"]

    y_log = np.log1p(y)
    y_log = np.where(np.isinf(y_log), np.nan, y_log)
    y_log = np.nan_to_num(y_log, nan=np.nanmean(y_log))

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_path = os.path.join(ROOT, "goat_models", "goat_model_upd_cv2.pkl")
    model: XGBRegressor = joblib.load(model_path)

    old_params = model.get_params()
    model.set_params(
        n_estimators=old_params['n_estimators'] + 1500,
        learning_rate=0.005,
        max_depth=8,
        min_child_weight=2,
        reg_alpha=0.2,
        reg_lambda=1.5
    )

    model.fit(
        X_train_scaled, y_train,
        xgb_model=model.get_booster(),
        verbose=100
    )

    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae_log = mean_absolute_error(y_test, y_pred_log)
    mae_real = mean_absolute_error(y_true, y_pred)
    mse_real = mean_squared_error(y_true, y_pred)

    print(f"MAE после upd7 (логарифм цены): {mae_log:.6f}")
    print(f"MAE после upd7 (реальная цена): {mae_real:.2f} ₽")
    print(f"MSE после upd7 (реальная цена): {mse_real:.2f}")

    out_path = os.path.join(ROOT, "goat_models", "goat_model_upd7.pkl")
    joblib.dump(model, out_path)
    print(f"✅ Модель сохранена: {out_path}")

if __name__ == "__main__":
    main()
