import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from data_prep import load_data, prepare_datasets
from model import build_tabular_model

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained price model")
    BASE_DIR = Path(__file__).resolve().parent.parent
    default_path = BASE_DIR / "data" / "data.csv"
    p.add_argument(
        "--data_path",
        type=str,
        default=str(default_path),
        help="Path to your CSV or Excel file"
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="best_model.h5",
        help="Path to saved model"
    )
    return p.parse_args()

def main():
    args = parse_args()
    df = load_data(args.data_path)

    num_feats = [
        "min_to_metro",
        "total_area",
        "living_area",
        "floor",
        "number_of_floors",
        "construction_year",
        "ceiling_height",
        "number_of_rooms"
    ]
    cat_feats = ["region_of_moscow", "is_new", "is_apartments"]
    df = df.dropna(subset=num_feats + cat_feats + ["price"])
    df["log_price"] = np.log1p(df["price"])

    X_tr, X_te, y_tr, y_te, preprocessor = prepare_datasets(
        df,
        num_features=num_feats,
        cat_features=cat_feats,
        target="log_price",
        test_size=0.2,
        random_state=42
    )

    model = tf.keras.models.load_model(args.model_path, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    preds_log = model.predict(X_te).flatten()
    preds = np.expm1(preds_log)
    true  = np.expm1(y_te)

    results = pd.DataFrame({
        "true_price": true,
        "pred_price": preds
    })
    print(results.head(10))
    mae = np.mean(np.abs(true - preds))
    print(f"\nMAE (rub): {mae:.2f}")

    # Сохраняем для анализа
    results.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

if __name__ == "__main__":
    main()
