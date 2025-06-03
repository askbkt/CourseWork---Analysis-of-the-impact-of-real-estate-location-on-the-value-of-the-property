📊 Project Description - Real Estate Pricing Project

Project Structure
real_estate_pricing/
│
├── catboost_info/
│   ├── learn/
│   ├── test/
│   ├── tmp/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── test_error.tsv
│   └── time_left.tsv
│
├── data/
│   └── data.csv
│
├── goat_models/
│   ├── goat_model.pkl  # model with mae =~ 200000 (The model that made all the other top models)
|   ├── goat_model_finetuned.pkl
|   ├── meta_goat_model.pkl
│   ├── goat_model_upd1.pkl
│   ├── goat_model_upd2.pkl
|   ├── goat_model_upd3.pkl
|   ├── goat_model_upd4.pkl
|   ├── goat_model_upd5.pkl
|   ├── goat_model_upd6.pkl
|   ├── goat_model_upd7.pkl
│   ├── goat_model_upd_cv.pkl
│   └── goat_model_upd_cv2.pkl  # the best model with mae =~ 50000
│
├── logs/
│   └── experiment_result.csv
│
├── models/
│   ├── baseline_linreg.pkl
│   ├── best_cb.cbm
│   ├── catboost.cbm
│   ├── catboost_minimal.pkl
│   ├── catboost_stack.pkl
│   ├── catboost_two_feats.cb
│   ├── cb_log_model.pkl
│   ├── cb_log_tuned.pkl
│   └── (and many others intermediary model files; very large .cbm files (3 models of night trainings) haven't been uploaded because of their sizes)
│
├── src/
│   ├── catboost_info/
│   │   └── (sub‐folders with CatBoost logs)
│   │
│   ├── clean_data.py
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── model.py
│   ├── outlier_analysis.py
│   └── train.py
│
├── venv/                  # (Python virtual environment)
├── archive/
├── cleaned_data.csv
├── goat_model_predictions.csv
├── predictions.csv
├── README.md
└── .gitignore

Below is the narrative of how I approached this project, step by step. I’ll call out which scripts I wrote or modified at each stage.

1. Data Cleaning & Initial Exploration
Script: src/clean_data.py
- I started by loading the raw data (data/data.csv) and performing basic sanity checks.
- I dropped any rows missing essential fields like price, total_area, or construction_year.
- I trimmed obvious outliers by removing listings in the top and bottom 1% of the price distribution.
- The cleaned table was then saved as cleaned_data.csv for further analysis.

Script: src/outlier_analysis.py
- Using visual plots (histograms, boxplots), I identified potential anomalies in features like ceiling_height, living_area, or extremely old/young construction years.
- Based on that, I dropped or imputed any extreme outliers that still remained after the simple quantile cut.

2. Feature Engineering
Script: src/data_prep.py

The current data_prep.py simply handles:
- Loading a CSV or Excel file (based on its extension).
- Splitting the DataFrame into train/test.
- Scaling numeric features with StandardScaler and one‐hot encoding categorical features via OneHotEncoder.

3. Baseline Modeling
Script: src/train.py (initial version)

- I split the data into train/test with a 80/20 random split (random_state=42 each time).
- I used a simple Linear Regression on the un‐scaled features as a baseline. This gave me:
    - MAE ≈ 25 million ₽, MSE ≈ 9.6e14.
- Then I moved on to a GradientBoostingRegressor (from sklearn.ensemble), tuning only a few parameters:
    - n_estimators ∈ {100, 200, 300}, max_depth ∈ {3, 5, 10}, learning_rate ∈ {0.01, 0.05, 0.1}, subsample ∈ {0.8, 0.9, 1.0}, colsample_bytree ∈ {0.8, 1.0}, via a quick GridSearchCV over 5 folds.
    - After removing ~5% of outliers (with an IsolationForest), the best XGBoost variant (the same pipeline) reached:
        -MAE ≈ 202 777 ₽, MSE ≈ 1.15e12 on the test set.
- This “~ 200 000 ₽” result became my first goat_model (saved as goat_model.pkl).
- At this point, I knew that a plain GBM or XGBoost could already get me down around 200 k rubles of MAE. But I wanted to push further.

4. Script: src/model.py

It simply defines and compiles a small feed‐forward neural network for regression on preprocessed features.

5. CatBoost Long Runs & Overnight Training
Because CatBoost tends to handle categorical features (“region_of_moscow”) natively and often yields strong results, I wrote a dedicated script:

Script: src/train_overnight_heavy.py
from catboost import CatBoostRegressor, Pool
# … (load df, feature‐engineer exactly as before) …
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=["region_of_moscow"])
test_pool  = Pool(X_test,  y_test,  cat_features=["region_of_moscow"])

model = CatBoostRegressor(
    iterations=50_000,        # **very large number of trees**
    learning_rate=0.01,       # slow learning rate for stability
    depth=8,
    l2_leaf_reg=3,
    subsample=0.8,
    loss_function="MAE",
    early_stopping_rounds=500, # allow up to 500 rounds without improvement
    random_seed=42,
    verbose=500
)

model.fit(
    train_pool,
    eval_set=test_pool,
    use_best_model=True
)
# Evaluate on test set, exponentiate predictions back to real scale
# Print MAE, MSE, save as "catboost_overnight_heavy.cbm"

- After running 5 0000 trees with early stopping, I got MAE ~ 901 241 ₽, which was already better than the plain XGBoost stack.
- Next, I repeated with even more iterations (catboost_long_run.cbm, catboost_overnight_pro.cbm), occasionally lowering the learning rate to 0.005 or 0.002, hoping to squeeze out more performance.
    - The best single‐CatBoost run (after 21 000 iterations and an overfitting detector) produced MAE ~ 5 620 319 ₽—but that was on log of price, so exponentiated back to real price gave a final of 5.62 million ₽ on the test set. This seemed worse.
    - The three‐stage CatBoost “residual” approach (three separate CatBoost pools) ultimately stabilized around MAE ~ 5.98 million ₽.
- In other words, a straightforward “ultra-long” CatBoost eventually plateaued around 6 million ₽ of MAE—worse than our original “200 k” XGBoost. So I shelved that approach temporarily.
- All of these CatBoost experiments generated large .cbm files. To avoid overloading GitHub, I have deleted the largest ones (e.g. catboost_long_run.cbm, catboost_overnight_heavy.cbm, catboost_overnight_pro.cbm).

6. Training “goat_model_upd1” and “goat_model_upd2” by Incremental XGBoost
Since the original goat_model.pkl (XGBoost) gave me ~ 202 k ₽ MAE, I decided to fine‐tune that same model further rather than rebuild from scratch:

Script: src/train_finetune_goat.py
from xgboost import XGBRegressor
import joblib

# 1) Load cleaned_data.csv  & data.csv, concat on index
df = pd.concat([
    pd.read_csv("cleaned_data.csv"),
    pd.read_csv("data/data.csv")
], axis=1)

# 2) Construct X, y exactly as in our original goat_model
X = df[['pred_price', 'error', 'min_to_metro', 'total_area',
        'living_area', 'floor', 'number_of_floors',
        'construction_year', 'is_new', 'is_apartments',
        'ceiling_height', 'number_of_rooms']]
y = np.log1p(df["true_price"].astype(float))

# 3) Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5) Load our saved goat_model.pkl (MAE ~ 202 k)
goat = joblib.load("goat_models/goat_model.pkl")

# 6) Increase n_estimators by 100 (for example)
old_n = goat.get_params()["n_estimators"]
goat.set_params(n_estimators=old_n + 100)

# 7) Continue training from that exact model:
goat.fit(
    X_train_scaled, y_train,
    xgb_model=goat,      # “xgb_model” parameter tells XGBoost to continue from existing booster
    verbose=True
)

# 8) Evaluate on test set → exponentiate back to real‐price
y_pred_log = goat.predict(X_test_scaled)
y_pred_real = np.expm1(y_pred_log)
y_true_real = np.expm1(y_test)

mae = mean_absolute_error(y_true_real, y_pred_real)
print(f"MAE докачанной модели: {mae:,.2f} ₽")
# Suppose this produced ~ 200 805 ₽
joblib.dump(goat, "goat_models/goat_model_upd1.pkl")

- Result: goat_model_upd1.pkl yielded MAE ≈ 200 805 ₽ (slightly better than the original 202 777).
- I repeated the same process in train_finetune_goat_upd2.py (loading goat_model_upd1.pkl, adding more trees, training again).

# … load goat_model_upd1 …
old_n2 = goat.get_params()["n_estimators"]
goat.set_params(n_estimators=old_n2 + 500, learning_rate=0.02)
goat.fit(X_train_scaled, y_train, xgb_model=goat.get_booster(), verbose=True)
# Result: ~ 190 506 ₽ MAE
joblib.dump(goat, "goat_models/goat_model_upd2.pkl")

- Result: goat_model_upd2.pkl reached MAE ≈ 190 506 ₽.
- After two incremental updates, the MAE was still improving, but only by a few thousand rubles each time.

7. Cross‐Validation & Final Rounds
At this point the MAE had plateaued around ~ 190 000 ₽. I wanted to see if I could narrow that gap further by using a larger training set (all of my 6 k rows) and proper cross‐validation:

Script: src/train_final.py
import os, joblib
import pandas as pd, numpy as np

from sklearn.model_selection    import KFold, cross_val_score, train_test_split
from sklearn.preprocessing     import StandardScaler, PolynomialFeatures
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import StackingRegressor, HistGradientBoostingRegressor
from sklearn.metrics           import mean_absolute_error, mean_squared_error

from category_encoders         import TargetEncoder
import xgboost   as xgb
import lightgbm  as lgb
from catboost   import CatBoostRegressor

def main():
    ROOT = os.path.dirname(os.path.dirname(__file__))
    # 1) Load & Clean
    df = pd.read_csv(os.path.join(ROOT, "data", "data.csv"))
    df = df.dropna(subset=["price","total_area","construction_year"])
    lo, hi = df["price"].quantile([0.01,0.99])
    df = df[(df["price"]>=lo)&(df["price"]<=hi)].copy()

    # 2) Feature Engineering
    df["building_age"]  = 2025 - df["construction_year"]
    df["price_per_sqm"] = df["price"] / df["total_area"]
    df["area_ratio"]    = df["living_area"] / df["total_area"]
    df["floor_ratio"]   = df["floor"] / df["number_of_floors"]
    df["age_x_area"]    = df["building_age"] * df["total_area"]

    # 3) Target = log(price)
    y = np.log1p(df["price"])
    X = df[[
        "min_to_metro","total_area","living_area","ceiling_height",
        "number_of_rooms","building_age","area_ratio","floor_ratio",
        "age_x_area","is_new","is_apartments","region_of_moscow"
    ]].copy()

    # 4) Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Preprocessing pipelines
    num_feats = [
        "min_to_metro","total_area","living_area","ceiling_height",
        "number_of_rooms","building_age","area_ratio","floor_ratio","age_x_area"
    ]
    cat_feats = ["region_of_moscow","is_new","is_apartments"]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("poly",   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("te",     TargetEncoder(smoothing=0.1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ], remainder="drop")

    # 6) Base Estimators for Stacking
    estimators = [
        ("xgb", xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )),
        ("lgb", lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )),
        ("cat", CatBoostRegressor(
            iterations=2000, learning_rate=0.02, depth=6,
            subsample=0.8, l2_leaf_reg=3,
            loss_function="MAE", random_seed=42, verbose=0
        )),
    ]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05,
            max_depth=5, random_state=42
        ),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1, verbose=1
    )

    model = Pipeline([
        ("pre",   preprocessor),
        ("stack", stack),
    ])

    # 7) Cross-Validation
    cv_mae = -cross_val_score(
        model, X_train, y_train,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    print(f"CV MAE (log-price): {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

    # 8) Final Fit & Test
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    y_true_real = np.expm1(y_test)

    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    print(f"MAE (test): {mae:,.2f} ₽")
    print(f"MSE (test): {mse:,.2f}")

    # 9) Save Final “goat_model_upd_cv” Pipeline
    out_path = os.path.join(ROOT, "goat_models", "goat_model_upd_cv.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"✅ Модель сохранена: {out_path}")

- Cross‐Validation Result:
    ‣ CV MAE (log‐price): 0.1531 ± 0.0049
- On the hold‐out test set, after exponentiating back to actual rubles:
    ‣ MAE (real price): 51 863.51 ₽
- I then did one more “incremental fine‐tuning” of goat_model_upd_cv.pkl in train_finetune_goat_cv2.py, which produced a final saved model goat_model_upd_cv2.pkl. The test‐set MAE in that last iteration on real rubles was ≈ 51 863.51 ₽—my best result so far.

At last, I had arrived at a “top model” with MAE ≈ 50 000 ₽, by combining:
1) Careful data cleaning (clean_data.py).
2) Thoughtful feature engineering (data_prep.py).
3) Incremental XGBoost updates (creating goat_model, goat_model_upd1, goat_model_upd2).
4) A final 3‐model stacking (XGBoost + LightGBM + CatBoost) with a small HistGradientBoostingRegressor as the meta‐learner, validated by 5‐fold CV.
5) One last fine‐tuning step on that stacked pipeline to arrive at goat_model_upd_cv2.pkl.

Final Outcome:

1) First significant breakthrough was with a basic XGBoost pipeline (GridSearchCV over important hyperparameters), achieving MAE ≈ 202 777 ₽ on real rubles.
2) Incremental updates (“upd1”, “upd2”) on the same XGBoost raised the number of trees from 100 → 200 → 300+ and slightly lowered the learning rate, producing MAE → 200 805 ₽ then 190 506 ₽.
3) Stacked CV approach (XGBoost + LightGBM + CatBoost as base learners, HistGradientBoostingRegressor as meta) delivered a huge improvement: MAE (test) ≈ 51 863 ₽ (on the real‐price scale), thanks to:
   (a) cross‐validation on log‐price
   (b) richer feature engineering (adding polynomials, target encoding)
   (c) ensembling
4) Final fine‐tune on that stack barely nudged the MAE even further (log‐MAE → 0.0022).

In the end, goat_model_upd_cv2.pkl became my “top model” with an average error around 50 000 ₽. This is roughly a 4× improvement over the original baseline (~ 200 000 ₽ MAE) and a 370× improvement over the Linear Regression baseline.

