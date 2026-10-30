import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from preprocess import preprocess_data

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "used_cars.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "car_price_rf.pkl"

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD & PREPROCESS DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

X = df.drop("price", axis=1)
y = df["price"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)

print("\n📊 MODEL PERFORMANCE")
print("-" * 30)
print(f"R² Score : {r2_score(y_test, pred):.3f}")
print(f"MAE      : ₹{mean_absolute_error(y_test, pred):,.0f}")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("\n✅ Model saved at:", MODEL_PATH)
