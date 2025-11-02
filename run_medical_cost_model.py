import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("📥 Loading dataset...")
df = pd.read_csv("insurance.csv")
print(f"Shape: {df.shape}")
print("Columns:", list(df.columns), "\n")

X = df.drop("charges", axis=1)
y = df["charges"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first"), cat_cols),
    ("scale", StandardScaler(), num_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    print(f"🔹 Training {name}...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

results_df = pd.DataFrame(results)
print("\n📊 Model Comparison:\n", results_df)

plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="R2", data=results_df)
plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("\n✅ Saved chart as: model_comparison.png")
