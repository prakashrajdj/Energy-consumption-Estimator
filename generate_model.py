import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# LOAD YOUR DATASET (place smart_home_energy.csv in same folder)
df = pd.read_csv("smart_home_energy.csv")

target = "monthly_energy_kwh"

df = df.dropna().reset_index(drop=True)

X = df.drop(columns=[target])
y = df[target]

# Encode categorical
X = pd.get_dummies(X, drop_first=True)
X = X.select_dtypes(include=[np.number])

columns = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model.pkl
model_obj = {"model": model, "columns": columns}
with open("model.pkl", "wb") as f:
    pickle.dump(model_obj, f)

# Save scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("model.pkl and scaler.pkl created successfully!")
