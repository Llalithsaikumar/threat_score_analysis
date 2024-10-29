import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_dataset(dataset_path):
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.txt'):
        df = pd.read_csv(dataset_path, delimiter=",")
    else:
        raise ValueError("Unsupported file format. Please use .csv or .txt.")
    return df

dataset_Path = 'testdata.txt' # DataSet Path
df = load_dataset(dataset_Path)

if 'threat_score' not in df.columns:
    raise ValueError("The dataset must contain a 'threat_score' column.")

X = df.drop('threat_score', axis=1)
y = df['threat_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

def predict_threat_score(new_data):
    new_data_scaled = scaler.transform(new_data)
    threat_score = model.predict(new_data_scaled)
    risk_level = ["High Risk" if score > 50 else "Low Risk" for score in threat_score]
    return list(zip(threat_score, risk_level))

test_results = predict_threat_score(pd.DataFrame(X_test, columns=X.columns))
for score, risk in test_results:
    print(f"Threat Score: {score}, Risk Level: {risk}")
