import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("sleep_logs.csv")

# Convert time columns to hour float
df["bed_hour"] = df["bed_time"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1])/60)
df["wake_hour"] = df["wake_time"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1])/60)

# Encode target
le = LabelEncoder()
df["sleep_quality_encoded"] = le.fit_transform(df["sleep_quality"])

X = df[["sleep_duration", "bed_hour", "wake_hour", "interruptions"]]
y = df["sleep_quality_encoded"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

def predict_sleep_quality(duration, bed_hour, wake_hour, interruptions):
    X_new = [[duration, bed_hour, wake_hour, interruptions]]
    pred = model.predict(X_new)[0]
    quality = le.inverse_transform([pred])[0]
    return quality
