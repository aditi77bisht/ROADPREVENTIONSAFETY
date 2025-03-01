import pandas as pd
import numpy as np
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the updated synthetic dataset
df = pd.read_csv("uttarakhand_synthetic_roadkill_data.csv")

# Ensure necessary columns exist, if not, create them
def ensure_column(df, column_name, default_value):
    if column_name not in df.columns:
        if isinstance(default_value, list):
            df[column_name] = np.random.choice(default_value, len(df))
        else:
            df[column_name] = default_value

ensure_column(df, "road_curve", np.random.uniform(0, 10, len(df)))
ensure_column(df, "speed_limit", np.random.randint(20, 80, len(df)))
ensure_column(df, "time_of_day", ["Morning", "Afternoon", "Night"])

# Convert time_of_day to numerical values
df["time_of_day"] = df["time_of_day"].map({"Morning": 0, "Afternoon": 1, "Night": 2})

# Strictly restrict data to only the given coordinates (30.0668° N, 79.0193° E)
df = df[(df["latitude"] >= 30.0668) & (df["longitude"] <= 79.0193)]

# Define Features (X) and Target (y)
X = df[["road_type", "traffic", "wildlife", "weather", "road_curve", "speed_limit", "time_of_day"]]
y = df["risk"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict risk
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Corrected Confusion Matrix Plot
plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No Risk", "High Risk"], yticklabels=["No Risk", "High Risk"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict risk for all locations
df["predicted_risk"] = model.predict(X)

# Get Place Names Using Geopy for All Locations
geolocator = Nominatim(user_agent="roadkill_predictor")
def get_location_name(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True)
        address = location.raw['address']
        return address.get('city', address.get('town', address.get('village', address.get('hamlet', "Unknown"))))
    except:
        return "Unknown"

# Apply for all locations
df["place_name"] = df.apply(lambda row: get_location_name(row["latitude"], row["longitude"]), axis=1)

# Create an Interactive Map
map_center = [30.0668, 79.0193]
m = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    color = "red" if row["predicted_risk"] == 1 else "green"
    risk_label = "High Risk" if row["predicted_risk"] == 1 else "Low Risk"
    popup_text = f"Place: {row['place_name']}<br>Risk: {risk_label}<br>Road Type: {row['road_type']}<br>Traffic: {row['traffic']} vehicles/hr<br>Wildlife: {row['wildlife']}<br>Weather: {'Rainy' if row['weather'] == 1 else 'Clear'}<br>Curve: {row['road_curve']:.1f}<br>Speed Limit: {row['speed_limit']} km/h"
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color)
    ).add_to(m)

# Save and Display the Map
m.save("uttarakhand_roadkill_risk_map.html")
print("Map saved! Open 'uttarakhand_roadkill_risk_map.html' to view risk areas.")