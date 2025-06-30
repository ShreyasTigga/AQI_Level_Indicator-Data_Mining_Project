# --- Step 1: Import Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 2: Load Dataset ---
df = pd.read_csv("delhi_aqi.csv")
df['date'] = pd.to_datetime(df['date'])
df.dropna(inplace=True)

# --- Step 3: Classify AQI Level ---
def classify_aqi(pm25):
    if pm25 <= 50:
        return 'Good'
    elif pm25 <= 100:
        return 'Satisfactory'
    elif pm25 <= 200:
        return 'Moderate'
    elif pm25 <= 300:
        return 'Poor'
    elif pm25 <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

df['AQI_Level'] = df['pm2_5'].apply(classify_aqi)

# --- Step 4: Prepare Features and Labels ---
features = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']
X = df[features]
y = df['AQI_Level']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# --- Step 5: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --- Step 6: Train Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Step 7: Evaluate ---
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Step 8: Feature Importance ---
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importances from Random Forest")
plt.tight_layout()
plt.show()
