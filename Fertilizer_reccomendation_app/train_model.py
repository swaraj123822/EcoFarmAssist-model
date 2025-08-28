import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os


# Load dataset
df = pd.read_csv('fertilizer_recommendation_dataset.csv')

# Check if 'Remark' column exists before dropping
if 'Remark' in df.columns:
    df = df.drop(columns=['Remark'])
else:
    print("Remark column not found in dataset. Proceeding without dropping.")

# Label encode Fertilizer (since it's the output)
label_encoder_fertilizer = LabelEncoder()
df["Fertilizer"] = label_encoder_fertilizer.fit_transform(df["Fertilizer"])

# Define categorical and numerical features based on the dataset
categorical_features = ["Soil", "Crop"]
numerical_features = ["Temperature", "Moisture", "Rainfall", "PH", "Nitrogen", "Phosphorous", "Potassium", "Carbon"]

# Verify that all features exist in the dataset
missing_numerical = [col for col in numerical_features if col not in df.columns]
missing_categorical = [col for col in categorical_features if col not in df.columns]
if missing_numerical or missing_categorical:
    raise ValueError(f"Missing columns in dataset: Numerical - {missing_numerical}, Categorical - {missing_categorical}")

# Define X and y
X = df.drop(columns=["Fertilizer"])
y = df["Fertilizer"]

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # Handles categories internally
])

# Define Model Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42, max_depth=6, n_jobs=1))
])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
pipeline.fit(X_train, y_train)

# Create directory for artifacts if it doesn't exist
os.makedirs("fertilizer_artifacts", exist_ok=True)

# Save Model & Label Encoder
dump(pipeline, "fertilizer_artifacts/model.joblib")
dump(label_encoder_fertilizer, "fertilizer_artifacts/label_encoder.joblib")

print("Model and label encoder saved successfully! 🚀")

# Evaluate model performance
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



