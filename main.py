import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Please install it with 'pip install xgboost'")

# Set paths
DATA_PATH = "data_core.csv"
PLOTS_DIR = "plots"
MODELS_DIR = "models"

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 🔹 STEP 1: Load & Understand Data
print("\n--- STEP 1: Loading Data ---")
df = pd.read_csv(DATA_PATH)

# Handling typos and column names
df.columns = df.columns.str.strip()
df.rename(columns={'Temparature': 'Temperature'}, inplace=True)

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nData Types:")
print(df.dtypes)

# Handle missing data
print("\nChecking for missing values:")
print(df.isnull().sum())

# Simple imputation (if any)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# 🔹 STEP 2: Feature Engineering
print("\n--- STEP 2: Feature Engineering (Creating Soil_Health) ---")

def calculate_soil_health(row):
    n, p, k = row['Nitrogen'], row['Phosphorous'], row['Potassium']
    
    # Logic:
    # 1. Good: High Nitrogen and decent Phosphorus/Potassium
    # 2. Moderate: Decent Nitrogen
    # 3. Poor: Low results
    # As specified in requirements:
    if n > 50 and p > 40 and k > 40:
        return "Good"
    elif 25 <= n <= 50:
        return "Moderate"
    else:
        return "Poor"

df['Soil_Health'] = df.apply(calculate_soil_health, axis=1)

print("Soil Health Distribution:")
print(df['Soil_Health'].value_counts())

# Save enhanced dataset for reference
df.to_csv("data_enhanced.csv", index=False)
print("\nEnhanced dataset saved as 'data_enhanced.csv'")

# 🔹 STEP 3: Exploratory Data Analysis (EDA)
print("\n--- STEP 3: EDA ---")

# Plot distributions of N, P, K
plt.figure(figsize=(15, 5))
for i, col in enumerate(['Nitrogen', 'Phosphorous', 'Potassium']):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'npk_distribution.png'))
plt.close()

# Correlation Heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
plt.close()

# Soil_Health vs Features
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Soil_Health', palette='viridis')
plt.title('Count of Soil Health categories')
plt.savefig(os.path.join(PLOTS_DIR, 'soil_health_counts.png'))
plt.close()

# Categorical Plots
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Soil Type', hue='Soil_Health')
plt.title('Soil Type vs Soil Health')
plt.xticks(rotation=45)
plt.savefig(os.path.join(PLOTS_DIR, 'soil_type_vs_health.png'))
plt.close()

print(f"Plots saved in '{PLOTS_DIR}' directory.")

# 🔹 STEP 4: Data Preprocessing
print("\n--- STEP 4: Data Preprocessing ---")

# Encoding categorical features
encoders = {}
categorical_cols = ['Soil Type', 'Crop Type']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Target Encoders
le_health = LabelEncoder()
df['Soil_Health_Encoded'] = le_health.fit_transform(df['Soil_Health'])
encoders['Soil_Health'] = le_health

le_fert = LabelEncoder()
df['Fertilizer_Encoded'] = le_fert.fit_transform(df['Fertilizer Name'])
encoders['Fertilizer'] = le_fert

# Scaling numerical data
scaler = StandardScaler()
numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Encoding and Scaling complete.")

# 🔹 STEP 5: Feature Selection
print("\n--- STEP 5: Feature Selection ---")

# Best features for Soil Health
# We use all numeric + categorical features excluding the targets
features = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']

# Using Random Forest to check feature importance for Soil Health
rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(df[features], df['Soil_Health_Encoded'])
importances = pd.Series(rf_temp.feature_importances_, index=features).sort_values(ascending=False)

print("\nFeature Importance (Soil Health):")
print(importances)

# Save feature importance plot
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='orange')
plt.title('Feature Importance for Soil Health')
plt.ylabel('Score')
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_health.png'))
plt.close()

# 🔹 STEP 6, 7, 8: Model Building, Tuning, and Evaluation
print("\n--- STEP 6, 7, 8: Modeling & Evaluation ---")

def evaluate_models(X_train, X_test, y_train, y_test, model_type="Soil Health"):
    results = []
    
    if model_type == "Soil Health":
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier()
        }
        # GridSearch for RF
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
        gs = GridSearchCV(models['RandomForest'], param_grid, cv=3, scoring='accuracy')
        gs.fit(X_train, y_train)
        models['RandomForest_Tuned'] = gs.best_estimator_
        print(f"Best params for {model_type} RF: {gs.best_params_}")
        
    else: # Fertilizer Model
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }
        try:
            models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        except:
            pass
            
        # GridSearch for RF
        param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        gs = GridSearchCV(models['RandomForest'], param_grid, cv=3, scoring='accuracy')
        gs.fit(X_train, y_train)
        models['RandomForest_Tuned'] = gs.best_estimator_

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Estimator': model
        })
        
    return pd.DataFrame(results)

# --- Track 1: Soil Health Model ---
print("\nTraining Soil Health Models...")
X_health = df[features]
y_health = df['Soil_Health_Encoded']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_health, y_health, test_size=0.2, random_state=42)

health_results = evaluate_models(X_train_h, X_test_h, y_train_h, y_test_h, "Soil Health")
print("\nSoil Health Model Comparison:")
print(health_results.drop(columns='Estimator'))

best_h_idx = health_results['F1-Score'].idxmax()
best_health_model = health_results.loc[best_h_idx, 'Estimator']
print(f"Best Soil Health Model: {health_results.loc[best_h_idx, 'Model']}")

# --- Track 2: Fertilizer Model ---
print("\nTraining Fertilizer Models...")
# For Fertilizer, we can also include Soil_Health as a feature
X_fert = df[features + ['Soil_Health_Encoded']]
y_fert = df['Fertilizer_Encoded']
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)

fert_results = evaluate_models(X_train_f, X_test_f, y_train_f, y_test_f, "Fertilizer")
print("\nFertilizer Model Comparison:")
print(fert_results.drop(columns='Estimator'))

best_f_idx = fert_results['F1-Score'].idxmax()
best_fert_model = fert_results.loc[best_f_idx, 'Estimator']
print(f"Best Fertilizer Model: {fert_results.loc[best_f_idx, 'Model']}")

# 🔹 STEP 9: Performance Visualization
print("\n--- STEP 9: Visualizing performance ---")

def plot_confusion_matrix(model, X_test, y_test, title, filename, labels):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

# Plot CM for best models
plot_confusion_matrix(best_health_model, X_test_h, y_test_h, "Best Soil Health", "cm_health.png", le_health.classes_)
plot_confusion_matrix(best_fert_model, X_test_f, y_test_f, "Best Fertilizer", "cm_fert.png", le_fert.classes_)

# Model Comparison Charts
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=health_results, x='Model', y='Accuracy')
plt.title('Soil Health Model Accuracy')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(data=fert_results, x='Model', y='Accuracy')
plt.title('Fertilizer Model Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'))
plt.close()

# 🔹 STEP 10: Prediction System
print("\n--- STEP 10: Prediction System ---")

def predict_soil_and_fertilizer(temp, hum, moist, soil_type, crop_type, n, p, k):
    # input processing (Encoding & Scaling)
    # create a dummy dataframe to keep feature names consistent for scaler
    input_df = pd.DataFrame([[temp, hum, moist, 0, 0, n, k, p]], 
                            columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
    
    # Encode categorical inputs
    input_df['Soil Type'] = encoders['Soil Type'].transform([soil_type])[0]
    input_df['Crop Type'] = encoders['Crop Type'].transform([crop_type])[0]
    
    # Scale numerical inputs
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # 1. Predict Soil Health
    health_pred_encoded = best_health_model.predict(input_df[features])[0]
    health_label = le_health.inverse_transform([health_pred_encoded])[0]
    
    # 2. Predict Fertilizer
    # Add predicted health to input features for fertilizer model
    input_df['Soil_Health_Encoded'] = health_pred_encoded
    fert_pred_encoded = best_fert_model.predict(input_df[features + ['Soil_Health_Encoded']])[0]
    fert_label = le_fert.inverse_transform([fert_pred_encoded])[0]
    
    return health_label, fert_label

# Test the function with raw sample from dataset (using original values)
# Example: 26, 52, 38, 'Sandy', 'Maize', 37, 0, 0 -> Urea
h_res, f_res = predict_soil_and_fertilizer(26.0, 52.0, 38.0, 'Sandy', 'Maize', 37, 0, 0)
print(f"\nSample Prediction Test:")
print(f"Inputs: Temp=26, Hum=52, Moist=38, Soil='Sandy', Crop='Maize', N=37, P=0, K=0")
print(f"Output: Soil Health = {h_res}, Recommended Fertilizer = {f_res}")

# 🔹 STEP 11: Save Models
print("\n--- STEP 11: Saving Models ---")
with open(os.path.join(MODELS_DIR, 'best_health_model.pkl'), 'wb') as f:
    pickle.dump(best_health_model, f)

with open(os.path.join(MODELS_DIR, 'best_fert_model.pkl'), 'wb') as f:
    pickle.dump(best_fert_model, f)

with open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'wb') as f:
    pickle.dump(encoders, f)

with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print(f"Models and transformers saved in '{MODELS_DIR}' directory.")
print("\nWorkflow completed successfully.")



