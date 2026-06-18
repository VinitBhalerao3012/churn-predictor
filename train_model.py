import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_and_preprocess():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Fix TotalCharges — convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Encode all object columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    return df

def train():
    print("Loading data...")
    df = load_and_preprocess()
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Churn rate: {y.mean():.1%}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))
    
    # Save model and feature names
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("\nModel saved to models/churn_model.pkl ✅")

if __name__ == "__main__":
    train()