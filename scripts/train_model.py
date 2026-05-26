import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def train_model(csv_path, model_path, encoder_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest Classifier on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    
    # print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
        
    print(f"\nModel saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")

if __name__ == "__main__":
    CSV_PATH = r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\mudra_landmarks.csv"
    MODEL_PATH = r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\mudra_classifier.pkl"
    ENCODER_PATH = r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\label_encoder.pkl"
    
    if os.path.exists(CSV_PATH):
        train_model(CSV_PATH, MODEL_PATH, ENCODER_PATH)
    else:
        print("Error: Landmark CSV not found. Run extraction script first.")
