import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\mudra_landmarks.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

mdl = pickle.load(open(r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\mudra_classifier.pkl", "rb"))
y_pred = mdl.predict(X_test)

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Total samples: {len(df)}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Classes: {df['label'].nunique()}")
print()
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(range(len(le.classes_)))))
