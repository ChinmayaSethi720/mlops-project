# src/train.py (skeleton)
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def main():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    joblib.dump(model, "../models/model.pkl")

if __name__ == "__main__":
    main()