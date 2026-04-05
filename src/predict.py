import joblib
import numpy as np
import pandas as pd
import os

# 📁 Robust paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")


# 🔄 Load once
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict(transaction, threshold=0.5):
    """
    transaction: list of feature values
    threshold: probability cutoff for fraud classification
    """
    data = np.array(transaction).reshape(1, -1)
    data = preprocessor.transform(data)

    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(data)[0][1]
        pred = int(prob > threshold)
    else:
        pred = model.predict(data)[0]

    return {
        "fraud": bool(pred),
        "probability": float(prob) if prob is not None else None
    }


def predict_from_csv(file_path, threshold=0.5):
    df = pd.read_csv(file_path)

    data = preprocessor.transform(df)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[:, 1]
        preds = (probs > threshold).astype(int)
    else:
        preds = model.predict(data)
        probs = [None] * len(preds)

    results = []
    for i, (p, pr) in enumerate(zip(preds, probs)):
        results.append({
            "transaction": i + 1,
            "fraud": bool(p),
            "probability": float(pr) if pr is not None else None
        })

    return results


if __name__ == "__main__":
    # 📄 Example CSV prediction
    sample_path = os.path.join(BASE_DIR, "data", "sample.csv")

    print("🔍 Running predictions on sample.csv...\n")
    results = predict_from_csv(sample_path, threshold=0.7)

    for r in results:
        status = "🚨 FRAUD" if r["fraud"] else "✅ NORMAL"
        prob = f"{r['probability']:.4f}" if r["probability"] else "N/A"
        print(f"Transaction {r['transaction']}: {status} (Confidence: {prob})")