import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocessing import load_data, split_features_target, Preprocessor
from src.model import get_models


def main():
    # 📁 Base directory (project root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
    preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

    print("📥 Loading data...")
    df = load_data(data_path)

    print("🔀 Splitting features and target...")
    X, y = split_features_target(df)

    print("✂️ Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("⚙️ Preprocessing training data...")
    preprocessor = Preprocessor()

    # 🔥 IMPORTANT: handle both cases (with or without resampling)
    transformed = preprocessor.fit_transform(X_train, y_train)

    if isinstance(transformed, tuple):
        X_train, y_train = transformed
    else:
        X_train = transformed

    print("⚙️ Transforming test data...")
    X_test = preprocessor.transform(X_test)

    print("\n🚀 Training Random Forest model...")
    model = get_models()["random_forest"]
    model.fit(X_train, y_train)

    print("\n📊 Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")

    # 💾 Save model & preprocessor
    print("\n💾 Saving model and preprocessor...")
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print("✅ Training complete!")
    print(f"📁 Model saved at: {model_path}")
    print(f"📁 Preprocessor saved at: {preprocessor_path}")


if __name__ == "__main__":
    main()