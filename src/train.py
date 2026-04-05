import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocessing import load_data, split_features_target, Preprocessor
from src.model import get_models


DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"


def train():
    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    X, y = split_features_target(df)

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = Preprocessor()

    print("⚙️ Preprocessing training data...")
    X_train, y_train = preprocessor.fit_transform(X_train, y_train)

    print("⚙️ Transforming test data...")
    X_test = preprocessor.transform(X_test)

    models = get_models()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")

        if name == "isolation_forest":
            model.fit(X_train)
            preds = model.predict(X_test)
            preds = [1 if p == -1 else 0 for p in preds]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        print(f"📊 Evaluation for {name}:")
        print(classification_report(y_test, preds))

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"ROC-AUC: {auc:.4f}")

        report = classification_report(y_test, preds, output_dict=True)
        f1 = report["1"]["f1-score"]

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

    print(f"\n🏆 Best model: {best_name} (F1: {best_score:.4f})")

    # Save model + preprocessor
    joblib.dump(best_model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    print("💾 Saved model and preprocessor!")


if __name__ == "__main__":
    train()