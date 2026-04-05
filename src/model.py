from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),

        "isolation_forest": IsolationForest(
            n_estimators=200,
            contamination=0.002,  # closer to real fraud %
            random_state=42
        )
    }