import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X, y):
        """
        Fit scaler on training data and apply SMOTE
        """
        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Handle imbalance ONLY on training
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)

        return X_res, y_res

    def transform(self, X):
        """
        Transform test or new data
        """
        return self.scaler.transform(X)


def load_data(path):
    return pd.read_csv(path)


def split_features_target(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y