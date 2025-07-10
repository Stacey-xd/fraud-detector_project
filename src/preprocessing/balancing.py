# src/preprocessing/balancing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek


class BaseBalancer:
    """Base class providing train-test split functionality."""
    def __init__(self, random_state=42):
        self.random_state = random_state

    def split_data(self, df):
        X = df.drop("Class", axis=1).values
        y = df["Class"].values
        return train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_state)


class SMOTEBalancer(BaseBalancer):
    """Applies SMOTE to the entire dataset."""
    def apply(self, df, save_path=None):
        X = df.drop("Class", axis=1)
        y = df["Class"]
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled["Class"] = y_resampled
        if save_path:
            df_resampled.to_csv(save_path, index=False)
        return df_resampled


class BorderlineSMOTEBalancer(BaseBalancer):
    """Applies Borderline-SMOTE only to the training part of the data."""
    def apply(self, df, save_path=None):
        X_train, _, y_train, _ = self.split_data(df)
        smote = BorderlineSMOTE(kind="borderline-1", random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        df_resampled = pd.DataFrame(X_resampled, columns=df.drop("Class", axis=1).columns)
        df_resampled["Class"] = y_resampled
        if save_path:
            df_resampled.to_csv(save_path, index=False)
        return df_resampled