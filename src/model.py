import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

SEED = 42

class DesercionPredictor:
    """
    Modelo de predicción de riesgo de deserción.
    Usa Logistic Regression con normalización estándar e imputación robusta.
    """
    def __init__(self, model=None, scaler=None, imputer=None, feature_names=None):
        self.model = model or LogisticRegression(random_state=SEED, max_iter=1000)
        self.scaler = scaler or StandardScaler()
        self.imputer = imputer or SimpleImputer(strategy='median')
        self.feature_names = feature_names or []
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Selecciona solo columnas numéricas y ordena según entrenamiento."""
        if not self.feature_names:
            # Primera vez: guardar nombres de columnas numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_names = [c for c in numeric_cols if not c.endswith('_missing')]
        
        # Asegurar que existen todas las columnas
        missing = [c for c in self.feature_names if c not in X.columns]
        for col in missing:
            X[col] = 0.0
        
        return X[self.feature_names].values
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Entrena el modelo con imputación y escalado."""
        X_arr = self._prepare_features(X)
        X_imputed = self.imputer.fit_transform(X_arr)
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.model.fit(X_scaled, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve probabilidades de clase positiva (riesgo alto)."""
        X_arr = self._prepare_features(X)
        X_imputed = self.imputer.transform(X_arr)
        X_scaled = self.scaler.transform(X_imputed)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # Probabilidad de clase 1
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve clases predichas."""
        X_arr = self._prepare_features(X)
        X_imputed = self.imputer.transform(X_arr)
        X_scaled = self.scaler.transform(X_imputed)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        """Guarda modelo, scaler e imputer."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Carga modelo desde disco."""
        data = joblib.load(path)
        return cls(
            model=data['model'],
            scaler=data['scaler'],
            imputer=data.get('imputer'),
            feature_names=data.get('feature_names', [])
        )