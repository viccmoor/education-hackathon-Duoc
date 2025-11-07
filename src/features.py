import pandas as pd
import numpy as np

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _norm01(s: pd.Series) -> pd.Series:
    s = _to_num(s)
    valid = s.dropna()
    if len(valid) > 0 and valid.gt(1).mean() > 0.1:
        s = s / 100.0
    return s

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features adicionales y maneja valores nulos.
    Retorna solo columnas numéricas listas para entrenamiento.
    """
    features = df.copy()

    if "año" in features.columns:
        features["año"] = pd.to_numeric(features["año"], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "promedio_asistencia", "porcentaje_aprobacion", "promedio_notas",
        "tasa_2020", "estudiantes_retirados", "porcentaje_retiro", "total_estudiantes"
    ]

    for col in numeric_cols:
        if col in features.columns:
            if col in ["porcentaje_aprobacion", "porcentaje_retiro", "tasa_2020"]:
                features[col] = _norm01(features[col])
            else:
                features[col] = _to_num(features[col])

            # Indicador de missing
            features[col + "_missing"] = features[col].isna().astype(int)

            # Imputar con mediana si hay valores válidos, sino con 0
            valid_vals = features[col].dropna()
            fill_val = valid_vals.median() if len(valid_vals) > 0 else 0.0
            features[col] = features[col].fillna(fill_val)

    # Derivadas
    if "promedio_asistencia" in features.columns:
        features["tasa_asistencia"] = _norm01(features["promedio_asistencia"])

    if "estudiantes_retirados" in features.columns and "total_estudiantes" in features.columns:
        denom = features["total_estudiantes"].replace(0, np.nan).astype(float)
        num = features["estudiantes_retirados"].astype(float)
        features["tasa_retiro"] = (num / denom).fillna(0.0)

    if "porcentaje_aprobacion" in features.columns:
        features["aprobacion_binaria"] = (features["porcentaje_aprobacion"] > 0.85).astype(int)

    # Solo columnas numéricas
    numeric_only = features.select_dtypes(include=[np.number]).columns.tolist()
    return features[numeric_only]

FEATURE_COLS_EDU = ["prom_gral", "asistencia_pct"]

def engineer_features_edu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features:
      - prom_gral: nota [1.0, 7.0]
      - asistencia_pct: asistencia en [0,1] (acepta 0-100 o 0-1)
      - agno: alias Int64 de AGNO
    """
    df = df.copy()

    df["prom_gral"] = pd.to_numeric(df.get("PROM_GRAL"), errors="coerce").clip(lower=1.0, upper=7.0)

    asis = pd.to_numeric(df.get("ASISTENCIA"), errors="coerce")
    frac_mask = asis.between(0, 1, inclusive="both")
    asis = np.where(frac_mask, asis, asis / 100.0)
    df["asistencia_pct"] = pd.Series(asis, index=df.index).clip(lower=0.0, upper=1.0)

    df["AGNO"] = pd.to_numeric(df.get("AGNO"), errors="coerce").astype("Int64")
    df["agno"] = df["AGNO"]

    return df
