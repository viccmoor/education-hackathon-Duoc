import pandas as pd
import numpy as np

SAFE_FEATURES = ["prom_gral", "asistencia_pct", "AGNO"]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera SOLO features seguras:
      - prom_gral: PROM_GRAL (1..7)
      - asistencia_pct: ASISTENCIA en [0..1]
      - AGNO: para split temporal
    No incluye SIT_FIN / SIT_FIN_R ni derivados del resultado.
    """
    X = pd.DataFrame(index=df.index)

    # Año
    if "AGNO" in df.columns:
        X["AGNO"] = pd.to_numeric(df["AGNO"], errors="coerce").astype("Int64")
    else:
        X["AGNO"] = pd.NA

    # Promedio (1..7)
    prom = pd.to_numeric(df.get("PROM_GRAL"), errors="coerce")
    prom = prom.clip(lower=1.0, upper=7.0)
    X["prom_gral"] = prom

    # Asistencia → [0..1]
    asis = pd.to_numeric(df.get("ASISTENCIA"), errors="coerce")
    # normalizar si viene 0..1 → 0..100
    mask_01 = asis.between(0, 1, inclusive="both")
    asis = asis.copy()
    asis.loc[mask_01] = asis.loc[mask_01] * 100.0
    asis = asis.clip(lower=0.0, upper=100.0)
    X["asistencia_pct"] = asis / 100.0

    return X[SAFE_FEATURES]