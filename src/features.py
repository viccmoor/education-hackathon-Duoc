import pandas as pd
import numpy as np

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