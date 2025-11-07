import pandas as pd
import numpy as np

FEATURE_COLS_EDU = ["prom_gral", "asistencia_pct", "AGNO"]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera features a partir del DataFrame crudo.
    Retorna DataFrame con columnas: prom_gral, asistencia_pct, AGNO
    """
    out = pd.DataFrame(index=df.index)
    
    # PROM_GRAL -> prom_gral (normalizado 1-7)
    if "PROM_GRAL" in df.columns:
        prom = pd.to_numeric(df["PROM_GRAL"], errors="coerce")
    elif "prom_gral" in df.columns:
        prom = pd.to_numeric(df["prom_gral"], errors="coerce")
    elif "promedio" in df.columns:
        prom = pd.to_numeric(df["promedio"], errors="coerce")
    else:
        prom = pd.Series(np.nan, index=df.index)
    
    out["prom_gral"] = prom.clip(lower=1.0, upper=7.0).astype(float)
    
    # ASISTENCIA -> asistencia_pct (normalizado 0-1)
    if "ASISTENCIA" in df.columns:
        asist = pd.to_numeric(df["ASISTENCIA"], errors="coerce")
    elif "asistencia_pct" in df.columns:
        asist = pd.to_numeric(df["asistencia_pct"], errors="coerce")
    elif "asistencia" in df.columns:
        asist = pd.to_numeric(df["asistencia"], errors="coerce")
    else:
        asist = pd.Series(np.nan, index=df.index)
    
    # Si está en 0-100, convertir a 0-1
    mask_pct = asist.notna() & (asist > 1.0)
    asist.loc[mask_pct] = asist.loc[mask_pct] / 100.0
    out["asistencia_pct"] = asist.clip(lower=0.0, upper=1.0).astype(float)
    
    # AGNO -> año (convertir a float para sklearn, reemplazando pd.NA con np.nan)
    if "AGNO" in df.columns:
        agno = pd.to_numeric(df["AGNO"], errors="coerce")
    elif "año" in df.columns:
        agno = pd.to_numeric(df["año"], errors="coerce")
    elif "YEAR" in df.columns:
        agno = pd.to_numeric(df["YEAR"], errors="coerce")
    else:
        agno = pd.Series(np.nan, index=df.index)
    
    # Convertir Int64 (nullable) a float64 para evitar pd.NA
    out["AGNO"] = agno.astype(float)
    
    return out