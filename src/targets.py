from pathlib import Path
import pandas as pd
import numpy as np


def load_rendimiento_clean(data_dir: str = "datasets/csvClear") -> pd.DataFrame:
    base = Path(data_dir)
    files = sorted(base.glob("Rendimiento_*_clean.csv"))
    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos *_clean en {base}. Ejecuta src/load.py primero."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Normalizar columnas
        df.columns = [str(c).strip().upper().lstrip("\ufeff") for c in df.columns]
        # Asegurar columnas esperadas
        for c in ("AGNO","MRUN","PROM_GRAL","ASISTENCIA","SIT_FIN","SIT_FIN_R"):
            if c not in df.columns:
                df[c] = np.nan

        df["AGNO"] = pd.to_numeric(df["AGNO"], errors="coerce").astype("Int64")
        df["MRUN"] = df["MRUN"].astype(str).str.strip()
        df["PROM_GRAL"] = pd.to_numeric(df["PROM_GRAL"], errors="coerce")
        df["ASISTENCIA"] = pd.to_numeric(df["ASISTENCIA"], errors="coerce")

        for c in ("SIT_FIN","SIT_FIN_R"):
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

        df["SOURCE_FILE"] = f.name
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    # Normalizar asistencia a 0-100 y recortar a rango válido
    mask_01 = full["ASISTENCIA"].between(0, 1, inclusive="both")
    full.loc[mask_01, "ASISTENCIA"] = full.loc[mask_01, "ASISTENCIA"] * 100
    full["ASISTENCIA"] = full["ASISTENCIA"].clip(lower=0, upper=100)

    # Filtrar MRUN vacíos y deduplicar por AGNO+MRUN
    full = full[full["MRUN"].notna() & (full["MRUN"].str.len() > 0)].copy()
    full = full.sort_values(["AGNO","MRUN"]).drop_duplicates(["AGNO","MRUN"], keep="last")

    return full

def build_label_aprobacion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = pd.Series(np.nan, index=df.index, dtype="float")

    # Preferir SIT_FIN_R
    s = df["SIT_FIN_R"].fillna("").str.lower()
    aprob = s.str.contains(r"apro|promov") & ~s.str.contains(r"no\s*apro")
    reprob = s.str.contains(r"reprob|repit|retir|deser|elim|baja|aband")
    y.loc[aprob] = 1
    y.loc[reprob] = 0

    # Fallback SIT_FIN
    s2 = df["SIT_FIN"].fillna("").str.lower()
    aprob2 = s2.str.contains(r"apro|promov") & ~s2.str.contains(r"no\s*apro")
    reprob2 = s2.str.contains(r"reprob|repit|retir|deser|elim|baja|aband")
    y.loc[y.isna() & aprob2] = 1
    y.loc[y.isna() & reprob2] = 0

    # Regla por nota
    nota = pd.to_numeric(df["PROM_GRAL"], errors="coerce")
    y.loc[y.isna() & (nota >= 4.0)] = 1
    y.loc[y.isna() & (nota < 4.0)] = 0

    df["label_aprobado"] = y.astype("Int64")
    return df

def create_target(df):
    """
    Crea variable objetivo binaria de deserción.
    
    Estrategia:
    1. Intenta usar promedio_asistencia como proxy (< 75% = riesgo alto)
    2. Fallback: porcentaje_retiro o tasa_2020
    3. Si nada existe: target sintético balanceado
    
    Retorna Series binaria: 1 = riesgo alto, 0 = riesgo bajo
    """
    target_col = None
    
    # Opción 1: Usar asistencia como proxy (más datos disponibles)
    if "promedio_asistencia" in df.columns:
        asist = pd.to_numeric(df["promedio_asistencia"], errors="coerce")
        valid = asist.dropna()
        
        if len(valid) > 0:
            # Normalizar si está en 0-100
            if valid.gt(1).mean() > 0.1:
                asist = asist / 100.0
            
            # Asistencia < 75% = riesgo alto (1), >= 75% = riesgo bajo (0)
            y = (asist < 0.75).astype(int)
            print("Advertencia: No hay columna de target válida. Creando target sintético desde promedio_asistencia.")
            return y
    
    # Opción 2: Usar porcentaje_retiro
    if "porcentaje_retiro" in df.columns:
        target_col = df["porcentaje_retiro"].copy()
    elif "tasa_2020" in df.columns:
        target_col = df["tasa_2020"].copy()
    elif {"estudiantes_retirados", "total_estudiantes"}.issubset(df.columns):
        denom = pd.to_numeric(df["total_estudiantes"], errors="coerce").replace(0, np.nan).astype(float)
        num = pd.to_numeric(df["estudiantes_retirados"], errors="coerce").astype(float)
        target_col = (num / denom).fillna(0.0)
    
    if target_col is not None:
        target_col = pd.to_numeric(target_col, errors="coerce")
        
        # Normalizar a 0-1 si está en porcentaje
        valid = target_col.dropna()
        if len(valid) > 0 and valid.gt(1).mean() > 0.1:
            target_col = target_col / 100.0
        
        # Umbral en percentil 75
        valid_values = target_col.dropna()
        if len(valid_values) > 1:
            threshold = valid_values.quantile(0.75)
            y = (target_col > threshold).astype(float).fillna(0).astype(int)
            return y
    
    # Fallback: target sintético balanceado
    print("Advertencia: No hay valores válidos en la columna de target. Usando 0s.")
    return pd.Series(0, index=df.index, dtype=int)