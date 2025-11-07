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
    """
    Etiqueta binaria de aprobación:
      - Usa primero SIT_FIN_R, luego SIT_FIN (sin copiar df)
      - Fallback por nota PROM_GRAL (>= 4.0 => 1, < 4.0 => 0)
    Más eficiente en memoria/tiempo para ~millones de filas.
    """
    n = len(df)
    y = np.full(n, np.nan, dtype=np.float32)

    # Patrones (sin lower() ni fillna, usar na=False)
    aprob_pat = r"(apro|promov|^p$)"
    reprob_pat = r"(reprob|repit|retir|deser|elim|baja|aband)"

    # 1) Preferir SIT_FIN_R
    s_r = df.get("SIT_FIN_R")
    if s_r is not None:
        aprob = s_r.astype("string").str.contains(aprob_pat, regex=True, case=False, na=False).to_numpy()
        reprob = s_r.astype("string").str.contains(reprob_pat, regex=True, case=False, na=False).to_numpy()
        y[aprob] = 1
        y[reprob] = 0

    # 2) Fallback SIT_FIN
    s = df.get("SIT_FIN")
    if s is not None:
        aprob2 = s.astype("string").str.contains(aprob_pat, regex=True, case=False, na=False).to_numpy()
        reprob2 = s.astype("string").str.contains(reprob_pat, regex=True, case=False, na=False).to_numpy()
        na_mask = np.isnan(y)
        y[na_mask & aprob2] = 1
        y[na_mask & reprob2] = 0

    # 3) Fallback por nota
    nota = pd.to_numeric(df.get("PROM_GRAL"), errors="coerce").to_numpy()
    na_mask = np.isnan(y)
    y[na_mask & (nota >= 4.0)] = 1
    y[na_mask & (nota < 4.0)] = 0

    # Escribir en df (tipo compacto)
    df["label_aprobado"] = pd.Series(y, index=df.index).astype("Int8")
    return df

def create_target(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve 1 = riesgo alto (reprobación/deserción), 0 = riesgo bajo.
    Reglas:
      - Usa textos en SIT_FIN_R/SIT_FIN si existen.
      - Respaldo por nota: 1 si PROM_GRAL < 4.0, 0 si >= 4.0.
      - Respaldo por asistencia: 1 si ASISTENCIA < 60 (%).
    """
    y = pd.Series(np.nan, index=df.index, dtype="float")

    # 1) Texto de situación final (si existe)
    for col in ("SIT_FIN_R", "SIT_FIN"):
        if col in df.columns:
            s = df[col].astype(str).str.lower().fillna("")
            aprob = s.str.contains(r"apro|promov") & ~s.str.contains(r"no\s*apro")
            reprob = s.str.contains(r"reprob|repit|retir|deser|elim|baja|aband")
            y.loc[aprob & y.isna()] = 0
            y.loc[reprob & y.isna()] = 1

    # 2) Regla por nota
    if "PROM_GRAL" in df.columns:
        nota = pd.to_numeric(df["PROM_GRAL"], errors="coerce")
        y.loc[y.isna() & (nota < 4.0)] = 1
        y.loc[y.isna() & (nota >= 4.0)] = 0

    # 3) Respaldo por asistencia
    if "ASISTENCIA" in df.columns:
        asis = pd.to_numeric(df["ASISTENCIA"], errors="coerce")
        y.loc[y.isna() & (asis < 60)] = 1
        y.loc[y.isna() & (asis >= 60) & y.isna()] = 0

    # 4) Default final si aún quedan NaN
    y = y.fillna(0).astype("Int64")
    return y