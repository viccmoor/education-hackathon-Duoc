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