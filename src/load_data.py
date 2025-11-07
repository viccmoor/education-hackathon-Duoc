from pathlib import Path
import pandas as pd
import numpy as np
import re

SEED = 42

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas."""
    colmap = {c: re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns}
    df = df.rename(columns=colmap)

    # Mapeo a esquema del proyecto
    mapping = {
        "agno": "año",
        "anio": "año",
        "year": "año",
        "periodo": "año",
        "promedio_asistencia": "promedio_asistencia",
        "asistencia_promedio": "promedio_asistencia",
        "asistencia": "promedio_asistencia",
        "porcentaje_aprobacion": "porcentaje_aprobacion",
        "aprobacion": "porcentaje_aprobacion",
        "tasa_aprobacion": "porcentaje_aprobacion",
        "promedio_notas": "promedio_notas",
        "prom_gral": "promedio_notas",
        "nota_promedio": "promedio_notas",
        "tasa_2020": "tasa_2020",
        "estudiantes_retirados": "estudiantes_retirados",
        "retirados": "estudiantes_retirados",
        "porcentaje_retiro": "porcentaje_retiro",
        "retiro_porcentaje": "porcentaje_retiro",
        "total_estudiantes": "total_estudiantes",
        "matriculados": "total_estudiantes",
        "total_matriculados": "total_estudiantes",
    }

    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    return df

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas a tipos numéricos."""
    num_cols = [
        "promedio_asistencia",
        "porcentaje_aprobacion",
        "promedio_notas",
        "tasa_2020",
        "estudiantes_retirados",
        "porcentaje_retiro",
        "total_estudiantes",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Año como entero
    if "año" in df.columns:
        df["año"] = pd.to_numeric(df["año"], errors="coerce").fillna(0).astype(int)

    return df

def _infer_year_from_path(p: Path) -> int | None:
    """Extrae año desde el nombre del archivo (ej: 2023)."""
    m = re.search(r"(20\d{2})", p.name)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

def get_clean_data(
    base_dir: str | None = None,
    sample_frac: float | None = None,
    max_rows: int | None = None
) -> pd.DataFrame | None:
    """
    Carga y concatena CSVs limpios de educación desde datasets/csvClear/.
    
    Args:
        base_dir: Directorio raíz del proyecto
        sample_frac: Fracción de filas a samplear (ej: 0.1 para 10%)
        max_rows: Máximo de filas a leer por archivo (para pruebas rápidas)
    
    Returns:
        DataFrame con columnas normalizadas o None si falla
    """
    root = Path(base_dir) if base_dir else Path.cwd()
    data_dir = root / "datasets" / "csvClear"

    if not data_dir.exists():
        print(f"❌ Directorio no existe: {data_dir}")
        return None

    files = sorted(data_dir.glob("Rendimiento_*_clean.csv"))
    if not files:
        print(f"❌ No se encontraron CSV en {data_dir}")
        return None

    print(f"📁 Encontrados {len(files)} archivos CSV")
    
    dfs = []
    for i, fp in enumerate(files, 1):
        try:
            print(f"   [{i}/{len(files)}] Cargando {fp.name}...", end=" ")
            
            # Leer con optimizaciones
            df = pd.read_csv(
                fp,
                nrows=max_rows,
                low_memory=True,
                encoding='utf-8'
            )
            
            # Samplear si se especifica
            if sample_frac and 0 < sample_frac < 1:
                df = df.sample(frac=sample_frac, random_state=SEED)
            
            print(f"✓ {len(df):,} filas")
            
            # Normalizar columnas
            df = _standardize_columns(df)

            # Si falta 'año', inferir del nombre del archivo
            if "año" not in df.columns:
                y = _infer_year_from_path(fp)
                if y:
                    df["año"] = y

            df = _coerce_types(df)
            dfs.append(df)
            
        except KeyboardInterrupt:
            print("\n⚠️  Carga interrumpida por usuario")
            if dfs:
                print(f"   Usando {len(dfs)} archivos cargados hasta ahora")
                break
            else:
                return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

    if not dfs:
        print("❌ No se pudo cargar ningún CSV válido")
        return None

    print(f"\n🔄 Concatenando {len(dfs)} DataFrames...")
    data = pd.concat(dfs, ignore_index=True, sort=False)

    # Mantener solo columnas útiles
    wanted = [
        "año",
        "promedio_asistencia",
        "porcentaje_aprobacion",
        "promedio_notas",
        "tasa_2020",
        "estudiantes_retirados",
        "porcentaje_retiro",
        "total_estudiantes",
    ]
    keep = [c for c in wanted if c in data.columns]
    if keep:
        other = [c for c in data.columns if c not in keep]
        data = data[keep + other]

    print(f"✅ Datos cargados: {data.shape[0]:,} filas × {data.shape[1]} columnas")
    print(f"   Memoria: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Mostrar años disponibles si existe la columna
    if "año" in data.columns:
        years = data["año"].value_counts().sort_index()
        print(f"   Años: {dict(years)}")
    
    return data

def get_clean_data_all_final(path: str | Path = None, sample_frac: float | None = None, random_state: int = 42) -> pd.DataFrame:
    """
    Carga exclusivamente datasets/csvClear/Rendimiento_all_final.csv (o csvClear/Rendimiento_all_final.csv).
    Normaliza tipos y valores clave para compatibilidad con features/targets.
    """
    candidates = [
        Path(path) if path else None,
        Path("datasets/csvClear/Rendimiento_all_final.csv"),
        Path("csvClear/Rendimiento_all_final.csv"),
    ]
    csv_path = next((p for p in candidates if p and p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "No se encontró Rendimiento_all_final.csv en datasets/csvClear o csvClear. "
            "Indica path explícito con get_clean_data_all_final(path=...)."
        )

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().upper().lstrip("\ufeff") for c in df.columns]

    # Tipos base
    if "AGNO" in df.columns:
        df["AGNO"] = pd.to_numeric(df["AGNO"], errors="coerce").astype("Int64")
    if "PROM_GRAL" in df.columns:
        df["PROM_GRAL"] = pd.to_numeric(df["PROM_GRAL"], errors="coerce")
    if "ASISTENCIA" in df.columns:
        df["ASISTENCIA"] = pd.to_numeric(df["ASISTENCIA"], errors="coerce")
        mask_01 = df["ASISTENCIA"].between(0, 1, inclusive="both")
        df.loc[mask_01, "ASISTENCIA"] = df.loc[mask_01, "ASISTENCIA"] * 100
        df["ASISTENCIA"] = df["ASISTENCIA"].clip(lower=0, upper=100)

    if "MRUN" in df.columns:
        df["MRUN"] = df["MRUN"].astype(str).str.strip()
        df = df[df["MRUN"].notna() & (df["MRUN"].str.len() > 0)]

    if sample_frac and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    return df