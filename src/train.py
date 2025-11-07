import sys
from pathlib import Path
import time
import model
import os 

# Asegura imports del paquete src cuando se ejecuta como script
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from src.load_data import get_clean_data_all_final
from src.features import create_features
from src.targets import create_target
from src.model import DesercionPredictor

# Patrones de fuga por nombre de columna
LEAK_PATTERNS = ("sit_fin", "sit_fin_r", "aprob", "reprob", "deser", "retir", "baja", "elim", "label")


def _detect_leakage(df: pd.DataFrame) -> list[str]:
    leak_cols = []
    for c in df.columns:
        lc = c.lower()
        if any(p in lc for p in LEAK_PATTERNS):
            leak_cols.append(c)
    return leak_cols


def create_target_strict(df: pd.DataFrame) -> pd.Series:
    """
    Etiqueta SOLO con SIT_FIN_R/SIT_FIN (texto):
      1 = reprobación/deserción/retirado/elim/baja/abandono, 0 = aprobado/promovido.
    No usa PROM_GRAL ni ASISTENCIA para evitar que el modelo aprenda una regla determinista.
    """
    y = pd.Series(np.nan, index=df.index, dtype="float")

    for col in ("SIT_FIN_R", "SIT_FIN"):
        if col in df.columns:
            s = df[col].astype(str).str.lower().fillna("")
            aprob = s.str.contains(r"apro|promov") & ~s.str.contains(r"no\s*apro")
            reprob = s.str.contains(r"reprob|repit|retir|deser|elim|baja|aband")
            y.loc[aprob & y.isna()] = 0
            y.loc[reprob & y.isna()] = 1

    return y.astype("Float64")


def _drop_target_twins(X: pd.DataFrame, y: pd.Series, sample_n: int = 20000) -> pd.DataFrame:
    """
    Elimina columnas de X que son (casi) idénticas al target (o su complemento) o
    extremadamente correlacionadas con él. Usa una muestra para ser eficiente.
    """
    if len(X) == 0:
        return X

    idx = X.index
    sample_idx = idx if len(idx) <= sample_n else idx.to_series().sample(n=sample_n, random_state=42).index
    y_s = pd.to_numeric(y.loc[sample_idx], errors="coerce")

    drop_cols = []
    for c in X.columns:
        xs = pd.to_numeric(X.loc[sample_idx, c], errors="coerce")
        mask = xs.notna() & y_s.notna()
        if mask.sum() == 0:
            continue
        same = (xs[mask] == y_s[mask]).mean()
        comp = (xs[mask] == (1 - y_s[mask])).mean() if set(y_s.dropna().unique()) <= {0, 1} else 0.0
        corr = abs(np.corrcoef(xs[mask], y_s[mask])[0, 1]) if mask.sum() > 1 else 0.0
        if same >= 0.999 or comp >= 0.999 or corr >= 0.999:
            drop_cols.append(c)

    if drop_cols:
        print(f"⚠️  Eliminando columnas idénticas/al target: {drop_cols}")
        X = X.drop(columns=drop_cols)
    return X


def train_and_save(use_sample: bool = False, all_final_path: str | None = None):
    
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO - PREDICCIÓN DE DESERCIÓN")
    print("=" * 60)

    print("\n📊 Cargando datos (Rendimiento_all_final.csv)...")
    df = get_clean_data_all_final(
        path=all_final_path,
        sample_frac=0.1 if use_sample else None,
    )
    if df is None or len(df) == 0:
        print("❌ Error: no se pudieron cargar los datos")
        return

    print("\n🔧 Creando features y target...")
    X = create_features(df)

    # 1) Intentar construir target solo con SIT_FIN*/SIT_FIN
    y = create_target_strict(df)
    nan_ratio = float(y.isna().mean())
    if nan_ratio > 0.9:
        # 2) Fallback: usa la regla de targets original (nota/asistencia)
        print("ℹ️  No hay suficiente SIT_FIN*/SIT_FIN. Usando regla fallback (nota/asistencia).")
        y = create_target(df)
        # Para evitar fuga definicional, quita prom/asistencia si existen en X
        cols_to_drop = [c for c in X.columns if c.lower() in ("prom_gral", "promedio", "asistencia", "asistencia_pct")]
        if cols_to_drop:
            print(f"🔒  Evitando fuga: quitando features {cols_to_drop}")
            X = X.drop(columns=cols_to_drop)

    # Quitar filas sin etiqueta
    keep = ~y.isna()
    if keep.sum() < len(y):
        print(f"ℹ️  Filas sin etiqueta removidas: {(~keep).sum():,}")
    X, y = X.loc[keep], y.loc[keep].astype(int)

    # Chequeo de fuga por nombre y por similitud con el target
    leak_cols = _detect_leakage(X)
    if leak_cols:
        print(f"⚠️  Posible fuga por nombre en features: {leak_cols}. Eliminando...")
        X = X.drop(columns=leak_cols)
    X = _drop_target_twins(X, y)

    print(f"   Features: {X.shape}")
    dist = y.value_counts(dropna=False).to_dict()
    print(f"   Target distribución: {dist}")

    # Split temporal si hay AGNO / año; si no 80/20
    print("\n✂️  Dividiendo train/test...")
    year_col = None
    for cand in ("AGNO", "año"):
        if cand in X.columns:
            year_col = cand
            break

    if year_col:
        last_year = int(pd.to_numeric(X[year_col], errors="coerce").dropna().max())
        train_mask = X[year_col] < last_year
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]
        print(f"   Split temporal: {year_col} < {last_year} = train, == {last_year} = test")
    else:
        cutoff = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
        X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]
        print("   Split 80/20")

    print(f"   Train: {X_train.shape[0]:,} muestras")
    print(f"   Test:  {X_test.shape[0]:,} muestras")

    cls_train = y_train.value_counts().to_dict()
    if len(cls_train) < 2:
        raise ValueError(f"Target con 1 clase en train: {cls_train}. Ajusta reglas en src/targets.create_target.")
    print(f"   Train distribución: {cls_train}")

    # === Downsampling único antes del entrenamiento ===
    print("\n⚙️ Preparando dataset de entrenamiento...")
    max_major = int(os.getenv("MAX_MAJOR", "300000"))
    max_minor = int(os.getenv("MAX_MINOR", "300000"))  # opcional para limitar también la minoría
    vc = y_train.value_counts()
    maj = vc.idxmax()
    # minoría = la otra clase
    min_cls = [c for c in vc.index if c != maj][0]

    idx_major = y_train[y_train == maj].index
    idx_minor = y_train[y_train == min_cls].index

    if len(idx_major) > max_major:
        idx_major = idx_major.to_series().sample(n=max_major, random_state=42).index
    if len(idx_minor) > max_minor:
        idx_minor = idx_minor.to_series().sample(n=max_minor, random_state=42).index

    keep_idx = idx_major.union(idx_minor)
    X_train = X_train.loc[keep_idx]
    y_train = y_train.loc[keep_idx]
    print(f"   → Tamaño final train: {len(X_train):,} | maj={sum(y_train==maj):,} | min={sum(y_train==min_cls):,}")

    # FAST_TRAIN: usar SGDClassifier (mini‑batch) para acelerar
    fast_train = os.getenv("FAST_TRAIN", "0") == "1"

    print("\n🤖 Entrenando modelo...")
    from sklearn.linear_model import SGDClassifier, LogisticRegression

    if fast_train:
        print("   ⚡ FAST_TRAIN=1 → SGDClassifier (log loss, mini‑batch)")
        base_clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            max_iter=10,
            tol=1e-3,
            random_state=42,
            class_weight="balanced",
        )
    else:
        base_clf = LogisticRegression(
            solver="lbfgs",
            max_iter=600,
            random_state=42,
            class_weight="balanced"
        )

    # Una sola instancia del predictor
    model = DesercionPredictor(model=base_clf)

    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"   ✅ Entrenado en {dt/60:.2f} min")

    # Evaluación
    if len(X_test) > 0:
        print("\n📈 Evaluando en test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        try:
            report = classification_report(y_test, y_pred, target_names=["Riesgo Bajo", "Riesgo Alto"])
            cm = confusion_matrix(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)
            print(f"\n{report}")
            print(f"ROC-AUC: {roc:.3f}")
            print(f"Confusion Matrix:\n{cm}")
        except Exception as e:
            print(f"⚠️  Error métricas: {e}")

    # Guardar
    model_path = Path("models/desercion_predictor.joblib")
    model_path.parent.mkdir(exist_ok=True)
    print(f"\n💾 Guardando modelo en {model_path}...")
    model.save(str(model_path))
    print("✅ Modelo guardado")
    print("=" * 60)


if __name__ == "__main__":
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    # Permite ruta explícita al CSV combinado
    arg_path = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--all-final-path=")), None)
    train_and_save(use_sample=use_sample, all_final_path=arg_path)