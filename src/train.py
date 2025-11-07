import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_data import get_clean_data_all_final
from src.features import create_features
from src.targets import create_target
from src.model import DesercionPredictor
import pandas as pd

def train_and_save(use_sample: bool = False, all_final_path: str | None = None):
    print("="*60)
    print("ENTRENAMIENTO DE MODELO - PREDICCIÓN DE DESERCIÓN")
    print("="*60)

    print("\n📊 Cargando datos (Rendimiento_all_final.csv)...")
    df = get_clean_data_all_final(
        path=all_final_path,
        sample_frac=0.1 if use_sample else None
    )
    if df is None or len(df) == 0:
        print("❌ Error: no se pudieron cargar los datos")
        return

    print("\n🔧 Creando features y target...")
    X = create_features(df)
    y = create_target(df)

    print(f"   Features: {X.shape}")
    print(f"   Target distribución: {y.value_counts(dropna=False).to_dict()}")

    # Split temporal usando AGNO si existe, si no 80/20
    print("\n✂️  Dividiendo train/test...")
    year_col = "AGNO" if "AGNO" in X.columns else ("año" if "año" in X.columns else None)
    if year_col:
        last_year = int(pd.to_numeric(X[year_col], errors="coerce").dropna().max())
        train_mask = X[year_col] < last_year
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[~train_mask], y[~train_mask]
        print(f"   Split temporal: {year_col} < {last_year} = train, == {last_year} = test")
    else:
        cutoff = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
        X_test,  y_test  = X.iloc[cutoff:], y.iloc[cutoff:]
        print("   Split 80/20")

    print(f"   Train: {X_train.shape[0]:,} muestras")
    print(f"   Test:  {X_test.shape[0]:,} muestras")

    # Asegurar 2 clases en train
    cls = y_train.value_counts().to_dict()
    if len(cls) < 2:
        raise ValueError(f"Target con 1 clase en train: {cls}. Revisa reglas en src/targets.py:create_target")

    print(f"   Train distribución: {cls}")

    print("\n🤖 Entrenando modelo...")
    model = DesercionPredictor()
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        print("\n📈 Evaluando en test set...")
        from sklearn.metrics import classification_report, roc_auc_score
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        print(f"\n{classification_report(y_test, y_pred, target_names=['Riesgo Bajo', 'Riesgo Alto'])}")
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    model_path = Path("models/desercion_predictor.joblib")
    model_path.parent.mkdir(exist_ok=True)
    print(f"\n💾 Guardando modelo en {model_path}...")
    model.save(str(model_path))
    print("✅ Modelo guardado exitosamente")
    print("="*60)

if __name__ == "__main__":
    import sys
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    arg_path = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--all-final-path=")), None)
    train_and_save(use_sample=use_sample, all_final_path=arg_path)