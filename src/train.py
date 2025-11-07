import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load_data import get_clean_data
from src.features import create_features
from src.targets import create_target
from src.model import DesercionPredictor

def train_and_save(use_sample: bool = False):
    """
    Entrena y guarda el modelo.
    
    Args:
        use_sample: Si True, usa solo 10% de los datos para entrenamiento rápido
    """
    print("="*60)
    print("ENTRENAMIENTO DE MODELO - PREDICCIÓN DE DESERCIÓN")
    print("="*60)
    
    print("\n📊 Cargando datos...")
    if use_sample:
        print("   ⚡ Modo rápido: usando 10% de los datos")
        df = get_clean_data(sample_frac=0.1)
    else:
        print("   🐌 Cargando dataset completo...")
        df = get_clean_data()
    
    if df is None:
        print("❌ Error: no se pudieron cargar los datos")
        return
    
    print("\n🔧 Creando features y target...")
    X = create_features(df)
    y = create_target(df)
    
    print(f"   Features: {X.shape}")
    print(f"   Target distribución: {y.value_counts().to_dict()}")
    
    # Split temporal
    print("\n✂️  Dividiendo train/test...")
    if "año" in X.columns:
        train_mask = X["año"] < 2024
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        print(f"   Split temporal: año < 2024 = train, ≥ 2024 = test")
    else:
        cutoff = int(len(X) * 0.8)
        X_train = X.iloc[:cutoff]
        y_train = y.iloc[:cutoff]
        X_test = X.iloc[cutoff:]
        y_test = y.iloc[cutoff:]
        print(f"   Split 80/20")
    
    print(f"   Train: {X_train.shape[0]:,} muestras")
    print(f"   Test:  {X_test.shape[0]:,} muestras")
    print(f"   Train distribución: {y_train.value_counts().to_dict()}")
    
    print("\n🤖 Entrenando modelo...")
    model = DesercionPredictor()
    model.fit(X_train, y_train)
    
    # Evaluar en test
    if len(X_test) > 0:
        print("\n📈 Evaluando en test set...")
        try:
            from sklearn.metrics import classification_report, roc_auc_score
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            print(f"\n{classification_report(y_test, y_pred, target_names=['Riesgo Bajo', 'Riesgo Alto'])}")
            print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
        except Exception as e:
            print(f"⚠️  No se pudo evaluar: {e}")
    
    model_path = Path("models/desercion_predictor.joblib")
    model_path.parent.mkdir(exist_ok=True)
    
    print(f"\n💾 Guardando modelo en {model_path}...")
    model.save(str(model_path))
    print("✅ Modelo guardado exitosamente")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Si se pasa --sample, usar sample
    use_sample = "--sample" in sys.argv or "-s" in sys.argv
    
    train_and_save(use_sample=use_sample)