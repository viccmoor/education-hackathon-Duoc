import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from src.model import DesercionPredictor

def _safe_metric(metric_fn, y_true, y_score):
    try:
        return metric_fn(y_true, y_score)
    except Exception:
        return None

def report_metrics(model, X, y):
    """
    Reporta métricas principales y fairness por subgrupo.
    Evita errores cuando un subgrupo tiene solo una clase.
    """
    y_pred = model.predict_proba(X)

    auroc = _safe_metric(roc_auc_score, y, y_pred)
    auprc = _safe_metric(average_precision_score, y, y_pred)
    brier = _safe_metric(brier_score_loss, y, y_pred)
    fairness = {}

    for grupo in ["region", "dependencia", "genero"]:
        if grupo in X.columns:
            vals = X[grupo].dropna().unique()
            for g in vals:
                mask = X[grupo] == g
                if mask.sum() > 0 and y[mask].nunique() > 1:
                    fairness[f"auroc_{grupo}_{g}"] = _safe_metric(roc_auc_score, y[mask], y_pred[mask])
                else:
                    fairness[f"auroc_{grupo}_{g}"] = None

    print("Métricas globales (test):")
    print(f"  AUROC: {auroc:.3f}" if auroc is not None else "  AUROC: N/A")
    print(f"  AUPRC: {auprc:.3f}" if auprc is not None else "  AUPRC: N/A")
    print(f"  Brier: {brier:.3f}" if brier is not None else "  Brier: N/A")

    valid_f = [v for v in fairness.values() if v is not None]
    if valid_f:
        gap = max(valid_f) - min(valid_f)
        print(f"  Fairness gap (AUROC): {gap:.3f}")
    else:
        print("  Fairness gap (AUROC): N/A")

    print("\nFairness por grupo:")
    for k, v in fairness.items():
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: N/A (insuficientes datos)")

if __name__ == "__main__":
    from src.load_data import get_clean_data
    from src.features import create_features
    from src.targets import create_target

    print("Cargando datos para evaluación...")
    df = get_clean_data()

    if df is not None:
        X_all = create_features(df)
        y_all = create_target(df)

        if "año" in X_all.columns and X_all["año"].notna().any():
            test_year = 2024 if (X_all["año"] == 2024).any() else int(X_all["año"].dropna().max())
            test_mask = X_all["año"] == test_year
        else:
            cutoff = int(len(X_all) * 0.8)
            test_mask = pd.Series([False]*cutoff + [True]*(len(X_all)-cutoff), index=X_all.index)

        X_test = X_all[test_mask]
        y_test = y_all[test_mask]

        print(f"\nTest set: {X_test.shape}")
        print("Cargando modelo...")
        model = DesercionPredictor.load('models/desercion_predictor.joblib')

        print("\nEvaluando modelo...")
        report_metrics(model, X_test, y_test)

        print("\nDemo de predicción:")
        demo = pd.DataFrame({
            "promedio_asistencia": [85.0, 70.0, 95.0, 60.0],
            "porcentaje_aprobacion": [0.75, 0.60, 0.90, 0.50],
            "promedio_notas": [5.5, 4.8, 6.2, 4.0],
            "tasa_2020": [0.03, 0.08, 0.01, 0.12],
            "estudiantes_retirados": [10, 25, 5, 40],
            "porcentaje_retiro": [0.02, 0.05, 0.01, 0.10],
            "total_estudiantes": [500, 500, 500, 500]
        })
        from src.features import create_features as cf
        demo_features = cf(demo)
        preds = model.predict_proba(demo_features)
        for i, prob in enumerate(preds):
            nivel = "ALTO" if prob > 0.8 else "MEDIO" if prob > 0.5 else "BAJO"
            print(f"  Estudiante {i+1}: {prob:.2%} - Riesgo {nivel}")