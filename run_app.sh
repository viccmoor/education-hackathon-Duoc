# ADAPTACIÓN EDUCACIÓN - FASE 4: Modelo (Logistic Regression)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import joblib, os

pipeline_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
])

print("🔄 Entrenando Logistic Regression...")
pipeline_lr.fit(X_train, y_train)

proba = pipeline_lr.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"AUROC: {roc_auc_score(y_test, proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, proba):.4f}")
print(f"Brier: {brier_score_loss(y_test, proba):.4f}")
print("CM @0.5:\n", confusion_matrix(y_test, pred))

# Guardar donde lo usa run_app.sh y la API
os.makedirs("models", exist_ok=True)
os.makedirs("datasets/models", exist_ok=True)
joblib.dump(pipeline_lr, "models/model_edu_lr.pkl")
joblib.dump(feature_names, "models/feature_names_edu.pkl")
joblib.dump(pipeline_lr, "datasets/models/model_edu_lr.pkl")
joblib.dump(feature_names, "datasets/models/feature_names_edu.pkl")
print("💾 Guardado: models/* y datasets/models/*")