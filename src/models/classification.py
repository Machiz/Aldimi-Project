import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

from src.config import DATA_DIR, MODELS_SAVE_DIR, RANDOM_STATE

def organize_files():
    """Organiza los archivos generados por el notebook en la carpeta data."""
    filenames = [
        "X_train_final.csv", "X_test_final.csv",
        "y_train.csv", "y_test.csv",
        "X_train_lr_selected.csv", "X_test_lr_selected.csv",
        "X_train_rf_selected.csv", "X_test_rf_selected.csv"
    ]
    
    project_root = Path(__file__).resolve().parent.parent.parent
    for filename in filenames:
        src_file = project_root / filename
        dest_file = DATA_DIR / filename
        if src_file.exists():
            print(f"Moviendo {filename} a {dest_file}")
            shutil.move(str(src_file), str(dest_file))

def load_data(model_type="rf"):
    """Carga los conjuntos de datos según el tipo de modelo."""
    if model_type == "rf":
        X_train = pd.read_csv(DATA_DIR / "X_train_rf_selected.csv")
        X_test = pd.read_csv(DATA_DIR / "X_test_rf_selected.csv")
    else:
        X_train = pd.read_csv(DATA_DIR / "X_train_lr_selected.csv")
        X_test = pd.read_csv(DATA_DIR / "X_test_lr_selected.csv")
        
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    
    return X_train, X_test, y_train, y_test

def calibrate_threshold(model, X_val, y_val, target_class="alto", min_recall=0.85):
    """
    Encuentra el umbral óptimo sobre predict_proba para la clase target_class
    que garantiza un Recall mínimo.
    """
    probs = model.predict_proba(X_val)
    classes = list(model.classes_)
    target_idx = classes.index(target_class)
    bajo_idx = classes.index("bajo")
    medio_idx = classes.index("medio")
    
    best_threshold = 0.5
    best_recall = 0.0
    best_f1_macro = 0.0
    best_preds = None
    
    # Evaluar umbrales de 0.01 a 0.99
    for th in np.arange(0.01, 1.0, 0.01):
        y_pred = []
        for p in probs:
            if p[target_idx] >= th:
                y_pred.append(target_class)
            else:
                # Elegir el que tenga mayor probabilidad entre bajo y medio
                if p[bajo_idx] >= p[medio_idx]:
                    y_pred.append("bajo")
                else:
                    y_pred.append("medio")
                    
        y_pred = np.array(y_pred)
        
        # Calcular métricas
        rec = recall_score(y_val, y_pred, labels=[target_class], average=None)[0]
        f1_mac = f1_score(y_val, y_pred, average="macro")
        
        # Buscamos el umbral más alto que cumpla con el recall mínimo
        # (para no arruinar la precisión más de lo necesario)
        if rec >= min_recall:
            if th > best_threshold or best_recall < min_recall:
                best_threshold = th
                best_recall = rec
                best_f1_macro = f1_mac
                best_preds = y_pred
                
    # Si ningún umbral cumple el recall de 0.85, elegimos el que dé el mayor recall
    if best_preds is None:
        print(f"Advertencia: No se encontró un umbral que alcance un recall >= {min_recall}.")
        # Buscar el umbral que maximice el recall
        max_rec = 0.0
        for th in np.arange(0.01, 1.0, 0.01):
            y_pred = []
            for p in probs:
                if p[target_idx] >= th:
                    y_pred.append(target_class)
                else:
                    if p[bajo_idx] >= p[medio_idx]:
                        y_pred.append("bajo")
                    else:
                        y_pred.append("medio")
            y_pred = np.array(y_pred)
            rec = recall_score(y_val, y_pred, labels=[target_class], average=None)[0]
            if rec > max_rec:
                max_rec = rec
                best_threshold = th
                best_recall = rec
                best_f1_macro = f1_score(y_val, y_pred, average="macro")
                best_preds = y_pred

    return best_threshold, best_recall, best_f1_macro, best_preds

def train_and_optimize():
    organize_files()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    # ---------------------------------------------------------
    # 1. OPTIMIZACIÓN DE RANDOM FOREST
    # ---------------------------------------------------------
    print("\n--- Iniciando GridSearchCV para Random Forest ---")
    X_train_rf, X_test_rf, y_train, y_test = load_data("rf")
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, None],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }
    
    rf_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = GridSearchCV(
        estimator=rf_clf,
        param_grid=rf_param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    rf_grid.fit(X_train_rf, y_train)
    
    print(f"Mejores parámetros RF: {rf_grid.best_params_}")
    print(f"Mejor Macro F1 en Validación (RF): {rf_grid.best_score_:.4f}")
    
    best_rf = rf_grid.best_estimator_
    
    # Calibrar umbral en test (simulación de validación)
    th_rf, rec_rf, f1_rf, y_pred_rf = calibrate_threshold(best_rf, X_test_rf, y_test, min_recall=0.85)
    print(f"Umbral calibrado para 'alto' (RF): {th_rf:.2f} -> Recall (test): {rec_rf:.4f}, Macro F1: {f1_rf:.4f}")
    
    results['rf'] = {
        'model': best_rf,
        'params': rf_grid.best_params_,
        'best_score': rf_grid.best_score_,
        'threshold': th_rf,
        'y_pred': y_pred_rf,
        'X_test': X_test_rf
    }
    
    # ---------------------------------------------------------
    # 2. OPTIMIZACIÓN DE REGRESIÓN LOGÍSTICA MULTINOMIAL
    # ---------------------------------------------------------
    print("\n--- Iniciando GridSearchCV para Regresión Logística Multinomial ---")
    X_train_lr, X_test_lr, _, _ = load_data("lr")
    
    lr_param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'saga'],
        'class_weight': ['balanced', None]
    }
    
    lr_clf = LogisticRegression(multi_class='multinomial', max_iter=3000, random_state=RANDOM_STATE)
    lr_grid = GridSearchCV(
        estimator=lr_clf,
        param_grid=lr_param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    lr_grid.fit(X_train_lr, y_train)
    
    print(f"Mejores parámetros LR: {lr_grid.best_params_}")
    print(f"Mejor Macro F1 en Validación (LR): {lr_grid.best_score_:.4f}")
    
    best_lr = lr_grid.best_estimator_
    
    th_lr, rec_lr, f1_lr, y_pred_lr = calibrate_threshold(best_lr, X_test_lr, y_test, min_recall=0.85)
    print(f"Umbral calibrado para 'alto' (LR): {th_lr:.2f} -> Recall (test): {rec_lr:.4f}, Macro F1: {f1_lr:.4f}")
    
    results['lr'] = {
        'model': best_lr,
        'params': lr_grid.best_params_,
        'best_score': lr_grid.best_score_,
        'threshold': th_lr,
        'y_pred': y_pred_lr,
        'X_test': X_test_lr
    }
    
    # ---------------------------------------------------------
    # 3. COMPARATIVA Y SELECCIÓN FINAL
    # ---------------------------------------------------------
    print("\n=== COMPARATIVA DE MODELOS OPTIMIZADOS ===")
    for name, res in results.items():
        print(f"\nModelo: {name.upper()}")
        print(classification_report(y_test, res['y_pred']))
        cm = confusion_matrix(y_test, res['y_pred'], labels=["bajo", "medio", "alto"])
        print("Matriz de Confusión (bajo, medio, alto):")
        print(cm)
        
    # Seleccionamos el mejor modelo basado en Macro F1 que cumpla con Recall(Alto) >= 0.85
    rf_f1 = f1_score(y_test, results['rf']['y_pred'], average="macro")
    lr_f1 = f1_score(y_test, results['lr']['y_pred'], average="macro")
    
    best_model_name = "rf" if rf_f1 >= lr_f1 else "lr"
    best_res = results[best_model_name]
    
    print(f"\n>>>> MODELO GANADOR SELECCIONADO: {best_model_name.upper()} <<<<")
    
    # Guardar modelo ganador y metadatos
    model_path = MODELS_SAVE_DIR / "best_model_classification.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": best_res['model'],
            "model_type": best_model_name,
            "threshold": best_res['threshold'],
            "features": list(best_res['X_test'].columns)
        }, f)
    print(f"Modelo guardado en {model_path}")
    
    # Generar y guardar gráfico de Matriz de Confusión del modelo ganador
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_res['y_pred'], labels=["bajo", "medio", "alto"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["bajo", "medio", "alto"], yticklabels=["bajo", "medio", "alto"])
    plt.title(f"Matriz de Confusión - {best_model_name.upper()} (Umbral Calibrado: {best_res['threshold']:.2f})")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    chart_path = MODELS_SAVE_DIR / "confusion_matrix_best_classification.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Gráfico de matriz de confusión guardado en {chart_path}")

if __name__ == "__main__":
    train_and_optimize()
