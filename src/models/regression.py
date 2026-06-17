import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

from src.config import DATASET_2_PATH, MODELS_SAVE_DIR, RANDOM_STATE

def load_and_preprocess_demand_data():
    """Carga y preprocesa el dataset de series temporales de stock e insumos."""
    df = pd.read_csv(DATASET_2_PATH)
    
    # Asegurar orden temporal por fecha
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by='fecha').reset_index(drop=True)
    
    # Ingeniería de variables temporales de la fecha
    df['dia_semana_num'] = df['fecha'].dt.dayofweek
    df['is_weekend'] = df['dia_semana_num'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Escalado de variables relacionadas con pacientes y ocupación (Proyección de Crecimiento)
    paciente_ocupacion_cols = [
        'ocupacion_deseada', 'ocupacion_albergue', 'porcentaje_ocupacion',
        'pacientes_total', 'pacientes_prioridad_baja', 'pacientes_prioridad_media',
        'pacientes_prioridad_alta', 'pacientes_estadio_0_I', 'pacientes_estadio_II',
        'pacientes_estadio_III_IV', 'pacientes_quimioterapia', 'pacientes_cirugia',
        'pacientes_paliativos', 'pacientes_soporte_nutricional', 'pacientes_con_anemia',
        'pacientes_con_vomitos', 'pacientes_dolor_alto', 'pacientes_desnutricion',
        'pacientes_quimio_curativa', 'pacientes_quimio_avanzada',
        'pacientes_flot_estimado', 'pacientes_folfox_capox_estimado',
        'pacientes_her2_positivo_estimado'
    ]
    for col in paciente_ocupacion_cols:
        df[f'{col}_scaled_100_familias'] = df[col] * 2

    # Columnas categóricas a codificar (excluyendo 'fecha' y targets)
    categorical_cols = [
        'dia_semana', 'item_id', 'item_nombre', 'categoria_item', 'tipo_stock', 'unidad_medida'
    ]
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Conversión de stock crítico a binario (0/1)
    df_encoded['stock_critico_7d_binary'] = df_encoded['stock_critico_7d'].apply(lambda x: 1 if x == 'si' else 0)
    df_encoded['stock_critico_14d_binary'] = df_encoded['stock_critico_14d'].apply(lambda x: 1 if x == 'si' else 0)
    df_encoded['stock_critico_binary'] = df_encoded['stock_critico'].apply(lambda x: 1 if x == 'si' else 0)
    
    # Guardar las fechas para división temporal posterior
    fechas = df_encoded['fecha']
    
    # Eliminar columnas originales y metadatos no numéricos
    df_final = df_encoded.drop(columns=[
        'fecha', 'dia_semana_num', 'stock_critico', 'stock_critico_7d', 'stock_critico_14d'
    ])
    
    # Agregar fecha de nuevo de forma temporal para filtrado
    df_final['fecha'] = fechas
    
    # Restringir los datos al 3er bimestre (meses 5 y 6) tal como en el baseline
    df_modelo = df_final[df_final['mes'].isin([5, 6])].copy()
    
    # Recuperamos los targets de interés
    y_7d = df_modelo['demanda_7_dias']
    y_14d = df_modelo['demanda_14_dias']
    
    # Separamos X
    columnas_nombres_items = [col for col in df_modelo.columns if col.startswith('item_nombre_')]
    columnas_escaladas = [col for col in df_modelo.columns if 'scaled_100_familias' in col]
    
    columnas_a_eliminar = [
        'fecha', # Dropear fecha temporal
        'demanda_7_dias', 'demanda_14_dias',
        'stock_proyectado_7d', 'stock_proyectado_14d',
        'stock_critico_7d_binary', 'stock_critico_14d_binary', 'stock_critico_binary'
    ] + columnas_nombres_items + columnas_escaladas
    
    X = df_modelo.drop(columns=columnas_a_eliminar)
    
    # Asegurar que todas las columnas en X son numéricas
    X = X.select_dtypes(include=[np.number, bool])
    
    return X, y_7d, y_14d

def run_regression_gridsearch(X, y, target_name):
    """Ejecuta GridSearchCV para XGBoost con TimeSeriesSplit."""
    print(f"\n--- Optimización de XGBoost para {target_name} ---")
    
    # Validación temporal: TimeSeriesSplit de 5 pliegues
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Hiperparámetros a optimizar (learning_rate, max_depth, alpha, lambda)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'reg_alpha': [0.1, 1.0, 10.0],
        'reg_lambda': [0.1, 1.0, 10.0]
    }
    
    xgb = XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    print(f"Mejores parametros para {target_name}: {grid_search.best_params_}")
    
    # Evaluacion final de validacion temporal
    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[test_idx]
        
        # Re-ajustar el mejor modelo en esta particion historica
        best_model.fit(X_tr, y_tr)
        preds = best_model.predict(X_val)
        
        # Calcular metricas
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        # Calculo de MAPE robusto (excluyendo ceros)
        y_val_arr = np.array(y_val)
        preds_arr = np.array(preds)
        mask = y_val_arr != 0
        if np.sum(mask) > 0:
            mape = np.mean(np.abs((y_val_arr[mask] - preds_arr[mask]) / y_val_arr[mask])) * 100
        else:
            mape = 0.0
            
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        
    mean_mae = np.mean(maes)
    mean_rmse = np.mean(rmses)
    mean_mape = np.mean(mapes)
    
    print(f"Resultados Validacion Temporal (Promedio 5 pliegues):")
    print(f"  MAE : {mean_mae:.4f}")
    print(f"  RMSE: {mean_rmse:.4f}")
    print(f"  MAPE: {mean_mape:.2f}%")
    
    return best_model, mean_mae, mean_rmse, mean_mape

def train_and_optimize():
    X, y_7d, y_14d = load_and_preprocess_demand_data()
    print(f"Dimensiones de X para modelado: {X.shape}")
    
    # 1. Optimizar Modelo 7 dias
    best_xgb_7d, mae_7d, rmse_7d, mape_7d = run_regression_gridsearch(X, y_7d, "demanda_7_dias")
    
    # Verificar cumplimiento de meta de MAPE <= 15%
    if mape_7d <= 15.0:
        print(f"[OK] Cumple meta de MAPE para 7 dias (MAPE = {mape_7d:.2f}% <= 15.0%)")
    else:
        print(f"[ALERTA] No cumple meta de MAPE para 7 dias (MAPE = {mape_7d:.2f}% > 15.0%)")
        
    # Guardar modelo
    model_path_7d = MODELS_SAVE_DIR / "best_model_regression_7d.pkl"
    with open(model_path_7d, "wb") as f:
        pickle.dump({
            "model": best_xgb_7d,
            "features": list(X.columns),
            "mape": mape_7d,
            "rmse": rmse_7d,
            "mae": mae_7d
        }, f)
    print(f"Modelo 7d guardado en {model_path_7d}")
    
    # 2. Optimizar Modelo 14 dias
    best_xgb_14d, mae_14d, rmse_14d, mape_14d = run_regression_gridsearch(X, y_14d, "demanda_14_dias")
    
    # Verificar cumplimiento de meta de MAPE <= 20%
    if mape_14d <= 20.0:
        print(f"[OK] Cumple meta de MAPE para 14 dias (MAPE = {mape_14d:.2f}% <= 20.0%)")
    else:
        print(f"[ALERTA] No cumple meta de MAPE para 14 dias (MAPE = {mape_14d:.2f}% > 20.0%)")
        
    # Guardar modelo
    model_path_14d = MODELS_SAVE_DIR / "best_model_regression_14d.pkl"
    with open(model_path_14d, "wb") as f:
        pickle.dump({
            "model": best_xgb_14d,
            "features": list(X.columns),
            "mape": mape_14d,
            "rmse": rmse_14d,
            "mae": mae_14d
        }, f)
    print(f"Modelo 14d guardado en {model_path_14d}")

if __name__ == "__main__":
    train_and_optimize()
