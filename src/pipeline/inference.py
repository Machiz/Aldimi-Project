import os
import datetime
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODELS_SAVE_DIR
from src.database.connection import SessionLocal
from src.database.models import PacienteEvaluacionOCR, PacientePrediccionPrioridad

# Mapeos de codificación idénticos al notebook de preparación
MAPA_BINARIO = {"no": 0, "si": 1}

MAPAS_ORDINALES = {
    "grupo_edad": {
        "nino_adolescente": 0,
        "adulto_joven": 1,
        "adulto": 2,
        "adulto_mayor": 3
    },
    "nivel_vulnerabilidad": {
        "medio": 0,
        "alto": 1,
        "muy_alto": 2,
        "bajo": 0, # Mapeo de seguridad para evitar fallos
    },
    "dificultad_acceso_salud": {
        "baja": 0,
        "media": 1,
        "alta": 2
    },
    "estadio": {
        "0": 0,
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "Estadio 0": 0,
        "Estadio I": 1,
        "Estadio II": 2,
        "Estadio III": 3,
        "Estadio IV": 4
    },
    "grado_histologico": {
        "desconocido": -1,
        "G1": 1,
        "G2": 2,
        "G3": 3,
        "G4": 4
    },
    "profundidad_tumor": {
        "mucosa": 0,
        "submucosa": 1,
        "muscular": 2,
        "serosa": 3,
        "invasion_local": 4,
        "enfermedad_diseminada": 5,
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4
    },
    "estado_nutricional": {
        "adecuado": 0,
        "riesgo": 1,
        "desnutricion": 2,
        "bueno": 0,
        "riesgo desnutricion": 1
    },
    "dolor": {
        "bajo": 0,
        "medio": 1,
        "alto": 2,
        "ausente": 0,
        "leve": 0,
        "moderado": 1,
        "severo": 2
    },
    "fatiga": {
        "baja": 0,
        "media": 1,
        "alta": 2,
        "no": 0,
        "si": 1
    },
    "comorbilidades": {
        "ninguna": 0,
        "una": 1,
        "multiples": 2,
        "no": 0,
        "si": 1
    },
    "estado_funcional": {
        "desconocido": -1,
        "bueno": 0,
        "limitado": 1,
        "dependiente": 2,
        "totalmente activo": 0,
        "ambulatorio": 1,
        "dependiente en cama": 2
    }
}

def preprocess_single_record(record_dict):
    """
    Toma un diccionario con los datos crudos extraídos de un registro OCR
    y realiza el preprocesamiento y la ingeniería de características.
    """
    df_temp = pd.DataFrame([record_dict])
    
    # Imputaciones básicas
    df_temp["hemoglobina"] = df_temp["hemoglobina"].fillna(12.0)
    df_temp["perdida_peso_kg"] = df_temp["perdida_peso_kg"].fillna(0.0)
    df_temp["grado_histologico"] = df_temp["grado_histologico"].fillna("desconocido")
    df_temp["sitio_metastasis"] = df_temp["sitio_metastasis"].fillna("ninguno")
    df_temp["estado_funcional"] = df_temp["estado_funcional"].fillna("desconocido")
    
    # 1. Variables Temporales
    fecha_eval = pd.to_datetime(df_temp["fecha_evaluacion"])
    fecha_diag = pd.to_datetime(df_temp["fecha_diagnostico"])
    fecha_ing = pd.to_datetime(df_temp["fecha_ingreso_aldimi"])
    
    df_temp["dias_desde_diagnostico"] = (fecha_eval - fecha_diag).dt.days
    df_temp["dias_en_albergue"] = (fecha_eval - fecha_ing).dt.days
    
    # 2. Codificación Binaria
    variables_binarias = [
        "anemia", "vomitos_frecuentes", "sangrado_digestivo", "fatiga", 
        "dificultad_alimentarse", "comorbilidades", "requiere_cirugia", 
        "requiere_quimioterapia", "requiere_soporte_nutricional", 
        "requiere_cuidados_paliativos", "viaja_desde_provincia", "acompanante"
    ]
    for col in variables_binarias:
        val = str(df_temp[col].iloc[0]).lower().strip()
        df_temp[col] = MAPA_BINARIO.get(val, 0)
        
    # 3. Codificación Ordinal
    for col, mapping in MAPAS_ORDINALES.items():
        if col in df_temp.columns:
            val = str(df_temp[col].iloc[0]).lower().strip()
            df_temp[col] = mapping.get(val, 0)
            
    # 4. Ingeniería de Características
    df_temp["indice_avance_oncologico"] = (
        df_temp["estadio"] +
        df_temp["profundidad_tumor"] +
        df_temp["metastasis"] * 2 +
        df_temp["ganglios_afectados"]
    )
    
    df_temp["enfermedad_avanzada"] = (
        (df_temp["estadio"] >= 3) |
        (df_temp["metastasis"] == 1) |
        (df_temp["profundidad_tumor"] >= 4)
    ).astype(int)
    
    df_temp["alta_carga_ganglionar"] = (
        df_temp["ganglios_afectados"] >= 6
    ).astype(int)
    
    df_temp["perdida_peso_alta"] = (
        df_temp["perdida_peso_kg"] >= 8
    ).astype(int)
    
    df_temp["hemoglobina_baja"] = (
        df_temp["hemoglobina"] < 11
    ).astype(int)
    
    df_temp["indice_deterioro_nutricional"] = (
        df_temp["estado_nutricional"] +
        df_temp["dificultad_alimentarse"] +
        df_temp["perdida_peso_alta"] +
        df_temp["requiere_soporte_nutricional"]
    )
    
    # Síntomas
    df_temp["dolor_alto"] = (df_temp["dolor"] == 2).astype(int)
    df_temp["fatiga_alta"] = (df_temp["fatiga"] == 2).astype(int)
    
    df_temp["conteo_sintomas_relevantes"] = (
        df_temp["dolor_alto"] +
        df_temp["fatiga_alta"] +
        df_temp["vomitos_frecuentes"] +
        df_temp["sangrado_digestivo"] +
        df_temp["dificultad_alimentarse"] +
        df_temp["anemia"]
    )
    
    # Tratamiento
    df_temp["conteo_necesidades_tratamiento"] = (
        df_temp["requiere_cirugia"] +
        df_temp["requiere_quimioterapia"] +
        df_temp["requiere_soporte_nutricional"] +
        df_temp["requiere_cuidados_paliativos"]
    )
    
    df_temp["tratamiento_alta_complejidad"] = (
        (df_temp["requiere_quimioterapia"] == 1) |
        (df_temp["requiere_cuidados_paliativos"] == 1)
    ).astype(int)
    
    # OHE variables nominales
    # Para predicción unitaria o por lote, mapeamos manualmente los One-Hot de interes del modelo:
    # 1. tratamiento_principal
    tp_val = str(df_temp["tratamiento_principal"].iloc[0]).lower().strip()
    df_temp["tratamiento_principal_cuidados_paliativos"] = 1 if "paliativo" in tp_val else 0
    df_temp["tratamiento_principal_cirugia"] = 1 if tp_val == "cirugia" else 0
    
    # 2. sitio_metastasis
    sm_val = str(df_temp["sitio_metastasis"].iloc[0]).lower().strip()
    df_temp["sitio_metastasis_ninguno"] = 1 if sm_val == "ninguno" else 0
    
    return df_temp

def predict_priority(processed_df, model_meta):
    """Genera la prioridad predicha aplicando el umbral calibrado de clase Alta."""
    model = model_meta["model"]
    threshold = model_meta["threshold"]
    features = model_meta["features"]
    
    # Seleccionar las 30 columnas necesarias del modelo
    X_inference = processed_df[features]
    
    probs = model.predict_proba(X_inference)[0]
    classes = list(model.classes_)
    
    alto_idx = classes.index("alto")
    bajo_idx = classes.index("bajo")
    medio_idx = classes.index("medio")
    
    # Aplicar umbral calibrado para Recall de alta prioridad >= 0.85
    if probs[alto_idx] >= threshold:
        pred_label = "alto"
    else:
        if probs[bajo_idx] >= probs[medio_idx]:
            pred_label = "bajo"
        else:
            pred_label = "medio"
            
    return pred_label, probs[bajo_idx], probs[medio_idx], probs[alto_idx]

def run_pipeline():
    """Ejecuta el pipeline de inferencia leyendo de la base de datos."""
    model_path = MODELS_SAVE_DIR / "best_model_classification.pkl"
    if not model_path.exists():
        print(f"Error: El modelo no existe en {model_path}. Por favor entrena los modelos primero.")
        return
        
    with open(model_path, "rb") as f:
        model_meta = pickle.load(f)
        
    db = SessionLocal()
    try:
        # Obtener evaluaciones que no tengan predicción registrada
        subquery = db.query(PacientePrediccionPrioridad.evaluacion_id)
        pending_records = db.query(PacienteEvaluacionOCR).filter(~PacienteEvaluacionOCR.evaluacion_id.in_(subquery)).all()
        
        if not pending_records:
            print("No hay registros pendientes de predicción.")
            return
            
        print(f"Procesando {len(pending_records)} registros clínicos...")
        
        predictions_to_insert = []
        for rec in pending_records:
            # Convertir a dict
            rec_dict = rec.to_dict()
            # Preprocesar
            processed_df = preprocess_single_record(rec_dict)
            # Inferencia
            pred_label, p_bajo, p_medio, p_alto = predict_priority(processed_df, model_meta)
            
            # Crear predicción ORM
            pred_orm = PacientePrediccionPrioridad(
                evaluacion_id=rec.evaluacion_id,
                paciente_id=rec.paciente_id,
                fecha_prediccion=datetime.datetime.now(),
                prioridad_predicha=pred_label,
                probabilidad_bajo=float(p_bajo),
                probabilidad_medio=float(p_medio),
                probabilidad_alto=float(p_alto),
                umbral_aplicado=float(model_meta["threshold"])
            )
            predictions_to_insert.append(pred_orm)
            
        db.add_all(predictions_to_insert)
        db.commit()
        print(f"Inferencia completada con éxito. Se insertaron {len(predictions_to_insert)} predicciones.")
        
    except Exception as e:
        db.rollback()
        print(f"Error durante el pipeline de inferencia: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    run_pipeline()
