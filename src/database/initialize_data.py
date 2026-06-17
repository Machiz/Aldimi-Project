import os
import sys
import pickle
import datetime
import pandas as pd
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.connection import init_db, SessionLocal
from src.database.models import PacienteEvaluacionOCR, PacientePrediccionPrioridad, InsumoInventario
from src.pipeline.inference import preprocess_single_record, predict_priority
from src.config import MODELS_SAVE_DIR, DATASET_1_PATH, DATASET_2_PATH

def bootstrap_data():
    print("Inicializando base de datos y creando tablas...")
    init_db()
    
    db = SessionLocal()
    try:
        # 1. BOOTSTRAP INVENTARIO DE INSUMOS
        # Contamos cuántos insumos hay registrados
        insumos_count = db.query(InsumoInventario).count()
        if insumos_count == 0:
            print("Poblando inventario inicial de insumos desde insumos_cancer_final.csv...")
            if not DATASET_2_PATH.exists():
                print(f"Error: No se encontró el dataset de insumos en {DATASET_2_PATH}")
                return
            
            df_insumos = pd.read_csv(DATASET_2_PATH)
            # Seleccionar la última fecha disponible en el dataset histórico
            max_date = df_insumos['fecha'].max()
            print(f"Última fecha histórica en insumos: {max_date}")
            
            # Obtener el registro de stock para cada uno de los 15 insumos en esa fecha
            df_latest_stock = df_insumos[df_insumos['fecha'] == max_date]
            
            insumos_insertados = []
            for _, row in df_latest_stock.iterrows():
                insumo = InsumoInventario(
                    item_id=str(row['item_id']),
                    item_nombre=str(row['item_nombre']),
                    categoria_item=str(row['categoria_item']),
                    tipo_stock=str(row['tipo_stock']),
                    unidad_medida=str(row['unidad_medida']),
                    stock_actual=float(row['stock_actual']),
                    stock_minimo=float(row['stock_minimo']),
                    consumo_diario=float(row['consumo_diario']),
                    consumo_lag_1=float(row['consumo_lag_1']),
                    consumo_lag_7=float(row['consumo_lag_7']),
                    consumo_promedio_7d=float(row['consumo_promedio_7d']),
                    consumo_promedio_14d=float(row['consumo_promedio_14d']),
                    variabilidad_consumo=float(row['variabilidad_consumo'])
                )
                insumos_insertados.append(insumo)
                
            db.add_all(insumos_insertados)
            db.commit()
            print(f"Se insertaron {len(insumos_insertados)} insumos en el inventario inicial.")
        else:
            print(f"El inventario de insumos ya cuenta con {insumos_count} registros.")

        # 2. BOOTSTRAP PACIENTES ACTIVOS
        pacientes_count = db.query(PacienteEvaluacionOCR).count()
        if pacientes_count <= 5: # Si está prácticamente vacío (solo los de prueba)
            print("Poblando pacientes activos iniciales desde evaluaciones_pacientes_cancer_gastrico_sintetico.csv...")
            if not DATASET_1_PATH.exists():
                print(f"Error: No se encontró el dataset de evaluaciones en {DATASET_1_PATH}")
                return
            
            df_patients = pd.read_csv(DATASET_1_PATH)
            df_patients['fecha_evaluacion'] = pd.to_datetime(df_patients['fecha_evaluacion'])
            
            # Filtrar pacientes evaluados en o antes de '2026-02-10' (la fecha de cierre del inventario)
            # Ordenar por fecha de evaluación descendente y tomar los últimos 100 para simular la ocupación actual
            df_active_patients = df_patients[df_patients['fecha_evaluacion'] <= '2026-02-10']
            df_active_patients = df_active_patients.sort_values(by='fecha_evaluacion', ascending=False).head(100)
            
            # Cargar el clasificador
            model_path = MODELS_SAVE_DIR / "best_model_classification.pkl"
            if not model_path.exists():
                print(f"Error: No se encontró el modelo clasificador en {model_path}")
                return
            
            with open(model_path, "rb") as f:
                model_meta = pickle.load(f)
                
            print(f"Cargando {len(df_active_patients)} pacientes en el censo activo del albergue...")
            
            pacientes_insertados = 0
            preds_insertadas = 0
            
            for _, row in df_active_patients.iterrows():
                # Evitar duplicados por evaluacion_id
                exists = db.query(PacienteEvaluacionOCR).filter_by(evaluacion_id=str(row['evaluacion_id'])).first()
                if exists:
                    continue
                    
                # Crear objeto ORM de Evaluación
                eval_date = pd.to_datetime(row['fecha_evaluacion']).date()
                diag_date = pd.to_datetime(row['fecha_diagnostico']).date()
                ing_date = pd.to_datetime(row['fecha_ingreso_aldimi']).date()
                
                paciente = PacienteEvaluacionOCR(
                    evaluacion_id=str(row['evaluacion_id']),
                    paciente_id=str(row['paciente_id']),
                    fecha_evaluacion=eval_date,
                    fecha_diagnostico=diag_date,
                    fecha_ingreso_aldimi=ing_date,
                    edad=int(row['edad']),
                    sexo=str(row['sexo']),
                    region_procedencia=str(row['region_procedencia']),
                    zona_procedencia=str(row['zona_procedencia']),
                    viaja_desde_provincia=str(row['viaja_desde_provincia']),
                    acompanante=str(row['acompanante']),
                    nivel_vulnerabilidad=str(row['nivel_vulnerabilidad']),
                    dificultad_acceso_salud=str(row['dificultad_acceso_salud']),
                    tipo_histologico=str(row['tipo_histologico']),
                    estadio=str(row['estadio']),
                    grado_histologico=str(row['grado_histologico']) if pd.notna(row['grado_histologico']) else "desconocido",
                    profundidad_tumor=str(row['profundidad_tumor']),
                    ganglios_afectados=int(row['ganglios_afectados']),
                    metastasis=1 if str(row['metastasis']).strip().lower() == 'si' else 0,
                    sitio_metastasis=str(row['sitio_metastasis']) if pd.notna(row['sitio_metastasis']) else "ninguno",
                    perdida_peso_kg=float(row['perdida_peso_kg']) if pd.notna(row['perdida_peso_kg']) else 0.0,
                    estado_nutricional=str(row['estado_nutricional']),
                    hemoglobina=float(row['hemoglobina']) if pd.notna(row['hemoglobina']) else 12.0,
                    anemia=str(row['anemia']),
                    dolor=str(row['dolor']),
                    vomitos_frecuentes=str(row['vomitos_frecuentes']),
                    sangrado_digestivo=str(row['sangrado_digestivo']),
                    fatiga=str(row['fatiga']),
                    dificultad_alimentarse=str(row['dificultad_alimentarse']),
                    comorbilidades=str(row['comorbilidades']),
                    estado_funcional=str(row['estado_funcional']) if pd.notna(row['estado_funcional']) else "desconocido",
                    requiere_cirugia=str(row['requiere_cirugia']),
                    requiere_quimioterapia=str(row['requiere_quimioterapia']),
                    requiere_soporte_nutricional=str(row['requiere_soporte_nutricional']),
                    requiere_cuidados_paliativos=str(row['requiere_cuidados_paliativos']),
                    tratamiento_principal=str(row['tratamiento_principal'])
                )
                
                db.add(paciente)
                pacientes_insertados += 1
                
                # Ejecutar Modelo 1 (Inferencia de Prioridad)
                rec_dict = paciente.to_dict()
                processed_df = preprocess_single_record(rec_dict)
                pred_label, p_bajo, p_medio, p_alto = predict_priority(processed_df, model_meta)
                
                pred_orm = PacientePrediccionPrioridad(
                    evaluacion_id=paciente.evaluacion_id,
                    paciente_id=paciente.paciente_id,
                    fecha_prediccion=datetime.datetime.now(),
                    prioridad_predicha=pred_label,
                    probabilidad_bajo=float(p_bajo),
                    probabilidad_medio=float(p_medio),
                    probabilidad_alto=float(p_alto),
                    umbral_aplicado=float(model_meta["threshold"])
                )
                
                db.add(pred_orm)
                preds_insertadas += 1
                
            db.commit()
            print(f"Bootstrapping de pacientes completo: se insertaron {pacientes_insertados} pacientes y {preds_insertadas} predicciones.")
        else:
            print(f"La tabla de pacientes ya cuenta con {pacientes_count} registros.")
            
    except Exception as e:
        db.rollback()
        print(f"Error durante el bootstrapping: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    bootstrap_data()
