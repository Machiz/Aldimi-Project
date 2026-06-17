import os
import sys
import pickle
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.connection import SessionLocal
from src.database.models import PacienteEvaluacionOCR, PacientePrediccionPrioridad, InsumoInventario, InsumoPrediccion
from src.pipeline.inference import preprocess_single_record, predict_priority
from src.config import MODELS_SAVE_DIR

class AldimiPipeline:
    def __init__(self):
        """Inicializa el pipeline cargando los modelos entrenados y metadatos."""
        self.models_dir = MODELS_SAVE_DIR
        self._load_models()
        
    def _load_models(self):
        # 1. Cargar Modelo 1 (Clasificación de Prioridad)
        clf_path = self.models_dir / "best_model_classification.pkl"
        if not clf_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo clasificador en {clf_path}")
        with open(clf_path, "rb") as f:
            self.clf_meta = pickle.load(f)
            
        # 2. Cargar Modelos 2 (Regresión de Demanda 7d y 14d)
        reg7_path = self.models_dir / "best_model_regression_7d.pkl"
        if not reg7_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo regresor 7d en {reg7_path}")
        with open(reg7_path, "rb") as f:
            self.reg7_meta = pickle.load(f)
            
        reg14_path = self.models_dir / "best_model_regression_14d.pkl"
        if not reg14_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo regresor 14d en {reg14_path}")
        with open(reg14_path, "rb") as f:
            self.reg14_meta = pickle.load(f)

    def validate_patient_record(self, record_dict):
        """
        Ejecuta las reglas de validación de datos (Pre-ingesta) del contrato de datos.
        """
        errors = []
        
        # 1. Conversión de fechas a datetime.date
        try:
            fecha_eval = pd.to_datetime(record_dict["fecha_evaluacion"]).date()
            fecha_diag = pd.to_datetime(record_dict["fecha_diagnostico"]).date()
            fecha_ing = pd.to_datetime(record_dict["fecha_ingreso_aldimi"]).date()
        except Exception as e:
            errors.append(f"Error en formato de fechas: {e}")
            return False, errors, record_dict

        # 2. Validación Cronológica
        if not (fecha_eval >= fecha_diag and fecha_eval >= fecha_ing):
            errors.append("La fecha de evaluación debe ser posterior o igual a la fecha de diagnóstico e ingreso.")

        # 3. Control de Rango Edad
        edad = int(record_dict["edad"])
        if not (0 <= edad <= 120):
            errors.append(f"Edad fuera de rango permitido [0-120]: {edad}")

        # 4. Control de Rango Hemoglobina
        try:
            hemoglobina = float(record_dict["hemoglobina"]) if record_dict.get("hemoglobina") is not None else 12.0
            if not (3.0 <= hemoglobina <= 25.0):
                # Imputación de seguridad con mediana 12.0
                record_dict["hemoglobina"] = 12.0
        except ValueError:
            record_dict["hemoglobina"] = 12.0

        # Normalizar strings
        for key, value in record_dict.items():
            if isinstance(value, str):
                record_dict[key] = value.strip().lower()

        return len(errors) == 0, errors, record_dict

    def process_and_classify_patient(self, db, raw_record_dict):
        """
        Valida, preprocesa, clasifica la prioridad de un paciente y lo guarda en la base de datos.
        """
        # Validar registro
        is_valid, errors, record_dict = self.validate_patient_record(raw_record_dict)
        if not is_valid:
            raise ValueError(f"Validación de datos fallida: {', '.join(errors)}")

        # Convertir a objetos Date requeridos por SQLAlchemy
        fecha_eval = pd.to_datetime(record_dict["fecha_evaluacion"]).date()
        fecha_diag = pd.to_datetime(record_dict["fecha_diagnostico"]).date()
        fecha_ing = pd.to_datetime(record_dict["fecha_ingreso_aldimi"]).date()

        # Crear objeto PacienteEvaluacionOCR
        paciente_eval = PacienteEvaluacionOCR(
            evaluacion_id=record_dict["evaluacion_id"],
            paciente_id=record_dict["paciente_id"],
            fecha_evaluacion=fecha_eval,
            fecha_diagnostico=fecha_diag,
            fecha_ingreso_aldimi=fecha_ing,
            edad=int(record_dict["edad"]),
            sexo=record_dict["sexo"],
            region_procedencia=record_dict["region_procedencia"],
            zona_procedencia=record_dict["zona_procedencia"],
            viaja_desde_provincia=record_dict["viaja_desde_provincia"],
            acompanante=record_dict["acompanante"],
            nivel_vulnerabilidad=record_dict["nivel_vulnerabilidad"],
            dificultad_acceso_salud=record_dict["dificultad_acceso_salud"],
            tipo_histologico=record_dict["tipo_histologico"],
            estadio=record_dict["estadio"],
            grado_histologico=record_dict.get("grado_histologico", "desconocido"),
            profundidad_tumor=record_dict["profundidad_tumor"],
            ganglios_afectados=int(record_dict["ganglios_afectados"]),
            metastasis=1 if record_dict["metastasis"] in [1, "si", "1"] else 0,
            sitio_metastasis=record_dict.get("sitio_metastasis", "ninguno"),
            perdida_peso_kg=float(record_dict["perdida_peso_kg"]) if record_dict.get("perdida_peso_kg") else 0.0,
            estado_nutricional=record_dict["estado_nutricional"],
            hemoglobina=float(record_dict["hemoglobina"]),
            anemia=record_dict["anemia"],
            dolor=record_dict["dolor"],
            vomitos_frecuentes=record_dict["vomitos_frecuentes"],
            sangrado_digestivo=record_dict["sangrado_digestivo"],
            fatiga=record_dict["fatiga"],
            dificultad_alimentarse=record_dict["dificultad_alimentarse"],
            comorbilidades=record_dict["comorbilidades"],
            estado_funcional=record_dict.get("estado_funcional", "desconocido"),
            requiere_cirugia=record_dict["requiere_cirugia"],
            requiere_quimioterapia=record_dict["requiere_quimioterapia"],
            requiere_soporte_nutricional=record_dict["requiere_soporte_nutricional"],
            requiere_cuidados_paliativos=record_dict["requiere_cuidados_paliativos"],
            tratamiento_principal=record_dict["tratamiento_principal"]
        )

        db.add(paciente_eval)

        # Preprocesar para Modelo 1
        rec_dict_to_process = paciente_eval.to_dict()
        processed_df = preprocess_single_record(rec_dict_to_process)
        
        # Clasificar prioridad
        pred_label, p_bajo, p_medio, p_alto = predict_priority(processed_df, self.clf_meta)

        # Crear predicción asociada
        paciente_pred = PacientePrediccionPrioridad(
            evaluacion_id=paciente_eval.evaluacion_id,
            paciente_id=paciente_eval.paciente_id,
            fecha_prediccion=datetime.datetime.now(),
            prioridad_predicha=pred_label,
            probabilidad_bajo=float(p_bajo),
            probabilidad_medio=float(p_medio),
            probabilidad_alto=float(p_alto),
            umbral_aplicado=float(self.clf_meta["threshold"])
        )

        db.add(paciente_pred)
        db.flush() # Guardar cambios temporalmente para agregación posterior

        return paciente_pred

    def compute_active_patient_counts(self, db):
        """
        Obtiene los pacientes activos en la base de datos y calcula las variables
        agregadas (conteo diario) requeridas por el Modelo 2.
        """
        # Obtener evaluaciones y predicciones cruzadas
        query = db.query(PacienteEvaluacionOCR, PacientePrediccionPrioridad).join(
            PacientePrediccionPrioridad, PacienteEvaluacionOCR.evaluacion_id == PacientePrediccionPrioridad.evaluacion_id
        ).all()

        counts = {
            "pacientes_total": float(len(query)),
            "pacientes_prioridad_baja": 0.0,
            "pacientes_prioridad_media": 0.0,
            "pacientes_prioridad_alta": 0.0,
            "pacientes_estadio_0_I": 0.0,
            "pacientes_estadio_II": 0.0,
            "pacientes_estadio_III_IV": 0.0,
            "pacientes_quimioterapia": 0.0,
            "pacientes_cirugia": 0.0,
            "pacientes_paliativos": 0.0,
            "pacientes_soporte_nutricional": 0.0,
            "pacientes_con_anemia": 0.0,
            "pacientes_con_vomitos": 0.0,
            "pacientes_dolor_alto": 0.0,
            "pacientes_desnutricion": 0.0,
            "pacientes_quimio_curativa": 0.0,
            "pacientes_quimio_avanzada": 0.0,
            "pacientes_flot_estimado": 0.0,
            "pacientes_folfox_capox_estimado": 0.0,
            "pacientes_her2_positivo_estimado": 0.0,
        }

        if len(query) == 0:
            return counts

        for p_eval, p_pred in query:
            # Prioridad
            if p_pred.prioridad_predicha == "bajo":
                counts["pacientes_prioridad_baja"] += 1.0
            elif p_pred.prioridad_predicha == "medio":
                counts["pacientes_prioridad_media"] += 1.0
            elif p_pred.prioridad_predicha == "alto":
                counts["pacientes_prioridad_alta"] += 1.0

            # Estadio
            estadio = p_eval.estadio.lower().strip()
            if estadio in ["estadio 0", "estadio i", "0", "i"]:
                counts["pacientes_estadio_0_I"] += 1.0
            elif estadio in ["estadio ii", "ii"]:
                counts["pacientes_estadio_II"] += 1.0
            elif estadio in ["estadio iii", "estadio iv", "iii", "iv"]:
                counts["pacientes_estadio_III_IV"] += 1.0

            # Tratamientos binarios
            if p_eval.requiere_quimioterapia == "si":
                counts["pacientes_quimioterapia"] += 1.0
            if p_eval.requiere_cirugia == "si":
                counts["pacientes_cirugia"] += 1.0
            if p_eval.requiere_cuidados_paliativos == "si":
                counts["pacientes_paliativos"] += 1.0
            if p_eval.requiere_soporte_nutricional == "si":
                counts["pacientes_soporte_nutricional"] += 1.0

            # Síntomas
            if p_eval.anemia == "si":
                counts["pacientes_con_anemia"] += 1.0
            if p_eval.vomitos_frecuentes == "si":
                counts["pacientes_con_vomitos"] += 1.0
            if p_eval.dolor in ["alto", "severo", "2"]:
                counts["pacientes_dolor_alto"] += 1.0
            if p_eval.estado_nutricional == "desnutricion":
                counts["pacientes_desnutricion"] += 1.0

            # Quimioterapia curativa vs avanzada
            is_advanced = (estadio in ["estadio iii", "estadio iv", "iii", "iv"]) or (p_eval.metastasis == 1) or (p_eval.profundidad_tumor in ["t4"])
            if p_eval.requiere_quimioterapia == "si":
                if not is_advanced:
                    counts["pacientes_quimio_curativa"] += 1.0
                else:
                    counts["pacientes_quimio_avanzada"] += 1.0

                # Estimaciones de FLOT, FOLFOX/CAPOX y HER2-positivo
                # FLOT perioperatorio (curativo, estadios II o III y candidato a cirugía, edad < 75)
                if (estadio in ["estadio ii", "estadio iii", "ii", "iii"]) and (p_eval.requiere_cirugia == "si") and (p_eval.edad < 75):
                    counts["pacientes_flot_estimado"] += 1.0
                # FOLFOX/CAPOX (avanzado, estadios III inoperables o estadio IV o metastásicos)
                elif is_advanced:
                    counts["pacientes_folfox_capox_estimado"] += 1.0
                
                # HER2-positivo estimado: ~15% de pacientes avanzados (estable por hash de ID)
                if is_advanced:
                    # Usar el hash del paciente_id para simular positividad HER2 determinista
                    patient_hash = sum(ord(char) for char in p_eval.paciente_id)
                    if patient_hash % 6 == 0:
                        counts["pacientes_her2_positivo_estimado"] += 1.0

        return counts

    def run_demand_forecast(self, db, date_str="2026-02-10"):
        """
        Ejecuta el Modelo 2 para proyectar la demanda de los 15 insumos oncológicos,
        calculando alertas y guardando los resultados en la base de datos.
        """
        # 1. Obtener la distribución de pacientes en el albergue
        patient_counts = self.compute_active_patient_counts(db)
        
        # 2. Parsear fecha de proyección y fijar timestamp del lote
        now = datetime.datetime.now()
        pred_date = pd.to_datetime(date_str)
        semana = float(pred_date.isocalendar()[1])
        mes = float(pred_date.month)
        anio = float(pred_date.year)
        is_weekend = 1.0 if pred_date.dayofweek >= 5 else 0.0
        
        day_of_week_name = pred_date.strftime('%A').lower() # en inglés
        # Mapeo español a inglés para concordar
        dia_semana_map = {
            'monday': 'lunes', 'tuesday': 'martes', 'wednesday': 'miercoles',
            'thursday': 'jueves', 'friday': 'viernes', 'saturday': 'sabado', 'sunday': 'domingo'
        }
        day_of_week_es = dia_semana_map.get(day_of_week_name, 'domingo')
        
        # 3. Obtener stock actual de todos los insumos
        insumos_stock = db.query(InsumoInventario).all()
        if not insumos_stock:
            print("No hay insumos registrados en el inventario.")
            return []
            
        projections_inserted = []
        features_list = self.reg7_meta["features"]
        
        for insumo in insumos_stock:
            # Construir diccionario de características del insumo
            item_features = {}
            for col in features_list:
                item_features[col] = 0.0
                
            # Cargar variables de fecha
            item_features['semana'] = semana
            item_features['mes'] = mes
            item_features['anio'] = anio
            item_features['is_weekend'] = is_weekend
            
            # Cargar One-Hot de día de la semana
            day_col = f"dia_semana_{day_of_week_es}"
            if day_col in item_features:
                item_features[day_col] = 1.0
                
            # Cargar variables de censo de albergue
            capacidad = 100.0
            ocupacion = patient_counts["pacientes_total"]
            porcentaje_ocupacion = ocupacion / capacidad
            
            item_features['ocupacion_deseada'] = ocupacion
            item_features['ocupacion_albergue'] = ocupacion
            item_features['capacidad_albergue'] = capacidad
            item_features['porcentaje_ocupacion'] = porcentaje_ocupacion
            item_features['es_ocupacion_alta'] = 1.0 if porcentaje_ocupacion > 0.8 else 0.0
            
            for k, v in patient_counts.items():
                if k in item_features:
                    item_features[k] = v
                    
            item_features['ratio_pacientes_alta_prioridad'] = (
                patient_counts["pacientes_prioridad_alta"] / ocupacion if ocupacion > 0 else 0.0
            )
            
            # Cargar variables del insumo (stock y consumo)
            item_features['stock_actual'] = insumo.stock_actual
            item_features['stock_minimo'] = insumo.stock_minimo
            item_features['ingresos_stock'] = 0.0 # Proyección sin ingresos el día de hoy
            item_features['salidas_stock'] = insumo.consumo_diario
            item_features['stock_cierre'] = insumo.stock_actual - insumo.consumo_diario
            item_features['consumo_diario'] = insumo.consumo_diario
            item_features['consumo_lag_1'] = insumo.consumo_lag_1
            item_features['consumo_lag_7'] = insumo.consumo_lag_7
            item_features['consumo_promedio_7d'] = insumo.consumo_promedio_7d
            item_features['consumo_promedio_14d'] = insumo.consumo_promedio_14d
            item_features['ratio_stock_minimo'] = (
                insumo.stock_actual / insumo.stock_minimo if insumo.stock_minimo > 0 else 0.0
            )
            item_features['dias_cobertura'] = (
                insumo.stock_actual / insumo.consumo_diario if insumo.consumo_diario > 0 else 999.0
            )
            item_features['neto_movimiento'] = -insumo.consumo_diario
            item_features['variabilidad_consumo'] = insumo.variabilidad_consumo
            
            # Cargar variables One-Hot del Insumo
            # 1. item_id
            item_id_col = f"item_id_{insumo.item_id}"
            if item_id_col in item_features:
                item_features[item_id_col] = 1.0
                
            # 2. categoria_item
            cat_col = f"categoria_item_{insumo.categoria_item}"
            if cat_col in item_features:
                item_features[cat_col] = 1.0
                
            # 3. tipo_stock
            tipo_col = f"tipo_stock_{insumo.tipo_stock}"
            if tipo_col in item_features:
                item_features[tipo_col] = 1.0
                
            # 4. unidad_medida
            unidad_col = f"unidad_medida_{insumo.unidad_medida}"
            if unidad_col in item_features:
                item_features[unidad_col] = 1.0
                
            # Convertir a DataFrame y alinear columnas exactamente
            df_features = pd.DataFrame([item_features])[features_list]
            
            # Ejecutar Inferencia de Demanda
            pred_7d_raw = self.reg7_meta["model"].predict(df_features)[0]
            pred_14d_raw = self.reg14_meta["model"].predict(df_features)[0]
            
            # Regla de Negocio: Nunca devolver valores negativos de consumo o demanda
            demanda_7d = max(0.0, float(pred_7d_raw))
            demanda_14d = max(0.0, float(pred_14d_raw))
            
            # Cobertura en días reales
            dias_cobertura = (
                insumo.stock_actual / insumo.consumo_diario if insumo.consumo_diario > 0 else 99.0
            )
            
            # Evaluar stock crítico: Si Stock Actual < Demanda Proyectada -> Cambiar a SÍ (1)
            # Evaluamos stock crítico a 7 días y a 14 días
            stock_critico_7d = 1 if insumo.stock_actual < demanda_7d else 0
            stock_critico_14d = 1 if insumo.stock_actual < demanda_14d else 0
            # Stock crítico global (cualquiera de los horizontes de compra)
            stock_critico = 1 if (insumo.stock_actual < demanda_7d or insumo.stock_actual < demanda_14d) else 0
            
            # Crear registro de predicción
            pred_orm = InsumoPrediccion(
                item_id=insumo.item_id,
                item_nombre=insumo.item_nombre,
                fecha_prediccion=now,
                stock_actual=insumo.stock_actual,
                demanda_proyectada_7d=demanda_7d,
                demanda_proyectada_14d=demanda_14d,
                dias_cobertura=dias_cobertura,
                stock_critico=stock_critico,
                stock_critico_7d=stock_critico_7d,
                stock_critico_14d=stock_critico_14d
            )
            db.add(pred_orm)
            projections_inserted.append(pred_orm)
            
        db.commit()
        return projections_inserted

    def run_pipeline_for_new_patient(self, raw_record_dict, date_str="2026-02-10"):
        """
        Ejecuta el flujo secuencial completo:
        1. Ingesta y clasifica al nuevo paciente.
        2. Recalcula el censo de pacientes.
        3. Realiza la predicción de demanda de insumos clínicos.
        """
        db = SessionLocal()
        try:
            # 1. Inferencia del paciente
            print(f"Procesando paciente {raw_record_dict.get('paciente_id')}...")
            paciente_pred = self.process_and_classify_patient(db, raw_record_dict)
            print(f"Paciente clasificado con prioridad: {paciente_pred.prioridad_predicha.upper()}")
            
            # 2. Correr proyecciones de stock
            print("Ejecutando predicciones de demanda actualizadas...")
            projections = self.run_demand_forecast(db, date_str)
            print(f"Pronósticos actualizados para los {len(projections)} insumos clínicos.")
            
            return paciente_pred, projections
        except Exception as e:
            db.rollback()
            print(f"Error en el pipeline secuencial: {e}")
            raise e
        finally:
            db.close()

if __name__ == "__main__":
    pipeline = AldimiPipeline()
    # Inferencia de prueba
    db = SessionLocal()
    try:
        projections = pipeline.run_demand_forecast(db)
        print(f"Se generaron {len(projections)} proyecciones de prueba.")
        for p in projections[:3]:
            print(f"Insumo: {p.item_nombre} | Stock: {p.stock_actual} | Demanda 7d: {p.demanda_proyectada_7d:.2f} | Crítico: {p.stock_critico}")
    finally:
        db.close()
