import unittest
import os
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

from src.database.models import Base, PacienteEvaluacionOCR, PacientePrediccionPrioridad, InsumoInventario, InsumoPrediccion
from src.pipeline.data_pipeline import AldimiPipeline

class TestAldimiPipeline(unittest.TestCase):
    def setUp(self):
        # Configurar base de datos SQLite en memoria
        self.engine = create_engine("sqlite:///:memory:")
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        Base.metadata.create_all(self.engine)
        
        # Inicializar el pipeline
        self.pipeline = AldimiPipeline()
        
        # Poblar insumo de prueba para la simulación
        self.test_insumo = InsumoInventario(
            item_id="ITEM001",
            item_nombre="fluorouracilo_5fu",
            categoria_item="fluoropirimidina_iv",
            tipo_stock="antineoplasico",
            unidad_medida="viales",
            stock_actual=100.0,
            stock_minimo=50.0,
            consumo_diario=10.0,
            consumo_lag_1=10.0,
            consumo_lag_7=10.0,
            consumo_promedio_7d=10.0,
            consumo_promedio_14d=10.0,
            variabilidad_consumo=0.0
        )
        self.session.add(self.test_insumo)
        self.session.commit()

        # Mock de un registro de paciente
        self.valid_patient_dict = {
            "evaluacion_id": "E-VAL-999",
            "paciente_id": "P-PAC-999",
            "fecha_evaluacion": "2026-02-10",
            "fecha_diagnostico": "2025-12-01",
            "fecha_ingreso_aldimi": "2026-01-15",
            "edad": 65,
            "sexo": "masculino",
            "region_procedencia": "ancash",
            "zona_procedencia": "rural",
            "viaja_desde_provincia": "si",
            "acompanante": "si",
            "nivel_vulnerabilidad": "alto",
            "dificultad_acceso_salud": "alta",
            "tipo_histologico": "adenocarcinoma",
            "estadio": "Estadio III",
            "grado_histologico": "g3",
            "profundidad_tumor": "T3",
            "ganglios_afectados": 4,
            "metastasis": 0,
            "sitio_metastasis": "ninguno",
            "perdida_peso_kg": 6.2,
            "estado_nutricional": "riesgo desnutricion",
            "hemoglobina": 10.2,
            "anemia": "si",
            "dolor": "moderado",
            "vomitos_frecuentes": "no",
            "sangrado_digestivo": "no",
            "fatiga": "si",
            "dificultad_alimentarse": "no",
            "comorbilidades": "si",
            "estado_funcional": "ambulatorio",
            "requiere_cirugia": "si",
            "requiere_quimioterapia": "si",
            "requiere_soporte_nutricional": "si",
            "requiere_cuidados_paliativos": "no",
            "tratamiento_principal": "quimioterapia"
        }

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_validation_invalid_age(self):
        record = self.valid_patient_dict.copy()
        record["edad"] = 150
        is_valid, errors, _ = self.pipeline.validate_patient_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("edad" in err.lower() for err in errors))

    def test_validation_chronology_error(self):
        record = self.valid_patient_dict.copy()
        # Fecha de evaluación previa al diagnóstico (cronología incorrecta)
        record["fecha_evaluacion"] = "2025-10-01"
        is_valid, errors, _ = self.pipeline.validate_patient_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("cronología" in err.lower() or "evaluación" in err.lower() for err in errors))

    def test_validation_out_of_range_hemoglobin(self):
        record = self.valid_patient_dict.copy()
        record["hemoglobina"] = 40.0 # Fuera de rango [3.0 - 25.0]
        is_valid, errors, record_clean = self.pipeline.validate_patient_record(record)
        # Debe pasar porque la regla es imputar por la mediana (12.0) sin lanzar error fatal
        self.assertTrue(is_valid)
        self.assertEqual(record_clean["hemoglobina"], 12.0)

    def test_full_pipeline_run(self):
        # 1. Ingesta del paciente de prueba
        record = self.valid_patient_dict.copy()
        is_valid, errors, record = self.pipeline.validate_patient_record(record)
        self.assertTrue(is_valid)
        
        # Registrar y procesar usando una transacción local
        pred = self.pipeline.process_and_classify_patient(self.session, record)
        self.session.commit()
        
        # Validar inserción
        saved_patient = self.session.query(PacienteEvaluacionOCR).filter_by(evaluacion_id="e-val-999").first()
        self.assertIsNotNone(saved_patient)
        self.assertEqual(saved_patient.paciente_id, "p-pac-999")
        
        saved_pred = self.session.query(PacientePrediccionPrioridad).filter_by(evaluacion_id="e-val-999").first()
        self.assertIsNotNone(saved_pred)
        self.assertIn(saved_pred.prioridad_predicha, ["bajo", "medio", "alto"])
        
        # 2. Ejecutar pronóstico de demanda
        projections = self.pipeline.run_demand_forecast(self.session, date_str="2026-02-10")
        
        # Validar predicciones de insumos
        self.assertEqual(len(projections), 1)
        saved_proj = self.session.query(InsumoPrediccion).filter_by(item_id="ITEM001").first()
        self.assertIsNotNone(saved_proj)
        self.assertGreaterEqual(saved_proj.demanda_proyectada_7d, 0.0)
        self.assertGreaterEqual(saved_proj.demanda_proyectada_14d, 0.0)
        self.assertIn(saved_proj.stock_critico, [0, 1])

if __name__ == "__main__":
    unittest.main()
