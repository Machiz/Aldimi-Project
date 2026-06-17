import unittest
import os
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, PacienteEvaluacionOCR, PacientePrediccionPrioridad
from src.database.connection import get_db

class TestDatabaseIntegration(unittest.TestCase):
    def setUp(self):
        # Configurar una base de datos SQLite en memoria para pruebas
        self.engine = create_engine("sqlite:///:memory:")
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        Base.metadata.create_all(self.engine)

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_insert_and_retrieve_ocr_record(self):
        # Crear un registro simulado de OCR
        record = PacienteEvaluacionOCR(
            evaluacion_id="E-TEST-001",
            paciente_id="P-TEST-001",
            fecha_evaluacion=datetime.date(2026, 6, 17),
            fecha_diagnostico=datetime.date(2026, 1, 15),
            fecha_ingreso_aldimi=datetime.date(2026, 2, 10),
            edad=58,
            sexo="masculino",
            region_procedencia="Cusco",
            zona_procedencia="rural",
            viaja_desde_provincia="si",
            acompanante="si",
            nivel_vulnerabilidad="alto",
            dificultad_acceso_salud="media",
            tipo_histologico="adenocarcinoma",
            estadio="Estadio II",
            grado_histologico="G2",
            profundidad_tumor="T3",
            ganglios_afectados=2,
            metastasis=0,
            sitio_metastasis="ninguno",
            perdida_peso_kg=5.5,
            estado_nutricional="riesgo",
            hemoglobina=10.5,
            anemia="si",
            dolor="medio",
            vomitos_frecuentes="no",
            sangrado_digestivo="no",
            fatiga="si",
            dificultad_alimentarse="no",
            comorbilidades="una",
            estado_funcional="limitado",
            requiere_cirugia="si",
            requiere_quimioterapia="si",
            requiere_soporte_nutricional="no",
            requiere_cuidados_paliativos="no",
            tratamiento_principal="quimioterapia"
        )
        
        self.session.add(record)
        self.session.commit()
        
        # Recuperar el registro y validar
        retrieved = self.session.query(PacienteEvaluacionOCR).filter_by(evaluacion_id="E-TEST-001").first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.paciente_id, "P-TEST-001")
        self.assertEqual(retrieved.edad, 58)
        self.assertEqual(retrieved.estadio, "Estadio II")

    def test_insert_prediction_record(self):
        # Primero crear la evaluación correspondiente por restricción FK
        eval_record = PacienteEvaluacionOCR(
            evaluacion_id="E-TEST-002",
            paciente_id="P-TEST-002",
            fecha_evaluacion=datetime.date(2026, 6, 17),
            fecha_diagnostico=datetime.date(2026, 1, 15),
            fecha_ingreso_aldimi=datetime.date(2026, 2, 10),
            edad=45,
            sexo="femenino",
            region_procedencia="Lima",
            zona_procedencia="urbana",
            viaja_desde_provincia="no",
            acompanante="si",
            nivel_vulnerabilidad="bajo",
            dificultad_acceso_salud="baja",
            tipo_histologico="adenocarcinoma",
            estadio="Estadio I",
            grado_histologico="G1",
            profundidad_tumor="T1",
            ganglios_afectados=0,
            metastasis=0,
            sitio_metastasis="ninguno",
            perdida_peso_kg=1.0,
            estado_nutricional="adecuado",
            hemoglobina=13.0,
            anemia="no",
            dolor="bajo",
            vomitos_frecuentes="no",
            sangrado_digestivo="no",
            fatiga="no",
            dificultad_alimentarse="no",
            comorbilidades="ninguna",
            estado_funcional="bueno",
            requiere_cirugia="si",
            requiere_quimioterapia="no",
            requiere_soporte_nutricional="no",
            requiere_cuidados_paliativos="no",
            tratamiento_principal="cirugia"
        )
        self.session.add(eval_record)
        self.session.commit()

        # Crear predicción asociada
        pred = PacientePrediccionPrioridad(
            evaluacion_id="E-TEST-002",
            paciente_id="P-TEST-002",
            fecha_prediccion=datetime.datetime.now(),
            prioridad_predicha="bajo",
            probabilidad_bajo=0.85,
            probabilidad_medio=0.10,
            probabilidad_alto=0.05,
            umbral_aplicado=0.35
        )
        self.session.add(pred)
        self.session.commit()

        # Recuperar y verificar
        retrieved_pred = self.session.query(PacientePrediccionPrioridad).filter_by(evaluacion_id="E-TEST-002").first()
        self.assertIsNotNone(retrieved_pred)
        self.assertEqual(retrieved_pred.prioridad_predicha, "bajo")
        self.assertEqual(retrieved_pred.probabilidad_bajo, 0.85)

if __name__ == "__main__":
    unittest.main()
