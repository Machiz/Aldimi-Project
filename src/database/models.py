from sqlalchemy import Column, String, Integer, Float, Date, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PacienteEvaluacionOCR(Base):
    __tablename__ = "paciente_evaluaciones_ocr"

    evaluacion_id = Column(String(50), primary_key=True)
    paciente_id = Column(String(50), nullable=False)
    fecha_evaluacion = Column(Date, nullable=False)
    fecha_diagnostico = Column(Date, nullable=False)
    fecha_ingreso_aldimi = Column(Date, nullable=False)
    edad = Column(Integer, nullable=False)
    sexo = Column(String(10), nullable=False)
    region_procedencia = Column(String(50), nullable=False)
    zona_procedencia = Column(String(20), nullable=False)
    viaja_desde_provincia = Column(String(2), nullable=False)
    acompanante = Column(String(2), nullable=False)
    nivel_vulnerabilidad = Column(String(15), nullable=False)
    dificultad_acceso_salud = Column(String(15), nullable=False)
    tipo_histologico = Column(String(50), nullable=False)
    estadio = Column(String(15), nullable=False)
    grado_histologico = Column(String(15), nullable=True)
    profundidad_tumor = Column(String(10), nullable=False)
    ganglios_afectados = Column(Integer, nullable=False)
    metastasis = Column(Integer, nullable=False)
    sitio_metastasis = Column(String(50), nullable=True)
    perdida_peso_kg = Column(Float, nullable=True)
    estado_nutricional = Column(String(20), nullable=False)
    hemoglobina = Column(Float, nullable=True)
    anemia = Column(String(2), nullable=False)
    dolor = Column(String(15), nullable=False)
    vomitos_frecuentes = Column(String(2), nullable=False)
    sangrado_digestivo = Column(String(2), nullable=False)
    fatiga = Column(String(2), nullable=False)
    dificultad_alimentarse = Column(String(2), nullable=False)
    comorbilidades = Column(String(2), nullable=False)
    estado_funcional = Column(String(25), nullable=True)
    requiere_cirugia = Column(String(2), nullable=False)
    requiere_quimioterapia = Column(String(2), nullable=False)
    requiere_soporte_nutricional = Column(String(2), nullable=False)
    requiere_cuidados_paliativos = Column(String(2), nullable=False)
    tratamiento_principal = Column(String(50), nullable=False)

    def to_dict(self):
        return {col.name: getattr(self, col.name) for col in self.__table__.columns}


class PacientePrediccionPrioridad(Base):
    __tablename__ = "paciente_predicciones_prioridad"

    evaluacion_id = Column(String(50), ForeignKey("paciente_evaluaciones_ocr.evaluacion_id"), primary_key=True)
    paciente_id = Column(String(50), nullable=False)
    fecha_prediccion = Column(DateTime, nullable=False)
    prioridad_predicha = Column(String(10), nullable=False)
    probabilidad_bajo = Column(Float, nullable=False)
    probabilidad_medio = Column(Float, nullable=False)
    probabilidad_alto = Column(Float, nullable=False)
    umbral_aplicado = Column(Float, nullable=False)

    def to_dict(self):
        return {col.name: getattr(self, col.name) for col in self.__table__.columns}
