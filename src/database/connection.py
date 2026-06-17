from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from src.config import DATABASE_URI
from src.database.models import Base

# Crear motor de base de datos
engine = create_engine(
    DATABASE_URI,
    connect_args={"check_same_thread": False} if DATABASE_URI.startswith("sqlite") else {}
)

# Configurar fábrica de sesiones
session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
SessionLocal = scoped_session(session_factory)

def init_db():
    """Inicializa la base de datos y crea las tablas si no existen."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Generador de sesión de base de datos para usar con context managers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
