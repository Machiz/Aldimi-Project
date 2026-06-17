import os
from pathlib import Path

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
DATASET_1_PATH = DATA_DIR / "evaluaciones_pacientes_cancer_gastrico_sintetico.csv"
DATASET_2_PATH = DATA_DIR / "insumos_cancer_final.csv"

# Directorio de modelos guardados
MODELS_SAVE_DIR = BASE_DIR / "src" / "models" / "saved_models"
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

# Configuración de base de datos
DB_FILE = DATA_DIR / "aldimi_shared.db"
DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{DB_FILE}")

# Configuración de hiperparámetros y reproducibilidad
RANDOM_STATE = 42
