# Aldimi-Project
# Guía de Configuración del Proyecto

Sigue estos pasos en orden para configurar el entorno virtual, preparar los datos y ejecutar la aplicación. Asegúrate de estar ubicado en la **raíz del proyecto** (`Aldimi-Project`) en tu terminal.

## 1. Crear y Activar el Entorno Virtual

Borra cualquier entorno previo defectuoso y crea uno nuevo en la raíz del proyecto:

```bash
# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno virtual (en Linux/Mac)
source venv/bin/activate
En (Windows)
venv\Scripts\activate
```

## 2. Instalar Dependencias

Instala todas las librerías necesarias con las versiones de compatibilidad correctas:

```bash
pip install -r requirements.txt
```

## 3. Inicializar la Base de Datos

Antes de correr el pipeline o la app, debes preparar la base de datos local:

```bash
python src/database/initialize_data.py
```

## 4. Ejecutar el Pipeline de Datos

Carga, procesa los datos y ejecuta el flujo de entrenamiento/predicción:

```bash
python src/pipeline/data_pipeline.py
```

## 5. Ejecutar la Aplicación Web (Streamlit)

Finalmente, levanta la interfaz gráfica del proyecto para interactuar con ella en tu navegador:

```bash
streamlit run src/app.py
```
