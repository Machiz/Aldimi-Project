# Contrato de Interfaz de Datos - OCR a ML Pipeline

Este documento establece el contrato de datos formal entre el módulo de Inteligencia Artificial (OCR / extracción de texto) y el módulo de Machine Learning de la fundación ALDIMI. Define el esquema, tipos de datos, valores permitidos y validaciones para garantizar la interoperabilidad de los sistemas.

---

## 1. Detalles de Integración (Base de Datos)

El módulo de OCR depositará la información extraída en una tabla de la base de datos relacional común llamada `paciente_evaluaciones_ocr`.
El pipeline de Machine Learning leerá de esta tabla, procesará la información y registrará las predicciones en la tabla `paciente_predicciones_prioridad`.

---

## 2. Especificación de Columnas: Tabla `paciente_evaluaciones_ocr`

| Nombre de Columna | Tipo de Dato (SQL) | Tipo de Dato (Pandas) | Formato / Valores Permitidos | Descripción | Obligatorio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `evaluacion_id` | VARCHAR(50) | object | Cadena única (UUID o numérico) | Identificador único de la evaluación clínica | Sí |
| `paciente_id` | VARCHAR(50) | object | Cadena (ej. "P-1001") | Identificador único del paciente | Sí |
| `fecha_evaluacion` | DATE | datetime64[ns] | `YYYY-MM-DD` | Fecha en que se realiza la evaluación médica | Sí |
| `fecha_diagnostico` | DATE | datetime64[ns] | `YYYY-MM-DD` | Fecha del diagnóstico original del cáncer | Sí |
| `fecha_ingreso_aldimi` | DATE | datetime64[ns] | `YYYY-MM-DD` | Fecha en la que el paciente ingresa al albergue | Sí |
| `edad` | INTEGER | int64 | `[0 - 120]` | Edad en años cumplidos del paciente | Sí |
| `sexo` | VARCHAR(10) | object | `femenino`, `masculino` | Sexo biológico del paciente | Sí |
| `region_procedencia` | VARCHAR(50) | object | Nombres de regiones (ej. "Lima", "Cusco") | Región de procedencia del paciente | Sí |
| `zona_procedencia` | VARCHAR(20) | object | `urbana`, `rural` | Zona habitacional de origen | Sí |
| `viaja_desde_provincia`| VARCHAR(2) | object | `si`, `no` | Flag de traslado interprovincial | Sí |
| `acompanante` | VARCHAR(2) | object | `si`, `no` | Indica si viaja con un acompañante | Sí |
| `nivel_vulnerabilidad`| VARCHAR(15) | object | `bajo`, `medio`, `alto`, `muy alto` | Clasificación de vulnerabilidad socioeconómica | Sí |
| `dificultad_acceso_salud`| VARCHAR(15) | object| `baja`, `media`, `alta` | Nivel de barreras para atención de salud | Sí |
| `tipo_histologico` | VARCHAR(50) | object | Categórico (ej. "adenocarcinoma") | Tipo biológico de tejido canceroso | Sí |
| `estadio` | VARCHAR(15) | object | `Estadio 0`, `Estadio I`, `Estadio II`, `Estadio III`, `Estadio IV` | Estadio de gravedad del cáncer gástrico | Sí |
| `grado_histologico` | VARCHAR(15) | object | `bien diferenciado`, `moderadamente diferenciado`, `poco diferenciado`, `desconocido` | Grado de diferenciación celular | No |
| `profundidad_tumor` | VARCHAR(10) | object | `T1`, `T2`, `T3`, `T4` | Extensión primaria del tumor (TNM) | Sí |
| `ganglios_afectados` | INTEGER | int64 | `[0 - 100]` | Cantidad de ganglios linfáticos afectados | Sí |
| `metastasis` | INTEGER | int64 | `0` (No), `1` (Sí) | Flag de presencia de metástasis a distancia | Sí |
| `sitio_metastasis` | VARCHAR(50) | object | `ninguno`, `hepática`, `peritoneal`, `pulmonar`, `ósea`, `desconocido` | Localización anatómica de la metástasis | No |
| `perdida_peso_kg` | FLOAT | float64 | `[0.0 - 50.0]` | Pérdida de peso corporal reciente del paciente | No |
| `estado_nutricional` | VARCHAR(20) | object | `bueno`, `riesgo desnutricion`, `desnutricion` | Diagnóstico de estado nutricional | Sí |
| `hemoglobina` | FLOAT | float64 | `[3.0 - 25.0]` (g/dL) | Nivel medido de hemoglobina | No |
| `anemia` | VARCHAR(2) | object | `si`, `no` | Flag clínico de anemia diagnóstica | Sí |
| `dolor` | VARCHAR(15) | object | `ausente`, `leve`, `moderado`, `severo` | Nivel de dolor del paciente | Sí |
| `vomitos_frecuentes` | VARCHAR(2) | object | `si`, `no` | Flag de náuseas/vómitos severos | Sí |
| `sangrado_digestivo` | VARCHAR(2) | object | `si`, `no` | Presencia de hemorragia digestiva alta/baja | Sí |
| `fatiga` | VARCHAR(2) | object | `si`, `no` | Fatiga o astenia extrema | Sí |
| `dificultad_alimentarse`| VARCHAR(2) | object| `si`, `no` | Disfagia o problemas de ingesta oral | Sí |
| `comorbilidades` | VARCHAR(2) | object | `si`, `no` | Presencia de otras patologías crónicas | Sí |
| `estado_funcional` | VARCHAR(25) | object | `totalmente activo`, `limitado`, `ambulatorio`, `dependiente en cama`, `desconocido` | Índice de funcionalidad (tipo ECOG) | No |
| `requiere_cirugia` | VARCHAR(2) | object | `si`, `no` | Indicación quirúrgica | Sí |
| `requiere_quimioterapia`| VARCHAR(2)| object | `si`, `no` | Indicación de tratamiento quimioterapéutico | Sí |
| `requiere_soporte_nutricional`| VARCHAR(2)| object| `si`, `no` | Indicación de nutrición enteral/parenteral | Sí |
| `requiere_cuidados_paliativos`| VARCHAR(2)| object| `si`, `no` | Indicación de soporte paliativo | Sí |
| `tratamiento_principal`| VARCHAR(50)| object| Categórico (ej. "quimioterapia", "paliativo") | Plan de tratamiento central establecido | Sí |

---

## 3. Especificación de Columnas: Tabla `paciente_predicciones_prioridad`

Esta tabla es poblada de manera automática por el pipeline de Machine Learning tras consumir los datos del OCR.

| Nombre de Columna | Tipo de Dato (SQL) | Tipo de Dato (Pandas) | Valores Permitidos | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `evaluacion_id` | VARCHAR(50) | object | Cadena | Llave foránea que referencia a la evaluación de origen |
| `paciente_id` | VARCHAR(50) | object | Cadena | Identificador único del paciente |
| `fecha_prediccion` | TIMESTAMP | datetime64[ns] | `YYYY-MM-DD HH:MM:SS` | Fecha y hora en que se computó la inferencia |
| `prioridad_predicha` | VARCHAR(10) | object | `bajo`, `medio`, `alto` | Nivel de prioridad clínica asignado por el Modelo 1 |
| `probabilidad_bajo` | FLOAT | float64 | `[0.0 - 1.0]` | Probabilidad calculada para la clase baja |
| `probabilidad_medio` | FLOAT | float64 | `[0.0 - 1.0]` | Probabilidad calculada para la clase media |
| `probabilidad_alto` | FLOAT | float64 | `[0.0 - 1.0]` | Probabilidad calculada para la clase alta |
| `umbral_aplicado` | FLOAT | float64 | `[0.0 - 1.0]` | Umbral crítico de la clase alta utilizado en la regla |

---

## 4. Reglas de Validación de Datos (Pre-ingesta)

El pipeline de Machine Learning ejecutará de forma automática las siguientes validaciones sobre la tabla de OCR antes de correr los modelos predictivos:

1. **Validación Cronológica**: La `fecha_evaluacion` debe ser posterior o igual a la `fecha_diagnostico` y `fecha_ingreso_aldimi`.
2. **Control de Rango Edad**: El valor de `edad` debe estar en el intervalo `[0 - 120]`.
3. **Control de Rango Hemoglobina**: La `hemoglobina` debe estar en `[3.0 - 25.0]`. Cualquier valor fuera se marcará como atípico y se le aplicará imputación por la mediana.
4. **Tratamiento de Nulos**: Las columnas marcadas como no obligatorias que presenten valores nulos (`NULL`) serán imputadas con la categoría `"desconocido"` para variables categóricas o con la mediana de entrenamiento para variables numéricas.
5. **Codificación Categórica Estricta**: Todas las variables categóricas de tipo string serán convertidas a minúsculas y limpiadas de espacios iniciales o finales antes de realizar el LabelEncoding.
