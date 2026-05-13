# Análisis Exploratorio de Datos (EDA) para Machine Learning - Evaluaciones de Pacientes

Este documento consolida los hallazgos, metodologías y procesos llevados a cabo en el notebook `EDA_ML_evaluaciones_paciente.ipynb`. Su propósito es brindar una visión general sobre la exploración y preparación de los datos sintéticos de la fundación ALDIMI, garantizando que estén listos para el entrenamiento de futuros modelos predictivos.

---

## 1. Descripción y Diccionario de Datos

La información base proviene del archivo `evaluaciones_pacientes_cancer_gastrico_sintetico.csv`. Este dataset contiene **5,000 registros (evaluaciones)** correspondientes a **2,000 pacientes únicos**. Al existir múltiples evaluaciones por paciente, nos encontramos ante un dataset de naturaleza longitudinal.

El dataset cuenta con **41 columnas**, agrupadas de forma lógica para el análisis clínico y predictivo:

| Grupo de Variables | Columnas Incluidas | Descripción General |
| :--- | :--- | :--- |
| **Identificación** | `evaluacion_id`, `paciente_id` | Códigos únicos para rastrear cada registro individual y vincular múltiples evaluaciones a un solo paciente. |
| **Temporales** | `fecha_diagnostico`, `fecha_ingreso_aldimi`, `fecha_evaluacion`, `dias_desde_diagnostico`, `dias_en_albergue` | Fechas clave en la trayectoria del paciente y variables derivadas de tiempo que miden el tiempo de evolución o estancia. |
| **Demográficas y sociales** | `edad`, `grupo_edad`, `sexo`, `region_procedencia`, `zona_procedencia`, `viaja_desde_provincia`, `acompanante`, `nivel_vulnerabilidad`, `dificultad_acceso_salud` | Perfil sociodemográfico, origen del paciente y condiciones de acceso/soporte. |
| **Clínico-oncológicas** | `tipo_cancer`, `tipo_histologico`, `estadio`, `grado_histologico`, `profundidad_tumor`, `ganglios_afectados`, `metastasis`, `sitio_metastasis` | Características propias de la enfermedad oncológica, su severidad biológica y extensión anatómica. |
| **Síntomas y estado del paciente** | `perdida_peso_kg`, `estado_nutricional`, `hemoglobina`, `anemia`, `dolor`, `vomitos_frecuentes`, `sangrado_digestivo`, `fatiga`, `dificultad_alimentarse`, `comorbilidades`, `estado_funcional` | Indicadores del impacto físico del cáncer y el nivel de dependencia o deterioro funcional. |
| **Tratamiento** | `requiere_cirugia`, `requiere_quimioterapia`, `requiere_soporte_nutricional`, `requiere_cuidados_paliativos`, `tratamiento_principal` | Intervenciones terapéuticas pasadas, presentes o proyectadas. |
| **Variable Objetivo** | `nivel_prioridad` | Etiqueta objetivo (Target) a predecir, clasificada en Bajo, Medio o Alto. |

---

## 2. Pipeline de Preprocesamiento de Datos

Previo al modelado, la data fue sometida a una estricta auditoría y limpieza para asegurar la robustez de los futuros algoritmos:

1.  **Validación de Duplicados e Identificadores:** Se confirmó que no existen filas completamente duplicadas. Además, la columna `evaluacion_id` cuenta con 5,000 valores únicos, mientras que `paciente_id` tiene 2,000 (confirmando el promedio de 2.5 evaluaciones por paciente).
2.  **Verificación de Formatación de Texto:** A través de expresiones regulares, se barrió todo el dataset categórico buscando y descartando la existencia de espacios vacíos "ocultos" al inicio o final de las cadenas de texto que pudieran generar categorías fantasmas.
3.  **Validación de Rangos Numéricos Lógicos:** Se verificó matemáticamente que no existieran incongruencias biológicas, confirmando que no hay edades negativas ni mayores a 100 años, que no hay días de estancia negativos, que no hay pérdida de peso negativa y que los niveles de hemoglobina se encuentren entre 5 y 20 g/dL.
4.  **Transformación y Auditoría Temporal:** 
    *   Las columnas de fechas pasaron de ser tipo `object` a `datetime64`.
    *   Se validó cronológicamente que ninguna fecha de evaluación (visita) o ingreso ocurriera antes de la fecha de diagnóstico.
    *   Las variables `dias_desde_diagnostico` y `dias_en_albergue` fueron recalculadas manualmente para comparar con las del sistema, reportando cero diferencias (los datos son fiables).
5.  **Análisis y Tratamiento de Valores Nulos:** Solo 429 filas (el 8.58% del dataset) presentaron algún valor nulo. Los nulos se concentraron en apenas 5 variables clínicas:
    *   `hemoglobina` (2.80%), `perdida_peso_kg` (2.10%), `grado_histologico` (1.46%), `sitio_metastasis` (1.22%), `estado_funcional` (1.20%).
    *   Al ser porcentajes minúsculos y estar distribuidos de manera uniforme entre los tres niveles de prioridad (no sesgan una sola clase), se decidió **mantener las columnas**, imputando categoricamente los datos como "desconocido" o "nulo" para no perder información vital del resto del paciente.
6.  **Eliminación de Variables Constantes:** Se comprobó que `tipo_cancer` posee un 100% de cardinalidad igual a "cancer_gastrico". Dado que tiene nula varianza, **es imperativo eliminarla** antes del entrenamiento.

---

## 3. Resultados del Análisis Exploratorio (EDA)

El análisis univariado reveló la "fotografía" de la población atendida:

*   **Perfil Demográfico:** La edad media es de 58.56 años (pico máximo de 90 años), donde casi el 58.62% recae en la categoría "adulto mayor". Hay un equilibrio en género (52% masculino, 48% femenino). De manera destacable, el **87.9% viaja desde provincia**, y más del 75% califica en niveles de vulnerabilidad "Alto" o "Muy Alto".
*   **Perfil Clínico-Oncológico:** Se observó una dispersión de estadios, dominando el Estadio II (32.82%), seguido del Estadio III (25.96%), I (23.22%), IV (11.06%) y 0 (6.94%). Solo el 11.06% presenta metástasis, mayormente hepática y peritoneal.
*   **Perfil Sintomático:** El impacto biológico es claro: el **67.56% de los registros presentan anemia**, el 69.28% demanda soporte nutricional, y un amplio grupo tiene un estado funcional limitado o dependiente. La pérdida de peso promedia los 5.75 kg.
*   **Tratamientos y Balance de Clases:** El 59.66% requerirá cirugía en algún momento y el 48.20% quimioterapia. La variable a predecir (`nivel_prioridad`) goza de un **equilibrio de clases excelente** para ML: Bajo (30.4%), Medio (35.8%), y Alto (33.8%).

---

## 4. Selección de Características (Feature Selection)

Esta etapa es el núcleo del modelado predictivo. Mediante cruces bivariados e inferencia estadística, se midió la asociación de cada variable independiente con la dependiente (`nivel_prioridad`) para saber a qué prestarle atención.

### A. Variables Categóricas (Prueba V de Cramér)
La V de Cramér mide la fuerza de asociación entre variables categóricas (de 0 a 1).
**Top Predictores Categóricos:**
1.  **`estadio` (0.856):** Es la variable más poderosa de todo el dataset. Prácticamente todos los Estadios 0 o I son prioridad Baja, el Estadio II es Media, y los Estadios III o IV dictan prioridad Alta (Estadio IV tiene un 100% de correlación con prioridad alta).
2.  **`profundidad_tumor` (0.650):** Tumores en mucosa o submucosa son prioridad baja; invasiones en serosa o enfermedad diseminada son alta.
3.  **`requiere_quimioterapia` (0.551):** Más del 52% de quienes la requieren saltan a prioridad alta.
4.  **`tratamiento_principal` (0.517):** Casos paliativos son abrumadoramente de prioridad alta (casi 80%).
5.  **`anemia` (0.508) y `estado_funcional` (0.499):** Un estado funcional dependiente garantiza un 88.6% de probabilidad de ser prioridad Alta.

### B. Variables Numéricas (Prueba Eta Cuadrado)
El Eta Cuadrado (η²) mide la proporción de la varianza en una variable numérica explicada por los grupos categóricos del target.
**Top Predictores Continuos:**
1.  **`ganglios_afectados` (0.561):** Fuerte distinción entre clases. En promedio, pacientes de prioridad baja tienen 0.19 ganglios; los de media, 2.52; y los de alta prioridad, 7.84 ganglios afectados.
2.  **`perdida_peso_kg` (0.439):** La pérdida de peso crece escalonadamente a medida que sube la prioridad (3.19kg en Baja vs 8.15kg en Alta).
3.  **`hemoglobina` (0.318):** Presenta una asociación inversa; la hemoglobina disminuye bruscamente en niveles altos de prioridad.

### C. Variables de Bajo Poder Predictivo (Candidatas a descarte)
Contrario a la intuición, las variables sociales no están moviendo la aguja clínica. El nivel de vulnerabilidad social (0.050), si viaja de provincia (0.048), la dificultad de acceso a la salud (0.047) o el grupo de edad y sexo no superan el 0.05 en Cramér. De la misma manera, numéricamente la edad (0.000) y los días en el albergue (0.005) son "ruido" estadístico para predecir si un paciente está biológicamente grave.

---

## 5. Conclusiones Estratégicas

El pipeline ejecutado entrega un dataset robusto para la ingesta de modelos predictivos de Machine Learning. De cara al modelado, las principales conclusiones estratégicas son:

1.  **Enfoque Biológico sobre Social:** El modelo deberá basar el peso de sus decisiones en la dimensión oncológica (estadio, ganglios, profundidad del tumor), sintomática (anemia, estado funcional, pérdida de peso) y terapéutica (necesidad de paliativos o quimioterapia). Se aconseja probar modelos con y sin las variables sociales para medir si causan ruido innecesario.
2.  **Prevención Crítica de "Data Leakage" (Fuga de Datos):** Dado que tenemos 5,000 registros pero solo 2,000 pacientes, un modelo que separe el conjunto de Entrenamiento (Train) y Prueba (Test) de forma aleatoria, inevitablemente filtrará evaluaciones de un mismo paciente a ambos conjuntos. Es **obligatorio** que el train-test split se realice bajo un criterio de `GroupShuffleSplit` (agrupando por `paciente_id`), garantizando que si un paciente entra a Entrenamiento, ninguna de sus visitas posteriores llegue al conjunto de Prueba.
