# Análisis Exploratorio de Datos (EDA) para Machine Learning - Demanda y Stock de Insumos

Este documento consolida los hallazgos, metodologías y procesos llevados a cabo en el notebook `EDA_demanda_stock.ipynb`. Su propósito es brindar una visión general sobre la exploración y preparación de los datos sintéticos correspondientes a los inventarios y demanda de la fundación ALDIMI, asegurando que estén óptimos para futuros modelos predictivos (ej. Forecasting de demanda o alertas de quiebre de stock).

---

## 1. Descripción y Diccionario de Datos

La información base proviene del archivo `demanda_stock_insumos_cancer_gastrico_sintetico.csv`. Este dataset contiene información continua referida a la entrada, salida y requerimientos de insumos médicos/nutricionales, fuertemente influenciados por la cantidad de pacientes que se encuentran en el albergue.

Aunque no se lista la cardinalidad exacta en el notebook de forma estricta como en el anterior, las columnas analizadas pueden agruparse en 4 grandes dimensiones clave:

| Grupo de Variables | Columnas Incluidas (Identificadas) | Descripción General |
| :--- | :--- | :--- |
| **Stock del Inventario** | `stock_actual`, `stock_minimo`, `stock_cierre`, `stock_proyectado_7d`, `stock_proyectado_14d` | Indicadores directos del nivel de inventario disponible y proyectado a corto y mediano plazo. |
| **Flujo y Demanda** | `consumo_diario`, `consumo_promedio_7d`, `ingresos_stock`, `salidas_stock` | Volumen de material que entra y sale de la fundación, además del promedio histórico reciente. |
| **Dinámica del Albergue (Drivers)** | `porcentaje_ocupacion`, `pacientes_total`, `pacientes_prioridad_alta` | Factores poblacionales que impulsan directamente la demanda de los insumos (a más pacientes/prioridad, más consumo). |

---

## 2. Pipeline de Preprocesamiento de Datos

La limpieza de un dataset de demanda e inventario requiere especial atención a los valores extremos y posibles errores en el conteo. El pipeline de limpieza ejecutado fue el siguiente:

1.  **Auditoría Estructural Básica:** Se revisaron las dimensiones del dataset (`df.shape`), tipos de datos (`df.info()`) y se cuantificaron los valores nulos (`df.isnull().sum()`) y filas duplicadas (`df.duplicated()`).
2.  **Identificación de Valores Atípicos (Outliers):** Se utilizó el método del **Rango Intercuartílico (IQR)** para cada columna numérica. Se identificaron los Cuartiles 1 (25%) y 3 (75%), definiendo como valores atípicos a aquellos que escapaban de los límites inferior y superior. Se generó un conteo exacto de atípicos por cada columna.
3.  **Capeo de Valores Atípicos (Capping):** En lugar de eliminar las filas con outliers (lo cual reduciría significativamente el dataset y perdería series temporales), se aplicó una técnica de *capping*. Todos los valores fuera de rango fueron forzados (clipped) a adquirir el valor de límite inferior o superior, estabilizando la distribución de las variables.
4.  **Corrección de Valores Negativos (Imposibles):** En el contexto de inventarios y consumo, **no puede existir un stock negativo ni una demanda negativa**. Se iteró sobre todas las columnas de stock y consumo/demanda, reemplazando automáticamente cualquier valor negativo por 0 (cero) mediante la función `.clip(lower=0)`.

---

## 3. Resultados del Análisis Exploratorio (EDA)

Tras la limpieza, la fase exploratoria (EDA) evaluó el comportamiento histórico mediante visualizaciones gráficas de alto valor para las series de tiempo:

*   **Matriz de Correlación (Heatmap):** Se generó un mapa de calor para encontrar las asociaciones lineales más fuertes, por ejemplo, cómo la cantidad de pacientes (particularmente de prioridad alta) impacta en el consumo diario y salidas de stock.
*   **Análisis Longitudinal de Stock y Consumo:** Se trazaron gráficos de dispersión y líneas a lo largo del tiempo (Index) para las variables de stock (`stock_actual`, `stock_cierre`) y demanda (`consumo_diario`), lo que permite visualizar la estacionalidad, los picos de demanda y los ciclos de reabastecimiento.
*   **Distribución Poblacional:** Histogramas dedicados al volumen de pacientes y la ocupación del albergue mostraron si las instalaciones tienden a estar a máxima capacidad y cómo se reparte esto en la historia.
*   **Comparativa Antes/Después:** El hallazgo más relevante visualmente fue la comparativa de los histogramas del dataset original frente al dataset preprocesado. Se logró evidenciar cómo la técnica de *capping por IQR* eliminó las colas extremadamente largas provocadas por los outliers, "domando" la varianza sin destruir los datos centrales.

---

## 4. Ingeniería de Atributos (Feature Engineering)

A diferencia del notebook de pacientes (donde se seleccionaron características existentes), en este dataset la clave estuvo en **crear nuevas variables derivadas (Feature Engineering)**. Estas nuevas variables "traducen" relaciones lógicas del inventario en señales potentes para que un modelo de Machine Learning aprenda más rápido:

| Nueva Variable Derivada | Fórmula Lógica | Valor Predictivo (Por qué se creó) |
| :--- | :--- | :--- |
| **`ratio_stock_minimo`** | `stock_actual` / `stock_minimo` | Ayuda a los algoritmos a saber qué tan "en rojo" está el inventario (menor a 1 = peligro). |
| **`dias_cobertura`** | `stock_actual` / `consumo_diario` | Traduce el stock bruto en tiempo. Previene quiebres al informar "para cuántos días alcanza el almacén". |
| **`neto_movimiento`** | `ingresos` - `salidas` | Balance neto. Un valor constantemente negativo advierte que el albergue se está desabasteciendo progresivamente. |
| **`variabilidad_consumo`** | diferencia absoluta entre el `consumo_diario` y el `consumo_promedio_7d` | Mide la volatilidad (picos de uso inesperados). A mayor volatilidad, más difícil es predecir la demanda. |
| **`es_ocupacion_alta`** | 1 (sí) o 0 (no) si la ocupación actual supera la mediana histórica | Flag categórico que contextualiza si el albergue está en un estado de estrés poblacional. |
| **`ratio_pacientes_alta_prioridad`** | `pacientes_prioridad_alta` / `pacientes_total` | Los pacientes críticos consumen más recursos. Si este ratio sube, el modelo asume un mayor desgaste de inventario. |

---

## 5. Conclusiones Estratégicas

El pipeline ejecutado sobre este dataset logró su objetivo principal: sanear errores operativos y generar inteligencia sobre los datos. Las conclusiones más relevantes son:

1.  **Robustez frente a la Varianza:** El tratamiento de los outliers mediante *capping* garantiza que los futuros modelos de forecasting (predicción de series de tiempo o algoritmos de regresión) no sufrirán penalizaciones drásticas ni ruido por ingresos excepcionalmente grandes (como donaciones atípicas) o salidas masivas.
2.  **El Poder del Feature Engineering:** En la logística de inventarios, los valores brutos sirven menos que los ratios. La creación de la variable **`dias_cobertura`** es el mayor logro de este notebook, pues transforma el inventario estático en un indicador predictivo de quiebre de stock.
3.  **Dataset de Alto Rendimiento:** Tras la corrección de ceros negativos y el enriquecimiento de columnas, se generó y exportó el archivo definitivo **`insumos_cancer_final.csv`**, el cual se encuentra limpio y estructurado. Este archivo ya está listo para ser ingerido por modelos de Machine Learning que apunten a predecir cuántos insumos se necesitarán la próxima semana o para crear sistemas de alarmas automatizadas de reabastecimiento.
