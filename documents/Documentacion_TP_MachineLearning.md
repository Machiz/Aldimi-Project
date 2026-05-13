# Documentación Técnica: Modelo de Machine Learning para Consumo de Insumos Oncológicos

Este documento detalla el análisis, procesamiento y modelado predictivo realizado en el notebook `TP_MachineLearning (1).ipynb`. El objetivo principal es desarrollar un modelo de regresión con Machine Learning para predecir el consumo diario de los medicamentos oncológicos más críticos.

## 1. Análisis Exploratorio de Datos (EDA)

En la fase inicial, se carga el conjunto de datos `insumos_cancer_final.csv` y se realiza una inspección profunda para comprender su estructura y comportamiento:
*   **Información General y Calidad de Datos:** Se verifica que el DataFrame consta de 5010 registros y 57 columnas, sin presencia de valores nulos (missing values). Esto indica un dataset limpio y listo para el análisis.
*   **Preparación de Tipos de Datos:** Se convierte la columna `fecha` al formato `datetime` de Pandas para permitir el análisis y manipulación temporal.
*   **Distribución de Variables Numéricas Clave:** Se generan histogramas para visualizar la distribución estadística de métricas fundamentales como el *stock actual*, el *consumo diario*, la *demanda a 7 días* y el *stock proyectado a 7 días*.
*   **Análisis de Correlación:** Se calcula y visualiza una matriz de correlación (Heatmap) entre variables seleccionadas. Se destacan fuertes correlaciones esperadas, como la relación entre la demanda futura proyectada y el consumo diario actual.
*   **Análisis Temporal (Series de Tiempo):** Se gráfica la evolución del *stock actual* y del *consumo diario* a lo largo del tiempo, permitiendo observar la volatilidad diaria y los ciclos de reposición y consumo de los insumos.

## 2. Ingeniería de Características (Feature Engineering)

Esta fase se enfoca en transformar y seleccionar la información más relevante para optimizar el rendimiento del algoritmo de predicción:
*   **Identificación de Insumos Críticos:** El análisis se centra específicamente en la categoría de "Medicamentos Oncológicos". Se filtra el dataset original para crear `df_oncologicos`. Posteriormente, se identifican y grafican los 10 ítems con el volumen de consumo más alto.
*   **Selección para Modelado (Top 3):** Para desarrollar un modelo robusto y enfocado, se aísla la data exclusivamente de los 3 medicamentos oncológicos más consumidos, creando el subconjunto `df_top3`.
*   **Codificación de Variables Categóricas:** Dado que los modelos matemáticos requieren entradas numéricas, se utiliza `LabelEncoder` de Scikit-Learn para transformar variables de texto (como 'dia_semana' y el 'item_nombre') a representaciones numéricas. 
*   **Limpieza de Columnas Redundantes:** Se eliminan características que no aportan valor predictivo directo o que introducen redundancia, asegurando que el modelo reciba datos limpios y directos.
*   **Definición de Variables (X e y):** Se establecen como variables predictoras (`X`) las métricas históricas de consumo (como los rezagos de consumo de 1 y 7 días previos), datos de ocupación hospitalaria, el nivel de stock actual y los movimientos de ingresos/salidas. La variable objetivo a predecir (`y`) se establece como el 'consumo_diario'.

## 3. Machine Learning: Predicción de Consumo

Se implementa un modelo predictivo para estimar la demanda futura a 7 y 14 días. En lugar de limitarse a un único algoritmo, se evaluaron cuatro modelos distintos para identificar el más preciso:
*   **División de Datos (Train/Test Split):** Para evaluar la capacidad real de predicción, se respetó el orden cronológico. Se reservó el 80% más antiguo para entrenamiento (Train) y el 20% más reciente para evaluación (Test).
*   **Ridge Regression (Modelo Lineal Penalizado):** Un modelo lineal que aplica regularización L2 para evitar el sobreajuste y filtrar el "ruido" en los datos hospitalarios. Sirvió como buen modelo base o *baseline*.
*   **Random Forest Regressor:** Un ensamble de múltiples árboles de decisión. Se evaluó reduciendo el número de variables analizadas por corte (`max_features='sqrt'`) para forzar la diversidad entre los árboles, logrando mayor robustez frente al modelo lineal.
*   **Gradient Boosting Regressor:** Ensamble iterativo que construye árboles secuencialmente para corregir los errores de los anteriores. Entrenado con un `learning_rate` bajo (0.05) para mejorar gradualmente la precisión sin apresurar la convergencia.
*   **XGBoost Regressor (Extreme Gradient Boosting):** Versión altamente optimizada del Gradient Boosting. Se utilizaron penalizaciones matemáticas avanzadas (L1 y L2) para ignorar variables ruidosas y controlar valores extremos.
*   **Resultados de la Evaluación:** 
    * Para la predicción a 7 días, **XGBoost** fue el claro ganador con un RMSE de ~10.49 y un R2 de ~0.988, superando al Gradient Boosting (RMSE ~13.12), Random Forest (RMSE ~15.65) y Ridge Regression (RMSE ~16.82). 
    * Para la predicción a 14 días, **XGBoost** nuevamente obtuvo el mejor desempeño (RMSE ~12.46, R2 ~0.995), confirmando su superioridad matemática y predictiva para esta tarea.

## 4. Conclusiones y Recomendaciones

La ejecución completa de este pipeline de Machine Learning arroja conclusiones vitales para la toma de decisiones en la gestión hospitalaria:

1.  **Factibilidad y Valor Predictivo:** Los modelos evaluados demuestran ser herramientas excepcionalmente capaces de capturar la varianza de la demanda de medicamentos oncológicos. Específicamente, **XGBoost** destaca por su alta precisión, logrando métricas de R2-Score superiores a 0.98.
2.  **Superioridad Algorítmica:** La evaluación múltiple comprobó que si bien los modelos lineales (Ridge) y los ensambles paralelos (Random Forest) ofrecen resultados decentes, los algoritmos basados en *Boosting* (Gradient Boosting y XGBoost) dominan la tarea, manejando mejor las complejidades y relaciones temporales del consumo médico.
3.  **El Peso de la Historia:** Las variables derivadas de datos históricos, como el consumo de los días anteriores y los promedios móviles, continúan siendo los predictores más contundentes de la demanda futura. El consumo está fuertemente dictado por estas inercias a corto plazo.
4.  **Hacia una Gestión Proactiva del Inventario:** Al lograr pronosticar con alta exactitud la demanda a 7 y 14 días, el área logística puede transicionar de un enfoque reactivo a uno predictivo. Las predicciones del XGBoost deben usarse para calibrar las alertas de stock crítico y reordenes automáticos.
5.  **Recomendaciones para Futuras Iteraciones:** Con el éxito de XGBoost validado, se aconseja expandir su implementación para incluir toda la cartera de insumos oncológicos. El siguiente paso logístico es integrar directamente estas predicciones en un Dashboard interactivo que alerte diariamente a los gestores sobre posibles quiebres de stock.
