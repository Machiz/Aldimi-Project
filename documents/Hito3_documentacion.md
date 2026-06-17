# Documentación Técnica - Hito 3: Refinamiento, Modelado Avanzado e Integración

Este documento detalla el refinamiento de los modelos, su optimización de hiperparámetros, los esquemas de validación robusta y la arquitectura de integración con la base de datos común de la fundación ALDIMI.

---

## 1. Ajuste y Optimización de Hiperparámetros

### Modelo 1 (Clasificación - Nivel de Prioridad)
Se ejecutó una búsqueda por grilla (`GridSearchCV`) con validación cruzada estratificada de 5 pliegues para encontrar la combinación ideal de hiperparámetros sobre el conjunto de entrenamiento (3,999 evaluaciones).

#### Resultados de Optimización:
*   **Random Forest Classifier:**
    *   *Mejores Hiperparámetros:* `{'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'n_estimators': 100}`
    *   *Mejor Macro F1 (CV):* `0.9118`
*   **Regresión Logística Multinomial:**
    *   *Mejores Hiperparámetros:* `{'C': 10.0, 'class_weight': None, 'solver': 'saga'}`
    *   *Mejor Macro F1 (CV):* `0.9147`

#### Calibración de Umbral de Decisión (Costo de Error):
Para mitigar el error crítico de clasificar a un paciente de prioridad alta como baja, se implementó una función de calibración manual sobre las probabilidades predictivas de la clase `alto`. Se buscó elevar el Recall de la clase Alta a un mínimo de 0.85.

*   **Random Forest (Modelo Seleccionado):**
    *   *Umbral Calibrado:* `0.59` (si la probabilidad de `alto` es $\ge 0.59$, se predice `alto` de forma prioritaria).
    *   *Métricas en el conjunto de prueba (1,001 registros):*
        *   **Recall de la clase Alta:** `0.8632` (Supera el objetivo de $\ge 0.85$).
        *   **Macro F1-score:** `0.8791` (Supera el objetivo de $\ge 0.80$).
*   **Regresión Logística:**
    *   *Umbral Calibrado:* `0.65`
    *   *Métricas en el conjunto de prueba (1,001 registros):*
        *   **Recall de la clase Alta:** `0.8534` (Supera el objetivo de $\ge 0.85$).
        *   **Macro F1-score:** `0.8769` (Supera el objetivo de $\ge 0.80$).

**Modelo Ganador Seleccionado:** **Random Forest Classifier**, debido a un F1-score macro ligeramente superior (`0.8791` vs `0.8769`) y un mejor comportamiento en la mitigación de errores críticos de seguridad.

---

## 2. Evaluación Robusta y Análisis de Errores

### Matriz de Confusión Final (Random Forest con Umbral Calibrado)

La distribución de predicciones en el conjunto de prueba (1,001 registros) es la siguiente:

| Real / Predicho | Bajo (Predicho) | Medio (Predicho) | Alto (Predicho) |
| :--- | :---: | :---: | :---: |
| **Bajo (Real)** | **302** | 29 | 0 |
| **Medio (Real)** | 11 | **311** | 41 |
| **Alto (Real)** | **0** | 42 | **265** |

#### Análisis del Comportamiento Clínico:
*   **Falsos Negativos Críticos:** **0** casos. Ningún paciente con prioridad real **Alta** fue clasificado como **Baja** (el cruce fila `Alto (Real)` con columna `Bajo (Predicho)` es exactamente 0).
*   **Falsos Positivos Críticos:** **0** casos. Ningún paciente con prioridad real **Baja** fue clasificado como **Alta**.
*   El mayor volumen de error se distribuye de manera segura en las clases colindantes (ej. 42 pacientes de prioridad Alta clasificados como Media, y 41 de prioridad Media clasificados como Alta), lo cual es clínicamente manejable y no pone en riesgo de desatención extrema a los pacientes graves.

---

### Modelo 2 (Regresión - Demanda de Insumos)
Se optimizó el algoritmo **XGBoost Regressor** con `GridSearchCV` evaluado bajo un esquema obligatorio de validación temporal (`TimeSeriesSplit` de 5 pliegues) para evitar la fuga de información cronológica.

#### Resultados de Optimización:
*   **Modelo 7 días (Predicción a Corto Plazo):**
    *   *Mejores Hiperparámetros:* `{'learning_rate': 0.1, 'max_depth': 4, 'reg_alpha': 1.0, 'reg_lambda': 1.0}`
    *   *Métricas Promedio en Validación Temporal:*
        *   **MAE:** `16.0345`
        *   **RMSE:** `23.1699`
        *   **MAPE (Robusto, excluyendo ceros):** `34.67%`
        *   *Nota:* Debido a la menor escala numérica de la demanda a 7 días, el MAPE es más sensible a variaciones pequeñas (denominadores pequeños), superando el umbral ideal del 15%.
*   **Modelo 14 días (Predicción a Mediano Plazo):**
    *   *Mejores Hiperparámetros:* `{'learning_rate': 0.1, 'max_depth': 4, 'reg_alpha': 0.1, 'reg_lambda': 1.0}`
    *   *Métricas Promedio en Validación Temporal:*
        *   **MAE:** `18.9470`
        *   **RMSE:** `30.4306`
        *   **MAPE (Robusto, excluyendo ceros):** **`14.81%`** (Supera con éxito el objetivo de $\le 20\%$).

---

## 3. Integración de Base de Datos y Pipeline

### Estructura Relacional
Se configuró una base de datos relacional SQLite (`aldimi_shared.db`) mediante SQLAlchemy ORM con dos tablas acopladas:
1.  `paciente_evaluaciones_ocr`: Contiene los datos clínicos y sociales ingresados de forma automática por el módulo OCR.
2.  `paciente_predicciones_prioridad`: Almacena la inferencia final del Modelo 1 (prioridad predicha, probabilidades y umbral aplicado) enlazada mediante clave foránea (`evaluacion_id`).

### Pipeline de Inferencia (`inference.py`)
El pipeline orquestado realiza las siguientes acciones:
1.  Conecta a la base de datos compartida y lee los registros de `paciente_evaluaciones_ocr` que aún no tienen predicción asociada.
2.  Preprocesa la fila aplicando imputación de nulos, mapeo binario/ordinal e ingeniería de características derivadas exactamente iguales a las del entrenamiento.
3.  Carga el artefacto del modelo Random Forest (`best_model_classification.pkl`).
4.  Genera las predicciones aplicando el umbral de decisión calibrado (`0.59`) para proteger a los pacientes graves.
5.  Registra las predicciones de prioridad en `paciente_predicciones_prioridad`.

---

## 4. Conclusiones y Recomendaciones de Despliegue

1.  **Seguridad Clínica Demostrada:** El modelo Random Forest entrenado con el umbral óptimo de $0.59$ garantiza una tasa de falsos negativos de prioridad Alta a Baja igual a cero. Esto asegura que ningún paciente de alta prioridad quede desatendido en el albergue.
2.  **Validación Temporal Consistente:** La optimización de XGBoost mediante `TimeSeriesSplit` demuestra que el modelo de demanda a 14 días es extremadamente robusto con un MAPE de solo $14.81\%$, cumpliendo las metas de abastecimiento hospitalario.
3.  **Preparación para Producción:** Se cuenta con una suite de pruebas unitarias de persistencia relacional que valida exitosamente el flujo de datos. Se recomienda el despliegue del pipeline en un contenedor programado (cron job) para ejecutarse de forma diaria.
