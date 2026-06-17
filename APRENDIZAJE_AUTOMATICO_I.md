# Aprendizaje Automático I — Apunte Completo del Repositorio (2026 C1)

> **Materia:** Aprendizaje Automático I (Machine Learning I)
> **Institución:** Facultad de Ciencias Exactas, Ingeniería y Agrimensura (FCEIA) — Universidad Nacional de Rosario (UNR)
> **Cuerpo Docente:** Joel Spak, Giuliano Crenna
> **Año / Cuatrimestre:** 2026 / C1
> **Alcance de este documento:** recorrido detallado por la totalidad del material del repositorio — clases teóricas, notebooks prácticos, datasets, Trabajos Prácticos y herramientas de MLOps — con bibliografía asociada.

---

## Tabla de contenidos

1. [Descripción general del repositorio](#1-descripción-general-del-repositorio)
2. [Unidad 0 — Análisis Exploratorio de Datos (EDA)](#2-unidad-0--análisis-exploratorio-de-datos-eda)
3. [Unidad 1 — Introducción al Aprendizaje Automático](#3-unidad-1--introducción-al-aprendizaje-automático)
4. [Unidad 2 — Regresión Lineal](#4-unidad-2--regresión-lineal)
5. [Unidad 3 — Modelos Lineales para Clasificación](#5-unidad-3--modelos-lineales-para-clasificación)
6. [Unidad 4 — Comparación de Modelos y Ajuste Fino](#6-unidad-4--comparación-de-modelos-y-ajuste-fino)
7. [Unidad 5 — Introducción a Deep Learning](#7-unidad-5--introducción-a-deep-learning)
8. [Unidad 6 — MLOps](#8-unidad-6--mlops)
9. [Trabajos Prácticos (TP1 y TP2)](#9-trabajos-prácticos-tp1-y-tp2)
10. [Bibliografía completa](#10-bibliografía-completa)

---

## 1. Descripción general del repositorio

El repositorio contiene todo el material de la cursada 2026 C1, organizado en:

- **Siete unidades temáticas** (Unidades 0 a 6) con clases en formato `.pptx`, notebooks (`.ipynb`) y datasets (`.csv`).
- **Trabajos Prácticos** con sus consignas en `.docx` y los datasets asociados.
- **Carpeta de Bibliografía** con libros de referencia en formato PDF.
- **Apunte teórico** consolidado en `aprendizaje_automatico_1.pdf` (~3 MB).
- **README** institucional con la planificación de la materia.

### Estructura de carpetas

```
AA1 2026 C1/
├── Unidad 0_ Analisis Exploratorio de Datos/
├── Unidad 1_ Introducción a Aprendizaje Automático/
│   ├── TaTeTi_ Ejemplo de Reinforcement learning/
│   └── notebooks/
├── Unidad 2_ Regresión Lineal/
│   ├── data/
│   └── notebooks/
├── Unidad 3_ Modelos lineales para clasificación/
│   └── notebooks/
├── Unidad 4_ Comparación de modelos y ajuste fino/
│   └── notebooks/
├── Unidad 5_ Introducción a Deep Learning/
│   └── notebooks/
├── Unidad 6_ MLOps/
│   ├── mlflow/
│   ├── mlops-docker-example/
│   └── wine-app/
├── Trabajos Prácticos/
│   ├── TP1 - Regresión/
│   └── TP2 - Clasificación/
├── Bibliografia/
├── aprendizaje_automatico_1.pdf
└── readme.docx
```

---

## 2. Unidad 0 — Análisis Exploratorio de Datos (EDA)

**Material:**

- `EDA_Aprendizaje_Automatico_I.ipynb` — notebook integral de EDA.
- `correlacion.ipynb` — caso de estudio de correlación, multicolinealidad y feature engineering.
- `ml1_cap0.pdf` — apunte teórico de la unidad.
- Diapositivas introductorias.

> El detalle completo de esta unidad (notebooks, PDF, objetivos y conceptos) está en la [página de la Unidad 0](unidad_0/index.md).

### 2.1 Objetivos

El EDA es la primera etapa de cualquier proyecto de aprendizaje automático. Su propósito es **comprender la estructura, las relaciones y las anomalías** de un dataset antes de modelar.

### 2.2 Contenidos trabajados

El notebook principal cubre nueve secciones:

1. **Configuración del entorno y carga de datos** — librerías (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, `sklearn.preprocessing`).
2. **Inspección estructural del dataset Diamonds** (~54.000 filas, 10 variables): carat, cut, color, clarity, depth, table, price, x, y, z.
3. **Análisis descriptivo**: media, mediana, moda, varianza, desviación estándar, coeficiente de variación, **sesgo (skewness)** y **curtosis (kurtosis)**.

   $$CV = \frac{s}{\bar{x}} \times 100\%$$

   $$\gamma_1 = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^3$$

   $$\kappa = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{x_i - \bar{x}}{s}\right)^4 - 3$$

4. **Gestión de datos faltantes y duplicados** — mecanismos **MCAR, MAR, MNAR** (Rubin, 1976); reemplazo de ceros físicamente imposibles por NaN en `x, y, z`.
5. **Pruebas de normalidad**:
   - **Shapiro-Wilk** (válido para `n ≤ 5000`).
   - **D'Agostino-Pearson** (combina sesgo y curtosis en un estadístico $K^2 \sim \chi^2(2)$).
   - **Q-Q plots** para evaluación visual.
6. **Balanceo de clases** — *Imbalance Ratio* $IR = n_{\text{mayor}}/n_{\text{menor}}$; umbrales sugeridos ($IR > 5$ ⇒ desbalanceo alto).
7. **Correlaciones y colinealidad**:
   - **Pearson** $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \sum(y_i-\bar{y})^2}}$.
   - **Spearman** (basado en rangos, robusto a outliers).
   - **VIF** $= \frac{1}{1-R_j^2}$ (umbral: $VIF > 10$ ⇒ colinealidad severa).
8. **Detección y tratamiento de outliers** — métodos **IQR** (Tukey, 1977, factor 1.5) y **Z-score** ($|z| > 3$).
9. **Estandarización**: Z-score y Min-Max. Aplicación con `StandardScaler`.

### 2.3 Caso especial: feature engineering vs. descarte por correlación

El notebook `correlacion.ipynb` demuestra con un dataset sintético que **dos variables muy correlacionadas** (Temperatura máxima vs. Temperatura a las 15 hs) **aportan información complementaria** sobre la variable target (lluvia) cuando se incluye una feature derivada (su diferencia). Tres modelos logísticos y tres Random Forest son comparados, mostrando que el modelo con la nueva característica mejora el accuracy de 0.82 → 0.95, ilustrando que **alta correlación ≠ redundancia** y que el *feature engineering* es crítico.

### 2.4 Conceptos teóricos centrales

- Distinguir entre outliers reales y errores de medición.
- Manejo de inconsistencias físicas (e.g. dimensiones cero).
- Diferencia entre estandarización y escalado Min-Max.
- Cuándo aplicar transformaciones logarítmicas ante sesgo positivo.

---

## 3. Unidad 1 — Introducción al Aprendizaje Automático

**Material:**

- `Clase 1 - Introducción AA1.pptx` y `Clase 2 - Introducción AA1.pptx` — clases teóricas.
- `notebooks/scikit-learn.ipynb` — pipeline básico con el dataset Iris.
- `notebooks/overfitting.ipynb` — sobreajuste con regresión lineal vs. árbol de decisión.
- `TaTeTi_ Ejemplo de Reinforcement learning/` — ejemplo de RL con Tic-Tac-Toe.

> El detalle completo de esta unidad (definiciones, taxonomía, pipeline, sesgo-varianza, RL formal, ejemplo TaTeTi, buenas prácticas) está en la [página de la Unidad 1](unidad_1/index.md).

---

## 4. Unidad 2 — Regresión Lineal

**Material:**

- `Analisis descriptivo.pptx`
- `Clase 3 - Regresión Lineal.pptx`
- `Clase 4 - Gradiente Descendiente.pptx`
- `Clase 5 - Regularización.pptx`
- `PRACTICA - Analisis exploratorio, valores atípicos y faltantes.pptx`
- Notebooks: `RegresionLinealSimple.ipynb`, `Fit predict.ipynb`, `GradienteDescendiente.ipynb`, `Regularizacion.ipynb`, `Regularizacion2.ipynb`, `Regularizacion3.ipynb`, `3_Regularizacion_vehiculos_data.ipynb`, `Analisis exploratorio, valores atipicos y faltantes.ipynb`, `scalers.ipynb`.
- Datos: `happyscore_income.csv`, `income.csv`, `possum.csv`, `Car details v3.csv`, `CAR DETAILS FROM CAR DEKHO.csv`, `car data.csv`.

### 4.1 Modelo de regresión lineal

Para una variable objetivo $y$ y un vector de predictores $\mathbf{x} = (1, x_1, \dots, x_p)$:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} = w_0 + \sum_{j=1}^{p} w_j x_j$$

**Solución de mínimos cuadrados (forma cerrada)**:

$$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top y$$

Implementada en `RegresionLinealSimple.ipynb`:
- Caso con ordenada = 0: $b = (X^\top X)^{-1} X^\top y$.
- Caso con ordenada libre: agregar columna de unos a $X$.
- Comparación con `LinearRegression(fit_intercept=True|False)` de scikit-learn.

**Métricas**:
- **MSE**: $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$.
- **RMSE**: $\sqrt{MSE}$.
- **R²** (coeficiente de determinación): $1 - \frac{SS_{res}}{SS_{tot}}$.

### 4.2 Gradiente Descendente

Implementación desde cero en `GradienteDescendiente.ipynb`. Tres variantes:

#### Batch Gradient Descent (BGD)
Calcula el gradiente sobre todos los $n$ ejemplos en cada actualización:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \frac{2}{n} X^\top (X\mathbf{w} - y)$$

#### Stochastic Gradient Descent (SGD)
Actualiza $\mathbf{w}$ con **una muestra aleatoria** a la vez. Convergen más rápido por iteración pero con alta varianza.

#### Mini-Batch Gradient Descent
Compromiso: usa un *batch* de tamaño $b$ (e.g. 11-128). Es el método usado en la práctica para entrenar redes neuronales.

Comparación empírica realizada sobre el dataset `happyscore_income.csv` (GDP vs. happyScore) y `possum.csv` (longitud total de zarigüeyas). Se observa que:
- Sin escalado, la divergencia es frecuente.
- Con `StandardScaler`, todos los métodos convergen, con SGD siendo el más rápido por iteración.

### 4.3 Regresión polinómica y sobreajuste

En `overfitting.ipynb` se mostró que el polinomio de grado alto sobreajusta. La regularización es la solución estándar.

### 4.4 Regularización

Se introducen tres técnicas en `Regularizacion.ipynb`, `Regularizacion2.ipynb` y `Regularizacion3.ipynb`:

#### Ridge (L2)
$$\min_{\mathbf{w}} \|y - X\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

- Reduce la magnitud de los coeficientes.
- No produce coeficientes exactamente cero.
- `Ridge(alpha=α)`, `RidgeCV(cv=k)`.

#### Lasso (L1)
$$\min_{\mathbf{w}} \|y - X\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

- Promueve **sparsity**: muchos coeficientes se vuelven exactamente cero.
- Sirve como método de selección de variables.
- `Lasso(alpha=α)`, `LassoCV(cv=k)`.

#### ElasticNet (combinación L1 + L2)
$$\min_{\mathbf{w}} \|y - X\mathbf{w}\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \alpha (1-\rho)\|\mathbf{w}\|_2^2$$

- `ElasticNet(alpha=α, l1_ratio=ρ)`, `ElasticNetCV(cv=k)`.

Comparación sobre el dataset `Car details v3.csv` prediciendo el precio de vehículos en función de año, kilometraje, tipo de combustible, etc.

### 4.5 Análisis exploratorio, outliers y datos faltantes

El notebook `Analisis exploratorio, valores atipicos y faltantes.ipynb` trabaja con el dataset **Boston Housing** (corregido por ética) y realiza:

- **Histogramas** — estimación de densidad.
- **Boxplots** — cuartiles, IQR, outliers.
- **Violin plots** — combinación de boxplot y KDE.
- **Scatter plots** — análisis bivariado.
- **Detección de outliers**:
  - Eliminación (Z-score > 3).
  - Transformación **Yeo-Johnson** (`PowerTransformer`) hacia normalidad.
  - **Truncamiento** (clipping) entre percentiles o límites físicos.
  - **Imputación** con la mediana (robusta a outliers).
  - Modelos robustos como `RandomForestClassifier`.

- **Datos faltantes**:
  - Eliminación de filas (`dropna`).
  - **Imputación simple** (`SimpleImputer(strategy='mean'|'median'|'most_frequent')`).
  - **KNN Imputer** — imputación basada en los $k$ vecinos más cercanos.

### 4.6 Escalado de features

En `scalers.ipynb` se comparan seis estrategias con `sklearn.preprocessing`:

| Scaler | Característica | Robusto a outliers |
|---|---|---|
| `StandardScaler` | media 0, var 1 | No |
| `MinMaxScaler` | rango [0, 1] | No |
| `RobustScaler` | usa mediana e IQR | Sí |
| `MaxAbsScaler` | escala por el máximo | No |
| `PowerTransformer` (Yeo-Johnson) | mapea a gaussiana | Moderado |
| `QuantileTransformer` | mapea a uniforme/normal | Sí |

Aplicado sobre el dataset **California Housing**.

### 4.7 Caso de estudio: predicción del precio de vehículos

`3_Regularizacion_vehiculos_data.ipynb` aplica regresión lineal con regularización al dataset de Car Dekho (~4.000 vehículos hindúes), entrenando modelos Ridge/Lasso/ElasticNet y comparando coeficientes y métricas.

---

## 5. Unidad 3 — Modelos Lineales para Clasificación

**Material:**

- `Clase 6 - Data Leakage.pptx`
- `Clase 7 - Modelos lineales para Clasificación.pptx`
- `Clase 8 - Metricas de Clasificación.pptx`
- `Clase 9 - Balanceo de Clases.pptx`
- `Clase 10 - Explicabilidad de Modelos.pptx`
- Notebooks: `RegresionLogistica.ipynb`, `Iris.ipynb`, `MNIST.ipynb`, `Curva ROC - AUC.ipynb`, `Balance.ipynb`, `SHAP.ipynb`, `SHAP completo.ipynb`, `SHAP LogReg.ipynb`, `SHAP_Regresión.ipynb`, `plot_sparse_logistic_regression_mnist.ipynb`.
- Datos: `diabetes.csv`.

### 5.1 Regresión Logística

Modelo lineal para clasificación binaria. Modela el logaritmo de la odds:

$$\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^\top \mathbf{x} + b$$

Equivalente a:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

Se entrena minimizando la **binary cross-entropy** (log-loss):

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

En `RegresionLogistica.ipynb` se trabaja el ejemplo de **clasificación de vidrios** (UCI dataset). Se muestra:
- Reasignación de clases (ventanas vs. vajilla).
- Visualización de la **frontera de decisión** y de las **probabilidades predichas**.
- Visualización de la **recta de decisión** con la ecuación del modelo (`model.coef_`, `model.intercept_`, `model.decision_function`).
- Comparación entre el modelo con todos los features y con sólo dos.

Métricas evaluadas:
- **Accuracy**: $\frac{VP + VN}{VP + VN + FP + FN}$.
- **Precision**: $\frac{VP}{VP + FP}$.
- **Recall (Sensitivity)**: $\frac{VP}{VP + FN}$.
- **F1-score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **Support**: número de observaciones reales de la clase.

### 5.2 Curva ROC y AUC

En `Curva ROC - AUC.ipynb`:
- **Curva ROC**: gráfico de la tasa de verdaderos positivos (TPR) contra la tasa de falsos positivos (FPR), variando el umbral de decisión.
- **AUC** (Area Under the Curve): probabilidad de que el modelo clasifique un positivo aleatorio más alto que un negativo aleatorio. AUC=1 indica clasificador perfecto, AUC=0.5 indica aleatorio.
- Cálculo con `roc_curve`, `roc_auc_score`, `auc`.
- Caso multiclase con `OneVsRestClassifier` y `label_binarize`.

### 5.3 Balanceo de clases

En `Balance.ipynb` se trabaja con un dataset sintético (`make_classification(weights=[0.9, 0.1])`) con un 90/10 imbalance. Cuatro estrategias:

#### a) Class weighting
`LogisticRegression(class_weight='balanced')` — penaliza más los errores de la clase minoritaria (en este caso 9× más).

#### b) Oversampling aleatorio (RandomOverSampler)
Replica ejemplos de la clase minoritaria hasta equilibrar.

#### c) Undersampling aleatorio (RandomUnderSampler)
Reduce ejemplos de la clase mayoritaria.

#### d) SMOTE (Synthetic Minority Over-sampling Technique)
Genera ejemplos sintéticos por interpolación entre vecinos cercanos (`imblearn.over_sampling.SMOTE`).

Otras técnicas mencionadas: **ADASYN**, **NearMiss**.

Comparación cuantitativa de precision/recall/F1 con y sin balanceo para `LogisticRegression` y `RandomForestClassifier`.

### 5.4 Explicabilidad de modelos — SHAP

SHAP (**SHapley Additive exPlanations**, Lundberg & Lee, 2017) es el método estándar para explicar predicciones individuales y globales.

En `SHAP.ipynb`, `SHAP completo.ipynb` y `SHAP_Regresión.ipynb`:
- `shap.Explainer(model, X_train, feature_names=…)` — crea el explicador.
- `shap_values = explainer(X_test)` — calcula contribuciones.
- **Interpretabilidad local**:
  - `shap.force_plot(...)` — visualización tipo “fuerzas” empujando la predicción.
  - `shap.plots.waterfall(explanation)` — cascada de contribuciones.
  - `shap.plots.bar(explanation)` — ranking de features para una observación.
- **Interpretabilidad global**:
  - `shap.summary_plot(...)` o `shap.plots.beeswarm(...)` — distribución de contribuciones por feature.
  - `shap.plots.bar(explanation)` — importancia media absoluta global.
- **Cohortes**: segmentación por subgrupos (e.g., `alcohol ≤ 13.5` vs. `> 13.5`) para comparar comportamiento.

Aplicaciones:
- Wine dataset (multiclase) con `LogisticRegression`.
- Breast Cancer dataset con `LogisticRegression` y `RandomForestClassifier`.
- MNIST con regresión logística sparse.

### 5.5 Data Leakage

Concepto introducido en `Clase 6`: fuga de información del test al train. Ejemplos típicos:
- Hacer EDA (incluyendo el target) sobre todo el dataset antes de splittear.
- Estandarizar antes del split.
- Usar `GridSearchCV` sin `Pipeline` y fitear el preprocesador sobre todo el dataset.
- Selección de features supervisada sobre el dataset completo.

### 5.6 Clasificación multiclase

- **One-vs-Rest (OvR)**: un clasificador binario por clase.
- **Multinomial (Softmax)**: generalización directa de la regresión logística a $K$ clases.

---

## 6. Unidad 4 — Comparación de Modelos y Ajuste Fino

**Material:**

- `Clase 11 - AutoML.pptx`
- `Clase 11 - Comparación de modelos.pptx`
- `Clase 11 - Modelos base.pptx`
- `Clase 11 - Métodos de remuestreo.pptx`
- Notebooks: `Validacion Cruzada.ipynb`, `Tuning de hiperparametros.ipynb`, `OptimizacionHP.ipynb`, `AutoML_PyCaret.ipynb`.

### 6.1 Modelos base (baselines)

Definición: modelos simples (a menudo constantes o ingenuos) que sirven como referencia para evaluar si los modelos complejos aportan valor. Ejemplos:
- **Baseline constante**: predecir siempre la media (regresión) o la clase mayoritaria (clasificación).
- **Baseline lineal**: regresión lineal / regresión logística.
- **Baseline no lineal**: árbol de decisión con profundidad limitada.

### 6.2 Métodos de remuestreo

En `Validacion Cruzada.ipynb`:

#### Hold-out split
División train/test única (e.g., 80/20).

#### K-Fold Cross-Validation
Particionar en $k$ folds; cada fold se usa una vez como validación y el resto como entrenamiento.

#### Stratified K-Fold
Mantiene la proporción de clases en cada fold (esencial en datasets desbalanceados).

#### Leave-One-Out (LOO)
Caso extremo: $k = n$.

Comparación empírica con `RandomForestClassifier` sobre el dataset Iris. Métricas: media y desvío estándar de accuracy.

### 6.3 Tuning de hiperparámetros

En `Tuning de hiperparametros.ipynb` se comparan tres estrategias sobre el dataset **California Housing**:

#### Grid Search
Evalúa **todas** las combinaciones del espacio de búsqueda. Exhaustivo pero caro:

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(RandomForestRegressor(random_state=42), grid_params,
                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
```

#### Random Search
Evalúa $n_{iter}$ combinaciones al azar. Más eficiente cuando el espacio es grande:

```python
from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(model, params, n_iter=4, cv=5, n_jobs=-1)
```

#### Optimización bayesiana (Optuna)
Construye un modelo probabilístico del espacio de búsqueda y escoge las combinaciones más prometedoras. `study.optimize(objective, n_trials=10)`.

Comparación de MSE en el ejemplo:
| Método | MSE test |
|---|---|
| Grid Search | 0.2546 |
| Random Search | 0.2603 |
| Optuna | 0.2547 |

### 6.4 AutoML con PyCaret

En `AutoML_PyCaret.ipynb`:

```python
from pycaret.regression import setup, compare_models, predict_model, save_model

reg = setup(data=train_df, target='MedHouseVal', session_id=42,
            imputation_type='iterative')
best_model = compare_models()  # Entrena 15+ modelos y los rankea
predictions = predict_model(best_model, data=test_df)
save_model(best_model, 'best_california_housing_model')
```

- Compara automáticamente modelos de regresión/clasificación.
- Realiza preprocesamiento, imputación iterativa y feature engineering por defecto.
- Permite restringir con `include=['lr', 'lasso', 'ridge', 'en', 'br']`.
- Genera un **Pipeline** serializable que incluye el preprocesamiento y el modelo entrenado.

Modelo ganador típico: `LGBMRegressor` con R² ≈ 0.83.

---

## 7. Unidad 5 — Introducción a Deep Learning

**Material:**

- `Clase 12 - Introducción a Aprendizaje Profundo.pptx`
- `Clase 13 - Introducción a Aprendizaje Profundo.pptx`
- Notebooks: `NN.ipynb`, `MNIST.ipynb`, `MNIST_app.ipynb`, `XOR.ipynb`, `HP_Tuning.ipynb`.
- Modelos entrenados: `mnist.h5`, `mnist_model.h5`.

### 7.1 De lo lineal a lo profundo

Limitación fundamental de los modelos lineales: el **problema XOR**, resuelto en `XOR.ipynb`. Una regresión lineal predice siempre 0.5 sobre el XOR ($\text{MSE} = 0.25$), mientras que una red neuronal con una capa oculta de 2 neuronas sigmoides logra $\text{MSE} \approx 10^{-13}$.

Se demuestra implementando el **backpropagation a mano con NumPy**, mostrando cómo los gradientes se propagan capa por capa, y luego con **TensorFlow/Keras** y **PyTorch**.

### 7.2 Componentes básicos de una red neuronal

- **Capa (Layer)**: transformación afín $z = Wx + b$ seguida de una no-linealidad.
- **Funciones de activación**: ReLU (`max(0, x)`), sigmoid ($1/(1+e^{-x})$), tanh, softmax (salida multiclase).
- **Optimizador**: Adam (default), SGD con momentum, RMSprop.
- **Función de pérdida**: MSE (regresión), binary cross-entropy (binaria), categorical cross-entropy (multiclase).
- **Métricas**: accuracy, MAE, MSE.

### 7.3 Implementaciones

#### TensorFlow / Keras

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid),
                    epochs=200, batch_size=64)
```

#### PyTorch

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 7.4 Caso de estudio: MNIST

En `MNIST.ipynb` y `MNIST_app.ipynb`:

1. **Carga y exploración**: 60.000 imágenes de entrenamiento y 10.000 de test, 28×28 píxeles, 10 clases.
2. **Normalización**: dividir por 255 para llevar píxeles al rango [0, 1].
3. **One-hot encoding** de las etiquetas.
4. **Modelo denso (Fully Connected)**: `Flatten(28×28) → Dense(128, ReLU) → Dense(10, Softmax)`. Accuracy en test ≈ 97.4%.
5. **Modelo convolucional**: `Conv2D(32, (3,3), ReLU) → Flatten → Dense(100, ReLU) → Dense(10, Softmax)`. Accuracy en test ≈ 97.9%.
6. **Persistencia**: `model.save('mnist_model.h5')`.
7. **Aplicación de inferencia**: `MNIST_app.ipynb` carga el modelo y predice dígitos nuevos.

### 7.5 Hiperparámetros de una red neuronal

- **Arquitectura**: número de capas, número de neuronas por capa, funciones de activación.
- **Regularización**: **Dropout** (apagado aleatorio de neuronas), L2 (`kernel_regularizer=tf.keras.regularizers.l2(0.1)`), early stopping.
- **Optimización**: learning rate, batch size, número de épocas.
- **Inicialización de pesos**: Glorot/Xavier, He.

### 7.6 Hiperparam tuning en Keras

En `HP_Tuning.ipynb` se muestra el ajuste de learning rate, dropout y número de neuronas sobre el dataset Diabetes.

---

## 8. Unidad 6 — MLOps

**Material:**

- `Clase 14 - MLOps.pptx`
- `Implementación en Docker.pptx`
- `mlflow/mlflow.ipynb` y `mlflow_ejecucion.ipynb`.
- `mlops-docker-example/` (Dockerfile, `inference.py`, `pipeline.pkl`, `requirements.txt`, `mlops.ipynb`).
- `wine-app/app.py` y `wine_quality_prediction.pkl`.

### 8.1 ¿Qué es MLOps?

MLOps (**Machine Learning Operations**) es la disciplina que busca **unificar el desarrollo de modelos de ML (Dev) con su despliegue y operación en producción (Ops)**. Abarca:

- Versionado de código, datos y modelos.
- Reproducibilidad de experimentos.
- Tracking de métricas y parámetros.
- Despliegue automatizado (CI/CD para modelos).
- Monitoreo en producción y detección de drift.

### 8.2 Tracking de experimentos con MLflow

En `mlflow/mlflow.ipynb` se construye un wrapper que combina entrenamiento, evaluación y logging:

```python
import mlflow
import mlflow.sklearn

class MLflowLogger:
    def __init__(self, run_name="WineModelRun"):
        mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metric(self, name, value):
        mlflow.log_metric(name, value)

    def log_model(self, model, model_name="model"):
        mlflow.sklearn.log_model(model, model_name)

    def end_run(self):
        mlflow.end_run()
```

Flujo end-to-end:
1. Entrenar el modelo (`RandomForestClassifier`).
2. Iniciar un run con `mlflow.start_run(run_name=…)`.
3. Loguear hiperparámetros (`log_params`), métricas (`log_metric`) y el modelo serializado (`log_model`).
4. Finalizar el run (`end_run`).
5. **Cargar el modelo** desde el registry: `mlflow.pyfunc.load_model("runs:/<run_id>/model")`.
6. Hacer predicciones en producción y loguear el resultado como un nuevo run (`ProductionRun`), guardando artefactos (`log_artifact`).

### 8.3 Empaquetado y despliegue con Docker

En `mlops-docker-example/`:

#### `docker/Dockerfile`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python", "./inference.py" ]
```

#### `docker/inference.py`
```python
import joblib
import pandas as pd

pipeline = joblib.load('pipeline.pkl')
df_input = pd.read_csv('/files/input.csv')
output = pipeline.predict(df_input)
pd.DataFrame(output, columns=['MEDV_predicted']).to_csv('/files/output.csv', index=False)
```

Flujo:
1. Entrenar el modelo offline y serializar el pipeline (`joblib.dump`).
2. Definir dependencias en `requirements.txt`.
3. Construir la imagen Docker: `docker build -t mlops-app .`.
4. Ejecutar el contenedor: `docker run -v $(pwd)/files:/files mlops-app`.

### 8.4 Aplicación interactiva con Streamlit

En `wine-app/app.py` se implementa una app web con **Streamlit** que carga un modelo (`wine_quality_prediction.pkl`) y expone un formulario donde el usuario ingresa 11 features del vino (acidez, azúcar residual, pH, alcohol, etc.) y el modelo predice la calidad.

Componentes clave:
- `joblib.load(pkl_path)` para cargar el pipeline (con `QuantileTransformer`, `StandardScaler`, `KMeans` y `CatBoost`).
- `st.form(key='my_form')` con `st.number_input(...)` por feature.
- `st.form_submit_button(label='Submit')` para ejecutar `pipe.predict(user_input)`.

Notebook asociado: `wine-quality-eda-prediction-and-deploy.ipynb` documenta el EDA completo y el entrenamiento del modelo.

---

## 9. Trabajos Prácticos (TP1 y TP2)

### 9.1 TP1 — Regresión: predicción de tarifa de Uber

- **Archivo**: `Trabajos Prácticos/TP1 - Regresión/TrabajoPrácticoRegresiónAA1.docx`
- **Dataset**: `uber_fares.csv` (~21 MB).

**Objetivo**: predecir la tarifa de un viaje de Uber a partir de variables como distancia, hora, ubicación de pickup y dropoff, etc.

**Etapas típicas del trabajo**:
1. Análisis exploratorio y limpieza.
2. Ingeniería de features (distancias geodésicas, hora del día, día de la semana).
3. Manejo de outliers en tarifas.
4. Modelado con regresión lineal, Ridge, Lasso, ElasticNet.
5. Comparación con modelos no lineales (Random Forest, Gradient Boosting).
6. Selección del mejor modelo con validación cruzada.

### 9.2 TP2 — Clasificación: predicción de lluvia en Australia

- **Archivo**: `Trabajos Prácticos/TP2 - Clasificación/2026C1 - TrabajoPrácticoClasificacionAA1.docx`
- **Dataset**: `weatherAUS_2026C1.csv` (~17 MB).

**Objetivo**: predecir si lloverá al día siguiente (`RainTomorrow`) en distintas ciudades australianas a partir de variables meteorológicas del día actual.

**Etapas típicas del trabajo**:
1. EDA: distribuciones, correlaciones, valores faltantes.
2. Encoding de variables categóricas (ubicación, dirección del viento, etc.).
3. Manejo del desbalance (~78% No, ~22% Yes).
4. Modelos: Regresión Logística, KNN, Árboles, Random Forest, XGBoost.
5. Métricas: F1, ROC-AUC, PR-AUC (no accuracy, por el desbalance).
6. Técnicas de balanceo: `class_weight`, SMOTE.
7. Explicabilidad con SHAP.

---

## 10. Bibliografía completa

### 10.1 Libros en la carpeta `Bibliografia/`

| # | Título | Autor(es) | Editorial | Año |
|---|---|---|---|---|
| 1 | **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** (3rd ed.) | Aurélien Géron | O'Reilly Media | 2022 |
| 2 | **An Introduction to Statistical Learning (ISL) — Python Edition** | Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor | Springer | 2021 (`ISLPython.pdf`) |
| 3 | **The Hundred-Page Machine Learning Book** | Andriy Burkov | — | 2019 |
| 4 | **Machine Learning: An Algorithmic Perspective** (2nd ed.) | Stephen Marsland | Chapman & Hall / CRC | 2015 |
| 5 | **Outlier Analysis** | Charu C. Aggarwal | Springer | 2013 |
| 6 | **Flexible Imputation of Missing Data** | Stef van Buuren | Chapman & Hall / CRC | 2012 |
| 7 | **Scikit-Learn Cheat Sheet (Python)** | scikit-learn developers | — | — |

### 10.2 Temas por libro

- **Géron (2022)** — referencia principal. Cubre regresión, clasificación, árboles, ensembles, SVM, redes neuronales, CNN, RNN, AutoML, MLOps. Implementaciones con Scikit-Learn, Keras y TensorFlow.
- **ISL (James et al., 2021)** — enfoque más estadístico/matemático. Regresión lineal/logística, regularización, splines, GAM, árboles, boosting, SVM, clustering, PCA.
- **Burkov (2019)** — libro conciso de ~100 páginas, ideal para repaso y vista panorámica de los algoritmos.
- **Marsland (2015)** — algoritmos explicados con implementación en Python y NumPy.
- **Aggarwal (2013)** — análisis de outliers: métodos estadísticos, basados en proximidad, clustering y métodos específicos para alta dimensionalidad.
- **van Buuren (2012)** — métodos modernos de imputación: MICE, FCS, Multilevel imputation.

### 10.3 Recursos online y papers de referencia

- **Scikit-Learn User Guide** — <https://scikit-learn.org/stable/user_guide.html>
- **TensorFlow / Keras Documentation** — <https://www.tensorflow.org/api_docs>
- **PyTorch Documentation** — <https://pytorch.org/docs/stable/>
- **MLflow Documentation** — <https://mlflow.org/docs/latest/index.html>
- **SHAP (SHapley Additive exPlanations)** — Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. <https://github.com/shap/shap>
- **SMOTE** — Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research.
- **Pearson's Correlation Coefficient** — Pearson, K. (1895).
- **Spearman's Rank Correlation** — Spearman, C. (1904).
- **D'Agostino-Pearson Test** — D'Agostino, R. B., & Pearson, E. S. (1973).
- **Rubin (1976)** — *Inference and Missing Data*. Biometrika. Define los mecanismos MCAR, MAR, MNAR.
- **Tukey (1977)** — *Exploratory Data Analysis*. Define el método IQR para outliers (factor 1.5).
- **Mitchell (1997)** — *Machine Learning*. McGraw-Hill. Definición clásica de aprendizaje.
- **Yeo-Johnson Transformation** — Yeo, I. K., & Johnson, R. A. (2000).
- **PowerTransformer / QuantileTransformer** — scikit-learn documentation.

### 10.4 Tutoriales / artículos mencionados en los notebooks

- Towards Data Science — *Class Imbalance Strategies: A Visual Guide with Code* — <https://towardsdatascience.com/class-imbalance-strategies-a-visual-guide-with-code-8bc8fae71e1a>
- Kaggle Notebook — *🍷 Wine Quality — EDA, Prediction and Deploy* (referencia para wine-app) — <https://www.kaggle.com/code/lusfernandotorres/wine-quality-eda-prediction-and-deploy/notebook>

---

## Apéndice: mapa de archivos → temas

```
Unidad 0/
  EDA_Aprendizaje_Automatico_I.ipynb     → EDA completo, estadística descriptiva, outliers, correlaciones
  correlacion.ipynb                        → Feature engineering vs. multicolinealidad
  ml1_cap0.pdf                             → Apunte teórico Unidad 0

Unidad 1/
  Clase 1, 2 (pptx)                       → Conceptos introductorios, tipos de aprendizaje
  notebooks/scikit-learn.ipynb            → Pipeline estándar
  notebooks/overfitting.ipynb             → Sobreajuste, regresión vs árboles, RF
  TaTeTi_…/tic-tac-toe.ipynb              → Q-Learning, RL
  TaTeTi_…/ticTacToe.py                   → Juego TaTeTi con interfaz tkinter

Unidad 2/
  Clase 3 (pptx)                          → Regresión Lineal Simple/Múltiple, OLS
  Clase 4 (pptx)                          → Gradiente Descendente (BGD/SGD/Mini-Batch)
  Clase 5 (pptx)                          → Regularización Ridge/Lasso/ElasticNet
  PRACTICA — Análisis exploratorio…       → Outliers y faltantes
  notebooks/RegresionLinealSimple.ipynb   → OLS desde cero
  notebooks/Fit predict.ipynb             → scikit-learn LinearRegression
  notebooks/GradienteDescendiente.ipynb   → GD, SGD, Mini-batch desde cero
  notebooks/Regularizacion{,2,3}.ipynb    → Ridge, Lasso, ElasticNet
  notebooks/scalers.ipynb                 → Comparación de 6 escaladores
  notebooks/Analisis exploratorio…        → Boston Housing, outliers, missing
  notebooks/3_Regularizacion_vehiculos…   → Caso aplicado de regresión

Unidad 3/
  Clase 6 (pptx)                          → Data Leakage
  Clase 7 (pptx)                          → Regresión Logística
  Clase 8 (pptx)                          → Métricas de clasificación
  Clase 9 (pptx)                          → Balanceo de clases
  Clase 10 (pptx)                         → Explicabilidad (SHAP)
  notebooks/RegresionLogistica.ipynb      → Caso vidrios, frontera de decisión
  notebooks/Iris.ipynb                    → Clasificación multiclase
  notebooks/MNIST.ipynb                   → Regresión logística en imágenes
  notebooks/Curva ROC - AUC.ipynb         → ROC, AUC, multiclase
  notebooks/Balance.ipynb                 → Oversampling, SMOTE, undersampling
  notebooks/SHAP*.ipynb                   → Explicabilidad global y local

Unidad 4/
  Clase 11 (4 pptx)                       → Comparación, modelos base, remuestreo, AutoML
  notebooks/Validacion Cruzada.ipynb      → K-Fold, StratifiedKFold, LOO
  notebooks/Tuning de hiperparametros.ipynb → Grid, Random, Optuna
  notebooks/AutoML_PyCaret.ipynb          → Comparación automática con PyCaret

Unidad 5/
  Clase 12, 13 (pptx)                     → Deep Learning intro
  notebooks/NN.ipynb                      → Clases NN en TF y regresión diabetes
  notebooks/MNIST.ipynb                   → Dense y Conv2D sobre MNIST
  notebooks/MNIST_app.ipynb               → Inferencia con modelo guardado
  notebooks/XOR.ipynb                     → XOR con NumPy, TF y PyTorch
  notebooks/HP_Tuning.ipynb               → Tuning de hiperparámetros en Keras

Unidad 6/
  Clase 14 (pptx)                         → MLOps
  Implementación en Docker.pptx           → Docker para modelos
  mlflow/mlflow.ipynb                     → Tracking con MLflow
  mlops-docker-example/                   → Dockerfile + inference.py + pipeline.pkl
  wine-app/app.py                         → App Streamlit

Trabajos Prácticos/
  TP1 - Regresión/                        → Predicción tarifa Uber
  TP2 - Clasificación/                    → Predicción lluvia Australia

Bibliografia/
  Aurélien Géron - Hands-On ML (2022).pdf
  ISLPython.pdf
  The Hundred-Page Machine Learning Book.pdf
  [Stephen Marsland] - Machine Learning_ An Algorithmic Perspective.pdf
  Charu C. Aggarwal - Outlier Analysis.pdf
  [Chapman & Hall] Stef van Buuren - Flexible Imputation of Missing Data.pdf
  Scikit_Learn_Cheat_Sheet_Python.pdf
```

---

> **Sugerencia de cursada**: comenzar por Unidad 0 (sin prisa, es transversal al resto), luego seguir el orden secuencial. Las Unidades 5 y 6 son las más interesantes para quienes quieran llevar un modelo a producción real.