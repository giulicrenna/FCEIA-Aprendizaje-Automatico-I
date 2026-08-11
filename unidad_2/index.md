---
layout: default
title: Unidad 2 — Regresión Lineal
nav_order: 4
---

# Unidad 2 — Regresión Lineal

> **Material asociado:** notebooks, datasets y clases prácticas de la unidad. Esta página desarrolla la teoría con el mismo nivel de profundidad que la [Unidad 0](unidad_0/index.md) y la [Unidad 1](unidad_1/index.md).

## Sobre esta unidad

Esta unidad cubre el modelo más importante del aprendizaje supervisado: la **regresión lineal**. Aunque parece simple, es la base conceptual de técnicas mucho más complejas (ridge, lasso, ElasticNet, SVM, redes neuronales) y la herramienta de referencia para problemas de predicción con variable objetivo continua.

A lo largo de la unidad vamos a:

- Plantear el modelo de regresión lineal y derivar la solución de **mínimos cuadrados** en forma cerrada.
- Aprender a evaluarlo con las métricas estándar (MSE, RMSE, R², MAE).
- Ver el **gradiente descendiente** (batch, estocástico y mini-batch) y por qué es indispensable cuando la solución cerrada no escala.
- Discutir el **sobreajuste** en regresión polinómica y cómo controlarlo con **regularización** (L2/Ridge, L1/Lasso, ElasticNet).
- Hacer un **análisis exploratorio serio** (outliers, faltantes, distribuciones) usando el dataset Boston Housing.
- Comparar **estrategias de escalado** de features y entender por qué gradiente descendiente las necesita.
- Cerrar con un **caso de estudio** completo: predicción del precio de vehículos.

### Prerequisitos

Es deseable haber visto previamente:

- Python, NumPy, pandas, matplotlib.
- Conceptos de álgebra lineal (multiplicación matricial, inversa, transpuesta).
- Unidad 0 (sesgo-varianza, train/test split) y Unidad 1 (pipeline de scikit-learn, validación cruzada).

## Material de la unidad

### Notebooks

- [`RegresionLinealSimple.ipynb`](notebooks/RegresionLinealSimple.ipynb){:download} — mínimos cuadrados desde cero (con y sin ordenada al origen) y comparación con `LinearRegression` de scikit-learn.
- [`Fit predict.ipynb`](notebooks/Fit predict.ipynb){:download} — flujo `fit` / `predict` y analogía con la API de estimadores de scikit-learn.
- [`GradienteDescendiente.ipynb`](notebooks/GradienteDescendiente.ipynb){:download} — implementación de las tres variantes de gradiente descendiente.
- [`Regularizacion.ipynb`](notebooks/Regularizacion.ipynb){:download} — Ridge (L2).
- [`Regularizacion2.ipynb`](notebooks/Regularizacion2.ipynb){:download} — Lasso (L1).
- [`Regularizacion3.ipynb`](notebooks/Regularizacion3.ipynb){:download} — ElasticNet (combinación L1 + L2) y `RidgeCV` / `LassoCV` / `ElasticNetCV`.
- [`Analisis exploratorio, valores atipicos y faltantes.ipynb`](notebooks/Analisis exploratorio, valores atipicos y faltantes.ipynb){:download} — EDA, outliers y datos faltantes sobre el dataset Boston Housing.
- [`scalers.ipynb`](notebooks/scalers.ipynb){:download} — comparación de seis estrategias de escalado de features.
- [`3_Regularizacion_vehiculos_data.ipynb`](notebooks/3_Regularizacion_vehiculos_data.ipynb){:download} — caso de estudio: predicción del precio de vehículos con regularización.

### Slides

- [`Clase 3 - Regresión Lineal.pptx`](slides/Clase 3 - Regresión Lineal.pptx) — modelo de regresión lineal y métricas.
- [`Clase 4 - Gradiente Descendiente.pptx`](slides/Clase 4 - Gradiente Descendiente.pptx) — gradiente descendiente batch, estocástico y mini-batch.
- [`Clase 5 - Regularización.pptx`](slides/Clase 5 - Regularización.pptx) — Ridge, Lasso y ElasticNet.
- [`Analisis descriptivo.pptx`](slides/Analisis descriptivo.pptx) — caso de estudio de vehículos: análisis exploratorio.
- [`PRACTICA - Analisis exploratorio, valores atípicos y faltantes.pptx`](slides/PRACTICA - Analisis exploratorio, valores atípicos y faltantes.pptx) — clase práctica de EDA.

### Datasets

- [`happyscore_income.csv`](data/happyscore_income.csv) — felicidad vs. PIB per cápita por país.
- [`income.csv`](data/income.csv) — ingresos por país.
- [`possum.csv`](data/possum.csv) — medidas morfométricas de zarigüeyas australianas.
- [`Car details v3.csv`](data/Car details v3.csv) — detalle de vehículos usados en India.
- [`CAR DETAILS FROM CAR DEKHO.csv`](data/CAR DETAILS FROM CAR DEKHO.csv) — segundo dataset de Car Dekho.
- [`car data.csv`](data/car data.csv) — tercer dataset de precios de automóviles.

---

## Índice

1. [El modelo de regresión lineal](#1-el-modelo-de-regresión-lineal)
2. [Métricas para regresión](#2-métricas-para-regresión)
3. [Gradiente descendiente](#3-gradiente-descendiente)
4. [Regresión polinómica y sobreajuste](#4-regresión-polinómica-y-sobreajuste)
5. [Regularización](#5-regularización)
6. [Análisis exploratorio: outliers y datos faltantes](#6-análisis-exploratorio-outliers-y-datos-faltantes)
7. [Escalado de features](#7-escalado-de-features)
8. [Caso de estudio: predicción del precio de vehículos](#8-caso-de-estudio-predicción-del-precio-de-vehículos)

---

## 1. El modelo de regresión lineal

### 1.1. Planteo

Tenemos una variable objetivo \(y \in \mathbb{R}\) (continua) y un vector de predictores \(\mathbf{x} = (1, x_1, \dots, x_p)^\top\). El **modelo de regresión lineal** postula que la media condicional de \(y\) dado \(\mathbf{x}\) es una combinación lineal de los predictores:

\[
\hat{y} = \mathbf{w}^\top \mathbf{x} = w_0 + \sum_{j=1}^{p} w_j x_j.
\]

Los \(p+1\) coeficientes \(w_0, w_1, \dots, w_p\) se denominan **pesos** o **parámetros** del modelo; \(w_0\) es la *ordenada al origen* (en inglés: *intercept* o *bias*).

Para un dataset de \(n\) observaciones, escribimos el modelo de forma matricial. Sea \(X \in \mathbb{R}^{n \times (p+1)}\) la **matriz de diseño** (con una columna de unos en la primera posición) y \(\mathbf{y} \in \mathbb{R}^n\) el vector de targets:

\[
\hat{\mathbf{y}} = X \mathbf{w}.
\]

### 1.2. Solución de mínimos cuadrados en forma cerrada

La función de pérdida estándar es el **error cuadrático medio** (MSE) sobre los datos:

\[
L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n}\| \mathbf{y} - X\mathbf{w} \|_2^2.
\]

Esta función es **convexa** y diferenciable. Su mínimo se encuentra donde el gradiente es cero:

\[
\nabla_\mathbf{w} L = -\frac{2}{n} X^\top (\mathbf{y} - X\mathbf{w}) = 0
\;\Longrightarrow\;
X^\top X \mathbf{w} = X^\top \mathbf{y}.
\]

Si \(X^\top X\) es invertible (que requiere que \(X\) tenga rango completo, en particular \(n \geq p+1\)), la **solución de mínimos cuadrados** es:

\[
\boxed{\hat{\mathbf{w}}_{\text{OLS}} = (X^\top X)^{-1} X^\top \mathbf{y}.}
\]

En `RegresionLinealSimple.ipynb` se implementa esta fórmula de las dos maneras posibles:

- **Sin ordenada al origen** (forzamos \(w_0 = 0\)): usar la matriz \(X\) sin columna de unos.
- **Con ordenada al origen libre**: agregar una columna de unos a \(X\) antes de calcular.

Luego se compara el resultado con `LinearRegression(fit_intercept=True|False)` de scikit-learn. La coincidencia numérica es exacta (salvo redondeo de punto flotante) y confirma que la fórmula cerrada es correcta.

### 1.3. Limitaciones de la forma cerrada

La inversa \((X^\top X)^{-1}\) tiene costo computacional \(O(p^3)\). Para \(p\) en el orden de los miles o millones (piénsese en regresión logística con features de texto o redes neuronales), la inversión se vuelve prohibitiva. Además, \(X^\top X\) puede ser **singular** o **mal condicionada** cuando hay multicolinealidad, volviendo la solución inestable. Estos dos problemas motivan el uso de **gradiente descendiente** (Sección 3) y **regularización** (Sección 5).

---

## 2. Métricas para regresión

Una vez entrenado el modelo, ¿cómo sabemos si "anda bien"? Las métricas para regresión comparan \(\hat{y}_i\) con \(y_i\) y resumen el error en un solo número. Es importante medir el error en datos **no vistos** (test set) para tener una estimación honesta de la capacidad de generalización.

### 2.1. MSE y RMSE

El **error cuadrático medio** (*Mean Squared Error*):

\[
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2.
\]

El MSE penaliza mucho los errores grandes (por el cuadrado). Su raíz, la **raíz del error cuadrático medio** (*Root Mean Squared Error*):

\[
\mathrm{RMSE} = \sqrt{\mathrm{MSE}},
\]

vuelve a las unidades originales de \(y\), lo que la hace más interpretable.

### 2.2. MAE

El **error absoluto medio** (*Mean Absolute Error*) es más robusto a outliers:

\[
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|.
\]

Matemáticamente, optimizar MAE equivale a la **mediana** de los errores, en lugar del **promedio** (MSE). Es preferible cuando la distribución de \(y\) tiene colas pesadas.

### 2.3. R² (coeficiente de determinación)

El **R²** mide la fracción de varianza de \(y\) explicada por el modelo:

\[
\mathrm{R}^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
= 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}.
\]

Interpretación:

- \(\mathrm{R}^2 = 1\): el modelo explica toda la varianza (predice perfectamente).
- \(\mathrm{R}^2 = 0\): el modelo no es mejor que predecir \(\bar{y}\) para todos.
- \(\mathrm{R}^2 < 0\): el modelo es **peor** que la media (indicio de overfitting severo en test).

### 2.4. Consejos prácticos

- **Siempre** reportar MSE/MAE **y** R² juntos: el primero da la escala del error y el segundo lo relativo.
- Usar **MAE** sobre **MSE** cuando hay outliers o cuando el costo del error es lineal y no cuadrático.
- Para datasets muy desbalanceados (e.g. precios de vivienda con pocos *outliers* de豪宅), considerar la **mediana del error porcentual absoluto** (MAPE) o errores en escala logarítmica.

---

## 3. Gradiente descendiente

### 3.1. La idea

Cuando la forma cerrada no es viable (matrices grandes, mal condicionadas, o modelos donde la derivada no se puede escribir explícitamente), usamos **gradiente descendiente** (GD). La idea es iterativa:

1. Inicializamos \(\mathbf{w}^{(0)}\) al azar.
2. En cada paso \(t\), calculamos el gradiente de la pérdida en el punto actual.
3. Actualizamos \(\mathbf{w}\) en la **dirección opuesta** al gradiente:

\[
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \, \nabla_\mathbf{w} L(\mathbf{w}^{(t)}).
\]

El factor \(\eta > 0\) es la **tasa de aprendizaje** (en inglés: *learning rate*), el hiperparámetro más importante del método. La idea es que el gradiente apunta en la dirección de máximo crecimiento, así que avanzar en la dirección opuesta minimiza la función.

### 3.2. GD para regresión lineal

Para nuestra pérdida cuadrática, el gradiente es:

\[
\nabla_\mathbf{w} L = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}).
\]

Por lo tanto:

\[
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot \frac{2}{n} X^\top (X\mathbf{w}^{(t)} - \mathbf{y}).
\]

Este es el algoritmo implementado en `GradienteDescendiente.ipynb`.

### 3.3. Variantes

#### Batch Gradient Descent (BGD)

Calcula el gradiente sobre **todos** los \(n\) ejemplos en cada actualización. Es estable y converge suavemente, pero cada paso es caro (\(O(np)\)).

#### Stochastic Gradient Descent (SGD)

Actualiza \(\mathbf{w}\) con **una sola muestra** aleatoria por paso:

\[
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \cdot 2 \, x_i (x_i^\top \mathbf{w}^{(t)} - y_i).
\]

Convergen más rápido por iteración pero con **alta varianza** (las actualizaciones "ruidosas" pueden ayudar a escapar de mínimos locales). Es el método canónico para entrenar redes neuronales.

#### Mini-Batch Gradient Descent

Compromiso: usa un lote de tamaño \(b\) (típicamente \(b \in \{32, 64, 128, 256\}\)). Aprovecha la vectorización de NumPy/PyTorch y reduce la varianza respecto a SGD. **Es el método usado en la práctica** para entrenar modelos grandes.

### 3.4. Comparación empírica

En `GradienteDescendiente.ipynb` se comparan las tres variantes sobre dos datasets:

- `happyscore_income.csv` (GDP vs. happyScore).
- `possum.csv` (longitud total de zarigüeyas).

Observaciones:

- **Sin escalado**, las tres variantes divergen o convergen muy lento. La condición numérica de \(X^\top X\) afecta directamente la velocidad.
- **Con `StandardScaler`**, todos los métodos convergen.
- **SGD** es el más rápido por iteración, pero **BGD** da la trayectoria más suave.
- La solución de GD coincide con la forma cerrada (salvo tolerancia) tras suficientes iteraciones.

### 3.5. Escalado y convergencia

El **escalado de features** (Sección 7) es requisito previo para que GD funcione bien. Sin escalado, los features con magnitudes grandes dominan el gradiente y oscurecen la señal de los features pequeños. Una regla práctica: **siempre escalar antes de entrenar con GD**.

---

## 4. Regresión polinómica y sobreajuste

### 4.1. Extender el modelo con términos no lineales

La regresión lineal no puede capturar relaciones curvas. La solución estándar es agregar **términos polinómicos** como features:

\[
\mathbf{x} = (1, x, x^2, \dots, x^d)^\top.
\]

El modelo sigue siendo **lineal en los parámetros**, así que la solución de mínimos cuadrados aplica directamente. La flexibilidad del modelo crece con el grado \(d\).

### 4.2. El problema: sobreajuste

En el notebook `overfitting.ipynb` (de la Unidad 1) vimos que un polinomio de grado alto sobreajusta. Lo mismo ocurre en regresión multivariable: cuantos más features y más grados, más fácil es "memorizar" los datos de train a costa de generalización.

Tres señales de sobreajuste:

- **MSE(train) muy bajo**, **MSE(test) alto**: el modelo ajusta ruido.
- **Coeficientes con magnitudes enormes**: ajustes absurdos para datos puntuales.
- **R²(train) ≈ 1** mientras **R²(test) < 0**: predicción en test peor que la media.

### 4.5. Tres familias de solución

1. **Más datos**: si tenemos suficientes datos, el modelo flexible se regulariza solo.
2. **Menos features**: aplica el principio de parsimonia. Selección manual o automática.
3. **Regularización**: penalizar la magnitud de los coeficientes. La vemos en la próxima sección.

---

## 5. Regularización

La **regularización** agrega un término de penalización a la función de pérdida para impedir que los coeficientes crezcan sin control. La nueva pérdida es:

\[
L_{\text{reg}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \, R(\mathbf{w}),
\]

donde \(R(\mathbf{w})\) es la penalización y \(\lambda \geq 0\) controla su fuerza.

### 5.1. Ridge (L2)

Penaliza el cuadrado de la magnitud de los coeficientes:

\[
\min_{\mathbf{w}} \; \| \mathbf{y} - X\mathbf{w} \|_2^2 + \alpha \|\mathbf{w}\|_2^2.
\]

- **Reduce** la magnitud de los coeficientes (no los hace cero).
- Útil cuando **todos** los features aportan algo y queremos mantenerlos.
- Solución cerrada: \(\hat{\mathbf{w}}_{\text{Ridge}} = (X^\top X + \alpha I)^{-1} X^\top \mathbf{y}\).
- En scikit-learn: `Ridge(alpha=α)`. Con validación cruzada: `RidgeCV(cv=k)`.

### 5.2. Lasso (L1)

Penaliza el valor absoluto de los coeficientes:

\[
\min_{\mathbf{w}} \; \| \mathbf{y} - X\mathbf{w} \|_2^2 + \alpha \|\mathbf{w}\|_1.
\]

- Promueve **sparsity**: muchos coeficientes se vuelven **exactamente cero**.
- Sirve como **selección automática de variables**.
- En scikit-learn: `Lasso(alpha=α)`, `LassoCV(cv=k)`.

### 5.3. ElasticNet (combinación L1 + L2)

Combina ambas penalizaciones con un parámetro de mezcla \(\rho \in [0, 1]\):

\[
\min_{\mathbf{w}} \; \| \mathbf{y} - X\mathbf{w} \|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \alpha (1-\rho) \|\mathbf{w}\|_2^2.
\]

- En scikit-learn: `ElasticNet(alpha=α, l1_ratio=ρ)`, `ElasticNetCV(cv=k)`.
- Es el método recomendado cuando tenemos **muchos features** y sospechamos que solo algunos importan (mejor que Lasso puro cuando hay grupos de features correlacionados).

### 5.4. Selección de \(\alpha\)

El valor óptimo de \(\alpha\) se elige con **validación cruzada**:

- Empezar con un rango logarítmico: `[0.001, 0.01, 0.1, 1, 10, 100]`.
- Usar `RidgeCV`, `LassoCV` o `ElasticNetCV` para automatizar.
- Comparar las métricas en test para validar que la mejora es real.

### 5.5. Comparación práctica

En `Regularizacion.ipynb`, `Regularizacion2.ipynb` y `Regularizacion3.ipynb` se trabaja con el dataset `Car details v3.csv` prediciendo el precio de vehículos en función de año, kilometraje, tipo de combustible, etc. Se observa:

- **Ridge** reduce la varianza sin eliminar features.
- **Lasso** descarta variables irrelevantes (útil para interpretar).
- **ElasticNet** combina ambas ventajas y suele ganar en performance.

---

## 6. Análisis exploratorio: outliers y datos faltantes

Antes de entrenar cualquier modelo, hay que **mirar los datos**. Esta sección resume las técnicas del notebook `Analisis exploratorio, valores atipicos y faltantes.ipynb`, aplicado al dataset **Boston Housing** (corregido por las objeciones éticas).

### 6.1. Visualizaciones fundamentales

- **Histogramas**: estiman la densidad de cada variable. Permiten ver asimetría, multimodalidad y outliers.
- **Boxplots**: muestran cuartiles, IQR (rango intercuartil) y outliers según la regla de \(1.5 \times \text{IQR}\).
- **Violin plots**: combinación de boxplot y KDE (estimación de densidad por kernel).
- **Scatter plots**: análisis bivariado. Buscar relaciones lineales, heterocedasticidad y clusters.

### 6.2. Detección de outliers

- **Z-score**: \(|z_i| > 3\) marca outliers aproximadamente (válido solo bajo normalidad).
- **IQR**: \(x_i < Q_1 - 1.5 \cdot \text{IQR}\) o \(x_i > Q_3 + 1.5 \cdot \text{IQR}\).
- **Isolation Forest**, **Local Outlier Factor**: métodos multivariados más sofisticados.

### 6.3. Tratamiento de outliers

Tres opciones principales:

1. **Eliminación** (`drop`): válido si los outliers son errores de medición.
2. **Transformación** (`PowerTransformer` con Yeo-Johnson): lleva la distribución a algo cercano a gaussiana.
3. **Truncamiento** (*clipping*): reemplaza outliers por percentiles o límites físicos.
4. **Imputación** con la **mediana** (robusta a outliers).
5. **Modelos robustos**: `HuberRegressor`, `RANSACRegressor`, o `RandomForestRegressor` (no se afectan por valores extremos).

### 6.4. Datos faltantes

Ocurren en la mayoría de datasets reales. Tres estrategias:

- **Eliminación de filas** (`dropna`): solo si faltan pocos datos al azar.
- **Imputación simple** (`SimpleImputer(strategy='mean'|'median'|'most_frequent')`): rápido y razonable.
- **KNN Imputer**: imputa con el valor medio de los \(k\) vecinos más cercanos. Más costoso pero captura estructura local.

El notebook compara las tres estrategias sobre Boston Housing y muestra que KNN imputation suele ganar en performance, a costa de mayor costo computacional.

---

## 7. Escalado de features

### 7.1. Por qué importa

Muchos algoritmos (gradiente descendiente, SVM, KNN, redes neuronales) son **sensibles a la escala** de los features. Si un feature está en el rango \([0, 1]\) y otro en \([0, 1000]\), el segundo dominará el cálculo de distancias y los gradientes. Los algoritmos basados en árboles (Random Forest, Gradient Boosting) son la excepción: no necesitan escalado.

### 7.2. Estrategias disponibles

El notebook `scalers.ipynb` compara seis transformadores de `sklearn.preprocessing`:

| Scaler | Característica | Robusto a outliers |
|---|---|---|
| `StandardScaler` | media 0, var 1 | No |
| `MinMaxScaler` | rango \([0, 1]\) | No |
| `RobustScaler` | usa mediana e IQR | Sí |
| `MaxAbsScaler` | escala por el máximo absoluto | No |
| `PowerTransformer` (Yeo-Johnson) | mapea a gaussiana | Moderado |
| `QuantileTransformer` | mapea a uniforme/normal | Sí |

### 7.3. Cuándo usar cada uno

- **`StandardScaler`**: el más común. Por defecto para la mayoría de los modelos.
- **`MinMaxScaler`**: cuando se necesita un rango acotado (e.g. imágenes en [0, 1]).
- **`RobustScaler`**: cuando hay outliers que no se pueden eliminar.
- **`PowerTransformer`**: cuando la variable es claramente asimétrica.
- **`QuantileTransformer`**: el más robusto, pero pierde interpretabilidad.

### 7.4. Caso de estudio: California Housing

Aplicado sobre el dataset **California Housing**, se observa que:

- Sin escalado, gradiente descendiente diverge.
- Con `StandardScaler`, GD converge en pocas iteraciones.
- `RobustScaler` da resultados similares a `StandardScaler` cuando los outliers son pocos.
- `QuantileTransformer` mapea la distribución (asimétrica) a algo más simétrico, mejorando el ajuste.

---

## 8. Caso de estudio: predicción del precio de vehículos

El notebook `3_Regularizacion_vehiculos_data.ipynb` aplica regresión lineal con regularización al dataset de **Car Dekho** (~4.000 vehículos hindúes). El objetivo es predecir el **precio de venta** de un vehículo a partir de sus características.

### 8.1. Variables disponibles

- **Año** del vehículo.
- **Kilometraje** recorrido.
- **Tipo de combustible** (Petrol, Diesel, CNG, LPG, Electric).
- **Vendedor** (Individual vs. Dealer).
- **Transmisión** (Manual vs. Automatic).
- **Owner** (primer dueño, segundo, etc.).
- **Mileage** (rendimiento en kmpl o km/kg).
- **Engine** (cilindrada en CC).
- **Power** (potencia en bhp).
- **Seats** (cantidad de asientos).

### 8.2. Pipeline aplicado

1. **Carga y limpieza**: parseo de strings ("23.4 kmpl", "74 bhp"), manejo de valores faltantes.
2. **Encoding categórico**: `OneHotEncoder` para variables nominales (combustible, transmisión).
3. **Escalado**: `StandardScaler` sobre las variables numéricas.
4. **Train/test split**: 80/20 con `random_state` fijo para reproducibilidad.
5. **Modelos entrenados**:
   - `LinearRegression` (baseline).
   - `Ridge(alpha=α)` con α ∈ {0.1, 1, 10, 100}.
   - `Lasso(alpha=α)` con α ∈ {0.001, 0.01, 0.1, 1}.
   - `ElasticNet(alpha=α, l1_ratio=ρ)` con grid search.
6. **Comparación**: MSE, RMSE, R² en train y test.

### 8.3. Resultados típicos

- **Ridge** con \(\alpha \approx 1\) suele ser el mejor compromiso sesgo-varianza.
- **Lasso** descarta variables poco informativas (e.g. el número de asientos si es casi constante).
- **ElasticNet** iguala o mejora a Ridge cuando hay multicolinealidad.
- **Coeficientes**: año y potenciapositivos (más nuevo y más potente → más caro); kilometraje negativo (más uso → más barato).

### 8.4. Interpretación de negocio

El modelo final permite a un vendedor:

- Estimar un **precio de referencia** para un vehículo usado.
- Identificar **features clave** que más afectan el precio.
- Detectar **oportunidades** (vehículos con precio de venta muy por debajo del predicho).

---

## Apéndice: cómo cargar los datasets

Todos los datasets de esta unidad están en `data/`. Para cargarlos:

```python
import pandas as pd

vehicles = pd.read_csv("data/CAR DETAILS FROM CAR DEKHO.csv")
possum   = pd.read_csv("data/possum.csv")
income   = pd.read_csv("data/income.csv")
```

Los notebooks usan paths relativos a la raíz del repositorio. Si los ejecutás en Google Colab, subí el notebook y los datos al mismo directorio.

---

## Bibliografía por sección

- **Secciones 1-2**: Bishop, *Pattern Recognition and Machine Learning*, Cap. 3. Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Cap. 3.
- **Sección 3**: Ruder, *An overview of gradient descent optimization algorithms* (2016).
- **Sección 5**: Hastie et al., Cap. 3.4. Tibshirani, *Regression shrinkage and selection via the Lasso* (1996).
- **Secciones 6-7**: Huber, *Robust Statistics*. scikit-learn User Guide, sección *Preprocessing data*.
- **Sección 8**: Documentación de scikit-learn, sección *Glossary* y *User Guide*.
