---
layout: default
title: Unidad 1 — Introducción al Aprendizaje Automático
nav_order: 3
---

# Unidad 1 — Introducción al Aprendizaje Automático

> **Material asociado:** notebooks y clases prácticas de la unidad. Esta página desarrolla la teoría con el mismo nivel de profundidad que la [Unidad 0](unidad_0/index.md).

## Material de la unidad

- [`Clase 1 - Introducción AA1.pptx`](slides/Clase 1 - Introducción AA1.pptx) — primera clase teórica: qué es el ML, por qué importa, taxonomía de algoritmos.
- [`Clase 2 -  Introducción AA1.pptx`](slides/Clase 2 -  Introducción AA1.pptx) — segunda clase teórica: pipeline de trabajo, overfitting y regularización.
- [`scikit-learn.ipynb`](notebooks/scikit-learn.ipynb) — pipeline básico con el dataset Iris y `LogisticRegression`.
- [`overfitting.ipynb`](notebooks/overfitting.ipynb) — sobreajuste y subajuste: regresión lineal vs. árbol de decisión vs. Random Forest.
- [`tic-tac-toe.ipynb`](TaTeTi_ Ejemplo de Reinforcement learning/tic-tac-toe.ipynb) — entrenamiento de un agente con Q-Learning por auto-juego.
- [`ticTacToe.py`](TaTeTi_ Ejemplo de Reinforcement learning/ticTacToe.py) — implementación del juego TaTeTi con interfaz `tkinter`.
- Artefactos auxiliares del ejemplo de RL: [`board.png`](TaTeTi_ Ejemplo de Reinforcement learning/board.png), [`policy_p1`](TaTeTi_ Ejemplo de Reinforcement learning/policy_p1) y [`policy_p2`](TaTeTi_ Ejemplo de Reinforcement learning/policy_p2) (Q-tables serializadas con `pickle`).

---

## Índice

1. [¿Qué es el Aprendizaje Automático?](#1-qué-es-el-aprendizaje-automático)
2. [Taxonomía de algoritmos](#2-taxonomía-de-algoritmos)
3. [El pipeline estándar de scikit-learn](#3-el-pipeline-estándar-de-scikit-learn)
4. [Sesgo y varianza (*bias-variance tradeoff*)](#4-sesgo-y-varianza-bias-variance-tradeoff)
5. [Sobreajuste y subajuste en la práctica](#5-sobreajuste-y-subajuste-en-la-práctica)
6. [Validación y remuestreo](#6-validación-y-remuestreo)
7. [Introducción al Aprendizaje por Refuerzo](#7-introducción-al-aprendizaje-por-refuerzo)
8. [El ejemplo TaTeTi](#8-el-ejemplo-tateti)
9. [Buenas prácticas para arrancar en ML](#9-buenas-prácticas-para-arrancar-en-ml)

---

## 1. ¿Qué es el Aprendizaje Automático?

El **Aprendizaje Automático** (*Machine Learning*, ML) es la subdisciplina de la inteligencia artificial que estudia algoritmos capaces de **mejorar su desempeño en una tarea a partir de datos**, sin estar explícitamente programados para cada caso. La definición canónica es la de **Mitchell (1997)**:

> *Se dice que un programa **aprende** de la experiencia $E$ con respecto a una clase de tareas $T$ y una medida de desempeño $P$, si su desempeño en $T$, medido por $P$, mejora con la experiencia $E$.*

Formalmente, un sistema de ML consta de tres componentes:

- **Función hipótesis (modelo):** una familia paramétrica $f_\theta : \mathcal{X} \to \mathcal{Y}$ que aproxima la función desconocida que mapea entradas a salidas.
- **Función de pérdida (*loss*):** $\ell(y, \hat y)$ que cuantifica el error en una predicción individual. La pérdida agregada sobre el dataset es típicamente
  $$L(\theta) = \frac{1}{n}\sum_{i=1}^n \ell\big(y_i, f_\theta(x_i)\big).$$
- **Optimizador:** el procedimiento que ajusta $\theta$ para minimizar $L(\theta)$. En esta unidad veremos el caso simple de regresión lineal.

A diferencia de la programación clásica —donde el humano escribe reglas explícitas—, en ML **las reglas se inducen a partir de ejemplos**. Esto es útil cuando:

- Las reglas son difíciles de codificar a mano (reconocimiento de imágenes, traducción).
- La escala del problema es enorme (millones de transacciones).
- El entorno cambia con el tiempo (sistemas de recomendación).

---

## 2. Taxonomía de algoritmos

Los algoritmos de ML se agrupan en tres grandes paradigmas según el tipo de señal disponible durante el entrenamiento.

### 2.1. Aprendizaje supervisado

Disponemos de pares $(x_i, y_i)$ donde $y_i$ es la **etiqueta verdadera**. El objetivo es aprender una función $f$ que prediga $y$ para nuevas $x$. Se subdivide en:

- **Regresión:** $y \in \mathbb{R}$ (precio, temperatura, demanda).
- **Clasificación binaria:** $y \in \{0, 1\}$ (spam/no-spam, fraude/no-fraude).
- **Clasificación multiclase:** $y \in \{1, \dots, K\}$ con $K > 2$ (dígitos, especies).
- **Clasificación multi-etiqueta:** cada ejemplo puede tener varias etiquetas simultáneamente.

### 2.2. Aprendizaje no supervisado

Sólo tenemos $x_i$, sin etiquetas. Buscamos **estructura** en los datos:

- **Clustering** — agrupar ejemplos similares ($k$-means, DBSCAN, hierarchical clustering).
- **Reducción de dimensionalidad** — proyectar a un espacio de menor dimensión preservando estructura (PCA, t-SNE, UMAP).
- **Detección de anomalías** — identificar observaciones que se apartan del patrón general (Isolation Forest, One-Class SVM).

### 2.3. Aprendizaje por refuerzo

Un **agente** toma decisiones en un **entorno** y recibe **recompensas**. El objetivo es aprender una **política** $\pi$ que maximice la recompensa acumulada. Ejemplos: juegos (AlphaGo), robótica, sistemas de recomendación, manejo de inventarios.

---

## 3. El pipeline estándar de scikit-learn

El notebook [`scikit-learn.ipynb`](notebooks/scikit-learn.ipynb) introduce el flujo de trabajo canónico en `scikit-learn`. Todos los estimadores comparten el mismo contrato:

```python
estimator.fit(X_train, y_train)        # aprende los parámetros
y_pred = estimator.predict(X_test)     # predice sobre datos nuevos
score = estimator.score(X_test, y_test) # evalúa con la métrica por defecto
```

Los pasos esenciales son:

### 3.1. Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify preserva proporciones de clase
)
```

La semilla (`random_state`) fija la partición para que el experimento sea **reproducible**.

### 3.2. Preprocesamiento dentro de un `Pipeline`

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),          # .fit_transform en train, .transform en test
    ('classifier', LogisticRegression()),  # .fit en train, .predict en test
])
```

Reglas clave:

- **`fit_transform`** se aplica sólo al train set (calcula estadísticas y transforma).
- **`transform`** se aplica al test set (reutiliza las estadísticas del train).
- Encapsular todo en un `Pipeline` evita **data leakage**: garantiza que el preprocesamiento se entrena sólo con datos de train y se aplica igual en test (y en producción).
- En problemas con tipos de columnas heterogéneos se usa `ColumnTransformer` para aplicar preprocesamientos distintos a subconjuntos de features, y se compone dentro de un `Pipeline`.

### 3.3. Ejemplo end-to-end con Iris

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=200))])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['setosa','versicolor','virginica']))
```

---

## 4. Sesgo y varianza (*bias-variance tradeoff*)

Para cualquier modelo $\hat f$ entrenado sobre un dataset finito, el error esperado en un punto $x$ se descompone en tres términos independientes:

$$\mathbb{E}\!\left[\big(y - \hat f(x)\big)^2\right] = \underbrace{\text{Bias}^2\!\big(\hat f(x)\big)}_{\text{error de simplificación}} + \underbrace{\text{Var}\!\big(\hat f(x)\big)}_{\text{error de sensibilidad}} + \underbrace{\sigma^2}_{\text{ruido irreducible}}.$$

- **Sesgo²:** error por hacer suposiciones demasiado simples. Disminuye con modelos más flexibles.
- **Varianza:** sensibilidad del modelo a la muestra particular de entrenamiento. Aumenta con la flexibilidad.
- **Ruido irreducible** $\sigma^2$: varianza intrínseca de la variable objetivo, no eliminable.

**Tradeoff:** a medida que crece la complejidad del modelo, el sesgo baja pero la varianza sube. El objetivo es encontrar la complejidad que minimiza la suma. En la práctica esto se controla con:

- **Regularización** (L1/L2/dropout).
- **Selección de features** (menos features ⇒ menos varianza).
- **Ensemble** (bagging reduce varianza, boosting reduce sesgo).
- **Más datos** (reduce varianza sin tocar el sesgo).

**Diagnóstico con *learning curves*:** graficar la métrica en train y validación a medida que crece el tamaño del train set.

- Si ambas convergen a un valor alto → **subajuste** (alto sesgo). Probar un modelo más expresivo o mejores features.
- Si hay una brecha grande entre train (bajo) y validación (alto) → **sobreajuste** (alta varianza). Regularizar, juntar más datos, o reducir complejidad.

---

## 5. Sobreajuste y subajuste en la práctica

El notebook [`overfitting.ipynb`](notebooks/overfitting.ipynb) ilustra el fenómeno con un dataset unidimensional pequeño. Se entrenan cuatro modelos y se comparan las métricas en train y test:

| Modelo | MSE train | MSE test | R² train | R² test |
|---|---|---|---|---|
| `LinearRegression` | 1.26 | 1.85 | 0.44 | −6.38 |
| `DecisionTreeRegressor` (sin podar) | 0.0 | 2.5 | 1.0 | −9.0 |
| `DecisionTreeRegressor(max_depth=2)` | intermedio | intermedio | intermedio | intermedio |
| `RandomForestRegressor(n_estimators=1000)` | (mejor generalización) | — | — | — |

**Lectura del cuadro:**

- El **árbol sin restricciones** ajusta perfectamente el train (MSE=0, R²=1) pero **memoriza** el ruido y generaliza pésimamente. Es el caso textbook de **sobreajuste**.
- La **regresión lineal** es demasiado rígida: el R² en test es negativo, indicando que el modelo es **peor que predecir la media**. Es **subajuste**.
- El **árbol con `max_depth=2`** es un compromiso: menos flexible, generaliza mejor.
- El **Random Forest** promedia muchos árboles sobre subconjuntos del dataset (bagging), lo que **reduce la varianza** sin sacrificar flexibilidad.

**Conclusión:** la complejidad del modelo debe controlarse. Herramientas estándar: `max_depth`, `min_samples_leaf`, `min_samples_split`, número de árboles, regularización, early stopping.

---

## 6. Validación y remuestreo

Más allá del hold-out train/test, en ML se usan esquemas de remuestreo para **estimar el desempeño fuera de muestra** y **seleccionar hiperparámetros** sin "gastar" el test set.

### 6.1. Hold-out split

División única (e.g. 80/20). Es la opción más simple pero la estimación tiene alta varianza (depende de qué ejemplos cayeron en cada lado).

### 6.2. *k*-fold cross-validation

Se particionan los datos en $k$ folds; cada fold se usa una vez como validación y los $k-1$ restantes como entrenamiento. La métrica final es el promedio:

$$\text{CV}(k) = \frac{1}{k}\sum_{j=1}^{k} \text{métrica}_j.$$

Valores típicos: $k = 5$ o $k = 10$.

### 6.3. Stratified *k*-fold

Mantiene la proporción de clases en cada fold. **Esencial en problemas desbalanceados** (ver [Unidad 3 — Modelos Lineales para Clasificación](unidad_3) para más detalle). En `scikit-learn`:

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 6.4. Leave-One-Out (LOO)

Caso extremo: $k = n$. Cada ejemplo se valida una vez. **Carísimo** en datasets grandes, pero útil cuando $n$ es chico.

### 6.5. Validación cruzada anidada

Si usamos CV para elegir hiperparámetros **y** CV para estimar desempeño, necesitamos **dos loops anidados**: el interno optimiza hiperparámetros, el externo estima desempeño. Sin el loop externo, la estimación está **optimista** porque los hiperparámetros fueron elegidos sobre los mismos datos que ahora estamos evaluando.

---

## 7. Introducción al Aprendizaje por Refuerzo

El **Aprendizaje por Refuerzo** (*Reinforcement Learning*, RL) es el tercer paradigma del ML. A diferencia del supervisado, no hay etiquetas explícitas: el agente aprende por **ensayo y error** interactuando con un entorno.

### 7.1. Proceso de Decisión de Markov (MDP)

Un MDP se define por la tupla $(S, A, P, R, \gamma)$:

- $S$ — conjunto de **estados**.
- $A$ — conjunto de **acciones**.
- $P(s' \mid s, a)$ — probabilidad de transición al estado $s'$ luego de tomar la acción $a$ en el estado $s$.
- $R(s, a, s')$ — **recompensa** recibida.
- $\gamma \in [0, 1)$ — **factor de descuento** que pondera más el corto plazo que el largo plazo.

En cada paso $t$, el agente observa el estado $s_t$, elige una acción $a_t \sim \pi(\cdot \mid s_t)$ según su **política** $\pi$, recibe una recompensa $r_{t+1}$ y pasa a $s_{t+1}$.

### 7.2. Función de valor y de valor-acción

La **función de valor** de la política $\pi$ mide cuán buena es la recompensa esperada a largo plazo desde un estado:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \,\Big|\, s_0 = s\right].$$

La **función de valor-acción** (o *Q-función*) evalúa pares estado-acción:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_{t+1} \,\Big|\, s_0 = s, a_0 = a\right].$$

La política óptima cumple $Q^*(s, a) = \max_\pi Q^\pi(s, a)$, y la acción greedy sobre $Q^*$ es la política óptima:

$$\pi^*(s) = \arg\max_a Q^*(s, a).$$

### 7.3. Q-Learning

**Q-Learning** (Watkins, 1989) es un algoritmo *off-policy* que aprende $Q^*$ directamente, sin necesidad de un modelo del entorno. La actualización tras observar $(s, a, r, s')$ es:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[\,r + \gamma \max_{a'} Q(s', a') - Q(s, a)\,\big],$$

donde $\alpha \in (0, 1)$ es la **tasa de aprendizaje** y el término entre corchetes es el **error de diferencia temporal** (TD-error). Bajo condiciones estándar (todo par $(s, a)$ visitado infinitas veces, $\alpha$ decreciente), Q-Learning converge a $Q^*$ con probabilidad 1.

### 7.4. Exploración vs. explotación

El dilema central del RL: ¿exploro acciones nuevas (para descubrir mejores recompensas) o exploto las que ya conozco? La política $\varepsilon$-greedy es el equilibrio estándar:

$$a_t = \begin{cases} \arg\max_a Q(s_t, a) & \text{con prob. } 1 - \varepsilon, \\ \text{acción uniforme al azar} & \text{con prob. } \varepsilon. \end{cases}$$

Típicamente $\varepsilon$ se **decae** durante el entrenamiento: alto al principio (exploración) y bajo al final (explotación).

---

## 8. El ejemplo TaTeTi

El subdirectorio `TaTeTi_ Ejemplo de Reinforcement learning/` contiene un caso end-to-end de RL sobre el juego tres en raya. El juego es ideal para introducir RL porque:

- El espacio de estados es finito y pequeño: $|S| \le 3^9 \approx 2 \times 10^4$.
- Las recompensas son escasas y terminales (+1 gana, −1 pierde, 0 empate).
- El entorno es **determinista** (no hay $P$ estocástico).

### 8.1. El entorno: `ticTacToe.py`

El script [`ticTacToe.py`](TaTeTi_ Ejemplo de Reinforcement learning/ticTacToe.py) implementa el juego con una interfaz `tkinter` que dibuja el tablero como una grilla 3×3 de botones clickeables. Define:

- El **tablero** como una lista de 9 posiciones.
- Los **movimientos legales** (casillas vacías).
- La **detección de ganador** (filas, columnas, diagonales).
- La **interfaz gráfica** que invoca la política aprendida para decidir los movimientos del agente.

### 8.2. Entrenamiento: `tic-tac-toe.ipynb`

El notebook [`tic-tac-toe.ipynb`](TaTeTi_ Ejemplo de Reinforcement learning/tic-tac-toe.ipynb) entrena dos agentes por **auto-juego**: en cada episodio, ambos bandos comparten la misma Q-tabla y actualizan $Q$ después de cada movimiento. Los pasos del notebook son:

1. Definir las clases `State` (representa una configuración del tablero) y `Player` (implementa la política y el método de actualización).
2. Inicializar la Q-tabla como un `defaultdict` que mapea tuplas (estado, acción) a un valor inicial (típicamente 0).
3. Entrenar durante **~50.000 partidas**, decayendo $\varepsilon$ de 1.0 a ~0.1.
4. Serializar las políticas aprendidas con `pickle`:
   ```python
   import pickle
   with open('policy_p1', 'wb') as f: pickle.dump(player1.states_value, f)
   with open('policy_p2', 'wb') as f: pickle.dump(player2.states_value, f)
   ```
5. Permitir que un humano juegue contra el agente cargando la Q-tabla y eligiendo la mejor acción vía `argmax`.

### 8.3. Artefactos serializados

- [`policy_p1`](TaTeTi_ Ejemplo de Reinforcement learning/policy_p1) — Q-tabla del jugador 1 (X), arrancando.
- [`policy_p2`](TaTeTi_ Ejemplo de Reinforcement learning/policy_p2) — Q-tabla del jugador 2 (O), respondiendo.
- [`board.png`](TaTeTi_ Ejemplo de Reinforcement learning/board.png) — imagen estática del tablero (referencia visual; el GUI de `tkinter` redibuja en vivo).

Ambos pickles son `dict[hash(estado) -> float]` con los valores $Q(s, a)$ aprendidos. La política greedy es $\pi(s) = \arg\max_a Q(s, a)$.

### 8.4. Componentes del RL en el ejemplo

Resumiendo, los elementos del marco RL vistos en §7 aplicados al TaTeTi:

- **Estado ($s$)** — tupla de 9 elementos con valores en $\{-, X, O\}$.
- **Acción ($a$)** — índice de la casilla libre en $[0, 8]$.
- **Recompensa ($r$)** — $+1$ si gana, $-1$ si pierde, $0$ empate o jugada intermedia.
- **Función de valor $Q(s, a)$** — tabla aprendida por auto-juego.
- **Política $\pi(s)$** — `argmax` sobre $Q$ aprendido.
- **Exploración** — $\varepsilon$-greedy con decaimiento.

---

## 9. Buenas prácticas para arrancar en ML

Una checklist mínima antes de entrenar el primer modelo:

1. **Empezar con un baseline.** Predecir la media (regresión) o la clase mayoritaria (clasificación). Si tu modelo no le gana al baseline, no agrega valor.
2. **Validación honesta.** Train / validación / test separados; no mirar el test hasta el final.
3. **Sembrar semillas.** Fijar `random_state` en todos los estimadores y `np.random.seed` al principio del script.
4. **Encapsular en `Pipeline`.** Preprocesamiento + modelo en un solo objeto que se `fit`-ea sobre train y se aplica igual a test/producción.
5. **Monitorear overfitting.** Comparar métricas en train y validación. Si divergen, regularizar.
6. **Versionar todo.** Código (`git`), datos (DVC o similar), dependencias (`pip freeze` o `requirements.txt` con hashes).
7. **Documentar el experimento.** Hiperparámetros, métrica, semilla, dataset, fecha — para poder reproducir.

---

## Bibliografía

- Mitchell, T. (1997). *Machine Learning*. McGraw-Hill.
- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards*. PhD thesis, Cambridge.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly. [3rd ed., Cap. 1–4.]
- Scikit-learn User Guide — <https://scikit-learn.org/stable/user_guide.html>
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press. [2nd ed., disponible online.]