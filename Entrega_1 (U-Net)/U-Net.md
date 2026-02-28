# 1. La Arquitectura U-Net: Fundamentos y Mecánica

La arquitectura U-Net es el estándar de facto para la segmentación semántica (clasificación densa píxel por píxel). A diferencia de las redes de clasificación tradicionales que colapsan una imagen en un vector perdiendo la noción del espacio, U-Net mantiene y reconstruye la resolución espacial geométrica.

<div align="center">
  <img src="../imágenes/unet1.png" width="600">
  <p><em>Figura 1: Visualización arquitectónica de una U-Net estándar.</em></p>
</div>

### 1.1. Mecánica de Ingeniería

U-Net es una red neuronal convolucional (CNN) simétrica dividida en tres fases operativas principales. 

| Componente | Función Matemática y Mecánica |
| :--- | :--- |
| **1. Camino de Contracción (Encoder)** | Captura el contexto global. Aplica convoluciones sucesivas seguidas de activación ReLU ($f(x)=\max(0,x)$). Luego, aplica *max-pooling* para reducir la resolución espacial a la mitad, doblando la profundidad de los canales. |
| **2. Cuello de Botella (Bottleneck)** | Es el punto de máxima abstracción semántica y menor resolución espacial latente. |
| **3. Camino Expansivo (Decoder)** | Reconstruye la resolución espacial geométrica para una máscara de segmentación perfecta mediante convoluciones transpuestas (*upsampling*). |
| **4. Conexiones de Salto (Skip Connections)** | Es la innovación crítica. Concatenan mapas de características de alta resolución del encoder directamente con el decoder, fusionando el contexto global con detalles espaciales finos. |

---

### 1.2. Colapso de Vectores vs. Reconstrucción Espacial

Busquemos entender por qué es vital el enfoque de la U-Net. El "Vector Collapse" (el isomorfismo entre una matriz $3 \times 3$ y $\mathbb{R}^9$) destruye la noción topológica de vecindad que existe en el espacio bidimensional.

Supongamos una matriz de entrada $I \in \mathbb{R}^{3 \times 3}$ que representa una línea diagonal:
$$I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

| Clasificación Estándar (Vector Collapse) | Segmentación Densa (U-Net) |
| :--- | :--- |
| Antes de la capa densa, la matriz se aplana en 1D: <br> $V = [1, 0, 0, 0, 1, 0, 0, 0, 1]$. <br><br> En $\mathbb{R}^9$, la distancia entre el primer '1' y el segundo '1' es de 3 saltos. La red pierde la noción de conectividad (vecindad diagonal); **la geometría se destruye**. | El tensor se comprime en el *bottleneck*, pero la ruta expansiva y las *Skip Connections* fuerzan a la red a mapear las características de vuelta a la cuadrícula $\mathbb{Z}^2$. <br><br> La salida reconstruye el tensor $3 \times 3$ intacto, **preservando la relación de vecindad** para clasificar, por ejemplo, los bordes de un meningioma. |

<div align="center">
  <img src="../videos/skip_connections.gif" width="450">
  <p><em>Animación 1: El rol de las Skip Connections en la preservación de la topología.</em></p>
</div>

---

### 1.3. Ejemplo Dinámico: Max-Pooling Paso a Paso

El operador de Max-Pooling no solo abstrae características, sino que altera la dimensionalidad del tensor. Supongamos un canal de entrada $I \in \mathbb{R}^{4 \times 4}$:

$$I = \begin{bmatrix}
\mathbf{1} & \mathbf{3} & \mathit{2} & \mathit{1} \\
\mathbf{4} & \mathbf{2} & \mathit{0} & \mathit{5} \\
1 & 2 & \mathbf{6} & \mathbf{2} \\
3 & 1 & \mathbf{3} & \mathbf{8}
\end{bmatrix}$$

Usando un filtro de $2 \times 2$ con un salto (*stride*) de 2, el proceso mapea esta matriz a un espacio más grueso (*coarse grid*) $Y \in \mathbb{R}^{2 \times 2}$. La operación extrae el máximo topológico de cada sub-cuadrante:

* Cuadrante superior izquierdo: $\max(1, 3, 4, 2) = 4$
* Cuadrante superior derecho: $\max(2, 1, 0, 5) = 5$
* Cuadrante inferior izquierdo: $\max(1, 2, 3, 1) = 3$
* Cuadrante inferior derecho: $\max(6, 2, 3, 8) = 8$

Matriz resultante:
$$Y = \begin{bmatrix} 4 & 5 \\ 3 & 8 \end{bmatrix}$$

<div align="center">
  <img src="../imágenes/maxpool_ejemplo.png" width="450">
</div>

---

## 2. El Enfoque Matemático: U-Net como Problema de Control Óptimo

Más allá de su concepción heurística de implementación, se ha demostrado que la arquitectura U-Net no es un apilamiento arbitrario de tensores, sino la solución estructural a un problema de **control óptimo** acoplado con un método multigrilla (*Multigrid Method*) no lineal.

### 2.1. La Red Neuronal como un Sistema Dinámico

En el marco del control óptimo, la propagación hacia adelante (*forward pass*) de un tensor a través de las capas de la red neuronal se modela como la discretización de una Ecuación Diferencial Ordinaria (EDO). Tratando la profundidad de la red como un tiempo continuo $t \in [0, T]$, la transformación del mapa de características $x(t)$ se sujeta a un campo vectorial parametrizado por $\theta(t)$ (los pesos convolucionales):

$$\dot{x}(t) = \mathcal{F}(x(t), \theta(t)) \quad \text{o bien} \quad \frac{dx}{dt} = f(x(t), \theta(t), t)$$

Donde $x(0)$ es la imagen médica de entrada y $x(T)$ es la máscara de segmentación final. El objetivo del entrenamiento es encontrar la trayectoria óptima de $\theta(t)$ que minimice el coste funcional (ej. *Dice Loss*). Minimizar esta pérdida es equivalente a resolver un problema de Control Óptimo, donde el *Backpropagation* actúa calculando el estado adjunto (*costate*) del sistema bajo el Principio del Máximo de Pontryagin.

### 2.2. U-Net como un Ciclo-V (V-Cycle) Multigrilla

Optimizar tensores masivos de alta resolución (como un MRI) es numéricamente costoso; los gradientes se estancan tratando de corregir errores globales. Al resolver ecuaciones diferenciales, los métodos iterativos (ej. Gauss-Seidel) eliminan el error de alta frecuencia (ruido local), pero son ineficientes contra el error de baja frecuencia (desviaciones estructurales globales). La U-Net soluciona esto manipulando el dominio $\Omega$:

1. **Suavizado inicial (Smoother):** Las primeras convoluciones extraen características y eliminan el error local de alta frecuencia.
2. **Operador de Restricción ($I_h^{2h}$) (Encoder):** Matemáticamente, transfiere un vector del espacio fino $\Omega_h$ al grueso $\Omega_{2h}$ (Max-Pooling). Las variaciones globales se "comprimen" y se transforman en altas frecuencias relativas, permitiendo que las convoluciones profundas las resuelvan rápidamente.
    * *Ejemplo:* Si $x_h = [1, 5, 2, 8]$, al aplicar Max-Pooling de factor 2 se descompone en $\max(1,5)$ y $\max(2,8)$. Resulta en $x_{2h} = I_h^{2h} x_h = [5, 8]$.
3. **Operador de Prolongación ($I_{2h}^h$) (Decoder):** Interpola la solución calculada en el espacio latente grueso $\Omega_{2h}$ de vuelta a la grilla fina $\Omega_h$ (Convolución Transpuesta).
    * *Ejemplo:* Prolongar $x_{2h} = [5, 8]$ copiando al vecino más cercano produce un bloque difuso $x_h' = I_{2h}^h x_{2h} = [5, 5, 8, 8]$.
4. **Precondicionamiento de Error (Skip Connections):** Como vimos, la prolongación destruye la geometría original ($[5, 5, 8, 8] \neq [1, 5, 2, 8]$). Las conexiones de salto inyectan el estado exacto de la grilla fina ($x_h \oplus x_h'$), corrigiendo algorítmicamente el error de interpolación en cada nivel.

<div align="center">
  <img src="../videos/gif_grillas.gif" width="450">
  <p><em>Animación 2: Simulación de Operadores Multigrilla en un Ciclo-V.</em></p>
</div>

<div align="center">
  <img src="../imágenes/image_proceso.png" width="600">
</div>

---

## 3. La Evolución de la Restauración Espacial: SegNet y Seg-Unet

Las arquitecturas estándar difuminan los bordes finos. Para delimitar estructuras biológicas críticas, como tumores cerebrales, requerimos exactitud sub-píxel.

### 3.1. SegNet y Memoria Espacial

A diferencia de la U-Net que pasa densos mapas de características (pesados en memoria), SegNet transfiere **memoria espacial**. En el *max-pooling*, guarda los índices espaciales exactos $(p, q)$ del valor máximo. En el *decoder*, realiza un *unpooling* exacto, colocando los valores en sus coordenadas originales y rellenando el resto con ceros. Es como tener un "mapa del tesoro" que indica dónde estaba el objeto, en lugar de cargar con todo el objeto.

### 3.2. La Fusión: Seg-Unet

Seg-UNet combina la exactitud geométrica de SegNet (usando índices de unpooling) con la riqueza semántica de U-Net. Primero coloca los píxeles en su lugar geométrico exacto, y luego aplica las *Skip Connections* para rellenar la semántica y textura faltante con los mapas profundos del encoder.

<div align="center">
  <img src="../imágenes/upsampling.png" width="600">
</div>

### 3.3. Cálculo Analítico del Unpooling Exacto

Para evidenciar la diferencia geométrica, rastreemos la matriz de índices (*ArgMax*):

**1. Fase de Restricción (Max-Pooling + Tracking):**
Al aplicar Max-Pooling a $I \in \mathbb{R}^{4 \times 4}$, generamos simultáneamente una máscara $M$ que captura la coordenada del escalar victorioso.

$$I = \begin{bmatrix} 
1 & 3 & 2 & 1 \\ 
\mathbf{4} & 2 & 0 & \mathbf{5} \\ 
1 & 2 & \mathbf{6} & 2 \\ 
\mathbf{3} & 1 & 3 & 8 
\end{bmatrix} 
\xrightarrow{\text{Max-Pool}} 
Y = \begin{bmatrix} 4 & 5 \\ 3 & 8 \end{bmatrix}, 
\quad 
M = \begin{bmatrix} (1,0) & (1,3) \\ (3,0) & (3,3) \end{bmatrix}$$

**2. Fase de Prolongación (Max-Unpooling):**
El decodificador recibe un tensor latente $Z \in \mathbb{R}^{2 \times 2}$ actualizado (ej. tras las convoluciones profundas):

$$Z = \begin{bmatrix} 10 & 20 \\ 15 & 30 \end{bmatrix}$$

El *Max-Unpooling* inicializa un tensor nulo $\mathbb{R}^{4 \times 4}$ y utiliza los índices de $M$ para enrutar los valores de $Z$ a sus posiciones espaciales métricas exactas:

$$Z_{unpooled} = \begin{bmatrix}
0 & 0 & 0 & 0 \\
\mathbf{10} & 0 & 0 & \mathbf{20} \\
0 & 0 & \mathbf{30} & 0 \\
\mathbf{15} & 0 & 0 & 0
\end{bmatrix}$$

Las convoluciones subsecuentes rellenarán los ceros, pero el anclaje del borde biológico ha sido restaurado con precisión exacta.

---

### 3.4. El Fallo de las Métricas Locales (Dice / Cross-Entropy)

Aquí demostramos matemáticamente por qué optimizar el coeficiente de Dice ($\frac{2|X \cap Y|}{|X| + |Y|}$) no garantiza coherencia estructural.

| Ejemplo 1: El Anillo Vascular (Matriz 3x3) | Ejemplo 2: El Meningioma Sólido (Matriz 5x5) |
| :--- | :--- |
| El *Ground Truth* ($GT$) es un anillo continuo (1 agujero). La *Predicción* ($P$) falla en **un solo píxel** inferior.<br><br> $GT = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix} \quad P = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & \mathbf{0} & 1 \end{bmatrix}$<br><br> **Evaluación Métrica (Dice):**<br>$|GT|=8, |P|=7, |GT \cap P|=7$.<br>$Dice = \frac{14}{15} \approx$ **0.933 (93.3%)**<br>*¡Un modelo excelente a ojos de la pérdida!*<br><br>**Evaluación Topológica:**<br>$GT$ tiene un ciclo cerrado ($\beta_1 = 1$). En $P$, el anillo se rompió ($\beta_1 = 0$). **El error topológico es del 100%.** | Un meningioma es una masa sólida continua ($\beta_1 = 0$). La red predice casi todo, pero crea un **agujero falso** dentro del núcleo.<br><br> $GT = \begin{bmatrix} 0 & 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix} \quad P = \begin{bmatrix} 0 & 1 & 1 & 1 & 0 \\ 0 & 1 & \mathbf{0} & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{bmatrix}$ <br><br> **Evaluación Métrica (Dice):**<br>$|GT|=9, |P|=8, |GT \cap P|=8$.<br>$Dice = \frac{16}{17} \approx$ **0.941 (94.1%)**<br><br>**Evaluación Topológica:**<br>$P$ creó una cavidad inexistente ($\beta_1 = 1$). Modificó la firma geométrica severamente, lo cual es inaceptable para simulación pre-quirúrgica. |

<div align="center">
  <img src="../videos/video_ejemplos_final.gif" width="550">
  <p><em>Animación 3: Unpooling exacto y la vulnerabilidad de las métricas de error locales.</em></p>
</div>

---

## 4. El Salto Topológico: TDA-SegUNet y Geometría Global

### 4.1. El Problema: La "Amnesia" del Pooling Tradicional

Las redes CNN estándar asumen que cada píxel es independiente; **son ciegas a la topología**. En la segmentación de meningiomas (bordes difusos y tejidos adyacentes similares), una U-Net podría lograr un 99% de precisión por píxel. Sin embargo, como probamos matemáticamente, ese 1% de error puede crear un agujero falso. Para una métrica volumétrica el error es mínimo, pero para la planificación quirúrgica, el error estructural es catastrófico.

<div align="center">
  <img src="../imágenes/bordes.png" width="500">
</div>

### 4.2. La Filtración de Subnivel y TDA

Para resolver esto, integramos el **Análisis Topológico de Datos (TDA)** mediante homología persistente.
Imaginemos la RM como un paisaje montañoso (intensidad alta). El TDA inunda este paisaje con agua. A medida que el agua baja, aparecen "islas". Si las islas se unen formando un lago atrapado, nace un "agujero".

TDA-SegUNet ve esta conectividad. Si la red predice un tumor con un agujero central (genus > 0), pero el TDA sabe estructuralmente que los meningiomas son masas sólidas (genus 0), la red penaliza esa predicción estructuralmente.

### 4.3. Fundamentos e Implementación

Para cuantificar esto, se usan los **Números de Betti**:
* $\beta_0$: Número de componentes conectadas (masas de tejido).
* $\beta_1$: Número de agujeros 1-dimensionales (anillos o cavidades).

Al barrer el umbral de intensidad (filtración), registramos cuándo nacen y mueren las características topológicas, creando un **Diagrama de Persistencia**. Este diagrama es matemáticamente robusto al ruido (Teorema de Estabilidad).

Para que la U-Net lo pueda procesar, se convierte en una **Imagen de Persistencia (PI)** aplicando funciones Gaussianas 2D ponderadas por la vida útil de cada característica.

**TDA-SegUNet modifica su capa de entrada.** En lugar de recibir solo el canal de la imagen MRI, concatena los tensores volumétricos de las Imágenes de Persistencia ($\beta_0$ y $\beta_1$). 
La red aprende simultáneamente:
1. Los gradientes de intensidad locales (de la MRI).
2. Las reglas irrebatibles de topología global (de las PI).

<div align="center">
  <img src="../videos/evolucion_pool.gif" width="500">
</div>