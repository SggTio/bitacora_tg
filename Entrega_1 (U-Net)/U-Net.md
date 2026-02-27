## 1. La Arquitectura U-Net: Fundamentos y Mecánica
La arquitectura U-Net es el estándar de facto para la segmentación semántica (clasificación densa píxel por píxel). A diferencia de las redes de clasificación tradicionales que colapsan una imagen en un vector perdiendo la noción del espacio, U-Net mantiene y reconstruye la resolución espacial geométrica.

![visualización de una U-net](../imágenes/unet1.png)

### 1.1. Mecánica de Ingeniería:
U-Net es una red neuronal convolucional (CNN) simétrica dividida en tres fases operativas principales:Camino de Contracción (Encoder): Su objetivo es capturar el contexto global. Aplica convoluciones sucesivas seguidas de una función de activación no lineal, típicamente ReLU, definida como $f(x)=\max(0,x)$. Luego, aplica max-pooling para reducir la resolución espacial a la mitad, doblando la profundidad de los canales.Cuello de Botella (Bottleneck): Es el punto de máxima abstracción semántica y menor resolución espacial.Camino Expansivo (Decoder): Reconstruye la resolución espacial requerida para una máscara de segmentación perfecta mediante convoluciones transpuestas (upsampling).Conexiones de Salto (Skip Connections): Es la innovación crítica. Concatenan mapas de características de alta resolución del encoder directamente con el decoder. Esto fusiona el contexto global (baja resolución pero semántica profunda) con detalles espaciales finos (alta resolución geométrica).

*Ejemplo Dinámico: Max-Pooling Paso a Paso*
El max-pooling es un filtro que recorre una matriz quedándose solo con el valor dominante, abstrayendo la característica y reduciendo el tamaño.Si tenemos una matriz de $4 \times 4$:$$I = \begin{bmatrix} 1 & 3 & 2 & 1 \\ 4 & 2 & 0 & 5 \\ 1 & 2 & 6 & 2 \\ 3 & 1 & 3 & 8 \end{bmatrix}$$

Usando un filtro de $2 \times 2$ con un salto (stride) de 2, la operación extrae el máximo de cada sub-cuadrante:Cuadrante superior izquierdo: $\max(1, 3, 4, 2) = 4$Cuadrante superior derecho: $\max(2, 1, 0, 5) = 5$Cuadrante inferior izquierdo: $\max(1, 2, 3, 1) = 3$Cuadrante inferior derecho: $\max(6, 2, 3, 8) = 8$Matriz resultante $2 \times 2$:$$Y = \begin{bmatrix} 4 & 5 \\ 3 & 8 \end{bmatrix}$$

Ejemplo visual:
![ejemplo de maxpooling](../imágenes/maxpool_ejemplo.png)


## 2. El Enfoque Matemático: U-Net como Problema de Control Óptimo
Más allá de su concepción de implementación, se ha demostrado que la arquitectura U-Net no es un apilamiento arbitrario de tensores, sino la solución estructural a un problema de control óptimo acoplado con un método multigrilla (Multigrid Method) no lineal.

### 2.1. La Red Neuronal como un Sistema Dinámico

En el marco del control óptimo, la propagación de un tensor a través de las capas de la red neuronal se modela como la discretización de una Ecuación Diferencial Ordinaria (EDO). Si tratamos la profundidad de la red como un tiempo continuo $t \in [0, T]$, la transformación del mapa de características $x(t)$ se rige por:$$\frac{dx}{dt} = f(x(t), \theta(t), t)$$Donde $x(0)$ es la imagen médica de entrada y $x(T)$ es la máscara de segmentación final. El objetivo del entrenamiento es encontrar la trayectoria de los parámetros (los pesos) $\theta(t)$ que minimicen una función de coste funcional (la pérdida empírica, como el Coeficiente Dice), sujeta a la dinámica del sistema. 

En este paradigma, el algoritmo de backpropagation no es más que la resolución de las ecuaciones adjuntas del Principio del Máximo de Pontryagin.

(explicar más sobre Pontryagin)

### 2.2. U-Net como un Ciclo-V (V-Cycle) Multigrilla

El problema de optimizar tensores masivos de alta resolución (como un corte de MRI) es extremadamente costoso, ya que los gradientes se estancan tratando de corregir errores globales. Aquí es donde la U-Net emula exactamente un Algoritmo Multigrilla en Ciclo-V.

Al resolver ecuaciones diferenciales espaciales, los solucionadores iterativos (como Gauss-Seidel por poner un ejempl) eliminan rápidamente el error de alta frecuencia (ruido local), pero son ineficientes contra el error de baja frecuencia (desviaciones estructurales globales).

La U-Net soluciona esto manipulando la resolución del dominio:

- Suavizado inicial: Las primeras convoluciones eliminan el error local.

- Restricción (Encoder): El max-pooling proyecta el dominio espacial a una grilla más gruesa. En esta grilla reducida, las variaciones globales de baja frecuencia se "comprimen" y se transforman en altas frecuencias, permitiendo que las convoluciones profundas las capturen y resuelvan eficientemente.

- Prolongación (Decoder): La solución aproximada en la grilla gruesa se interpola de vuelta a la grilla fina mediante convoluciones transpuestas.

- Precondicionamiento de Error (Skip Connections): La interpolación pura difumina la geometría. Las conexiones de salto actúan como un precondicionador matemático: inyectan la solución exacta de los bordes de la grilla fina directamente en el paso de prolongación, corrigiendo el error de interpolación en cada nivel.


| Concepto de Control Óptimo / Multigrilla | Equivalente en la Arquitectura U-Net | Función Matemática y Mecánica |
| :--- | :--- | :--- |
| **Operador de Restricción** ($I_{h}^{2h}$) | **Encoder** (Max-Pooling $2 \times 2$) | Mapea el estado del sistema de una grilla fina espacial a una gruesa. Filtra altas frecuencias espaciales para aislar la semántica global. |
| **Operador de Prolongación** ($I_{2h}^{h}$) | **Decoder** (Convolución Transpuesta) | Interpola la corrección del error desde el espacio latente abstracto de vuelta a la resolución espacial original para la clasificación densa. |
| **Precondicionador / Inyección de Estado** | **Skip Connections** (Concatenación) | Transfiere el estado exacto de alta frecuencia ($x_{fina}$) a la fase de reconstrucción, evitando la pérdida de información del operador de prolongación. |
| **Operador de Suavizado (Smoother)** | **Bloques Convolucionales** (Conv + ReLU) | Relaja el sistema iterativamente. Extrae características locales y "suaviza" el gradiente de error en la escala espacial actual. |
| **Minimización Funcional** ($\min J(\theta)$) | **Optimización de Pérdida** (Ej. Dice Loss) | El objetivo de control: ajustar los pesos $\theta$ para que la salida de la EDO alcance la matriz objetivo (la máscara real de tejido). |

![Así se ve el proceso](../imágenes/image_proceso.png)

## 3. La Evolución de la Restauración Espacial: SegNet y Seg-Unet

Las arquitecturas estándar difuminan los bordes finos. Para estructuras biológicas críticas, requerimos exactitud sub-pixel.

### 3.1. SegNet: Memoria Espacial

En lugar de descartar datos en el max-pooling, SegNet guarda los índices espaciales exactos $(p, q)$ del valor máximo. En el decoder, hace un unpooling exacto, colocando los valores de vuelta en sus coordenadas originales y rellenando con ceros.Ejemplo de Unpooling (Restauración Exacta):Si en el paso de max-pooling anterior guardamos la posición del '8' (que estaba en la esquina inferior derecha de su cuadrante), al hacer el unpooling de la matriz $2 \times 2$ de vuelta a $4 \times 4$, ese '8' regresará exactamente a la coordenada $(4,4)$, preservando la geometría original de la imagen médica.

SegNet: A diferencia de la U-Net que pasa mapas de características densos (pesados en memoria), SegNet: solo pasa los índices (coordenadas). Es como tener un mapa donde está el bojeto en vez de cargar con el objeto.

### 3.2. Seg-Unet

Combina la geometría de SegNet (usando índices de unpooling) con la riqueza semántica de U-Net (concatenando los mapas profundos del encoder sobre esa grilla ya restaurada).

Dicho de otra forma, Usa los índices de SegNet para colocar los píxeles en su lugar geométrico exacto (unpooling) y luego aplica las conexiones de salto (skip connections) de U-Net para rellenar la semántica y textura.

![Cómo se ve el proceso de desempacar el codificador (encoder - Decoder)](../imágenes/upsampling.png)

## 4. El Salto Topológico: TDA-SegUNet y Geometría Global

### El problema: La "Amnesia" del Pooling Tradicional

En una arquitectura U-Net o CNN estándar, el Max-Pooling actúa como un filtro de abstracción agresivo. Al quedarse solo con el valor máximo de una ventana (ej. 2x2), la red "olvida" en qué píxel exacto residía esa intensidad. Cuando el Decoder intenta reconstruir la imagen mediante Upsampling Bilineal o Convoluciones Transpuestas, se ve obligado a "adivinar" o promediar la posición de los datos. El resultado es un borde difuso que en las aplicaciones médicas puede implicar tejido valioso o delicado.

![Ejemplo de bordes difusos](../imágenes/image_proceso.png)

El gran problema de las CNN tradicionales es que optimizan funciones de pérdida (como Dice o Cross-Entropy) que asumen que cada píxel es independiente. Son ciegas a la topología.

El Problema Clínico: En la segmentación de estructuras complejas como los meningiomas, un tumor puede tener bordes difusos y tejidos adyacentes muy similares. Una U-Net estándar podría lograr un 99% de precisión por píxel, pero ese 1% de error podría predecir un "agujero" falso en medio de la masa tumoral. Para una métrica volumétrica, el error es mínimo; para la topología y la planificación quirúrgica, el error estructural es catastrófico.Para resolver esto, integramos el Análisis Topológico de Datos (TDA) mediante homología persistente.

### El Salto Topológico que ofrece TDA-Seg-Unet:

 La Filtración de Subnivel
 
 Imaginemos que la imagen de la RM es un paisaje montañono (intensidad alta). El TDA inunda este paisaje con agua (umbral de intensidad). A medida que el agua baja, aparecen "islas" ($\beta_0$: componentes conexas).
 Si las islas se unen formando un lago atrapado, nace un "agujero" ($\beta_1$).
 El Diagrama de Persistencia registra cuánto tiempo sobrevive cada isla o lago antes de fusionarse.

 ¿Qué nos ofrece TDA-SegUnet?
 TDA-SegUNet ve conectividad. Si la red predice un tumor con un agujero en el medio (genus > 0), y el TDA sabe que los meningiomas suelen ser masas sólidas (genus 0), la red penaliza esa predicción estructuralmente.

### 4.1. Fundamentos Topológicos (Números de Betti)
$\beta_0$: Número de componentes conectadas (masas de tejido).$\beta_1$: Número de agujeros 1-dimensionales (anillos o cavidades).

### 4.2. Filtraciones e Imágenes de Persistencia

Al barrer un umbral de intensidad sobre la imagen (filtración), registramos cuándo "nace" y "muere" una característica topológica. Esto genera un Diagrama de Persistencia. Gracias al Teorema de Estabilidad, sabemos que este diagrama es matemáticamente robusto al ruido.Para que la red neuronal lo procese, el diagrama se convierte en una Imagen de Persistencia (PI) aplicando funciones Gaussianas bidimensionales ponderadas por la vida útil (persistencia) de cada característica.

4.3. Implementación: TDA-SegUNetLa red modifica su capa de entrada. En lugar de recibir solo el canal de la imagen MRI, concatena los tensores de las Imágenes de Persistencia ($\beta_0$ y $\beta_1$).La red aprende simultáneamente:Los gradientes de intensidad locales (de la MRI).Las reglas irrebatibles de topología global (de las PI).