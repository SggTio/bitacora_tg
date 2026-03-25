---
layout: post
title: "Redes Neuronales Convolucionales y U-Net: La Máquina de Segmentación"
date: 2026-03-24
categories: [deep-learning, arquitecturas]
tags: [CNN, U-Net, segmentacion, redes-neuronales, encoder-decoder]
math: true
---

# Redes Neuronales Convolucionales y U-Net: La Máquina de Segmentación

En este artículo realizamos un viaje desde los fundamentos de las redes neuronales artificiales hasta la arquitectura U-Net, que revolucionó la segmentación de imágenes médicas. Entender cómo funcionan estas máquinas no solo es esencial para aplicaciones de segmentación de tumores cerebrales, sino que también proporciona intuición sobre cómo las máquinas "ven" el mundo.

## 1. De las Neuronas Artificiales a las CNNs

### El Perceptrón: El Punto de Partida

Toda red neuronal comienza con una pregunta simple: ¿cómo puede una máquina aprender de los datos?

La respuesta más básica es el **perceptrón**, una neurona artificial que implementa la ecuación:

$$y = \sigma(w \cdot x + b)$$

donde:
- **x** es la entrada (vector)
- **w** es el vector de pesos que la red aprenderá
- **b** es el sesgo (bias)
- **σ** es una función de activación (no linealidad)

Esta ecuación es extraordinariamente simple, pero increíblemente poderosa. Dice: "toma todas las entradas, multiplícalas por pesos, suma, añade un sesgo, y aplica una función no lineal". Para un problema de dos dimensiones, esto define una línea (o hiperplano) que separa las clases.

### El Problema: ¿Por Qué las Redes Totalmente Conectadas Fallan?

Ahora imaginemos que queremos clasificar imágenes. Una imagen de 256×256 píxeles en escala de grises es un vector de 65,536 números. Si conectamos esto directamente a una red neuronal totalmente conectada (fully connected, FC), obtenemos un desastre:

#### EJEMPLO NUMÉRICO: Explosión de Parámetros

**Red Totalmente Conectada (FC):**
- Entrada: 256 × 256 = 65,536 píxeles
- Primera capa oculta: 1,024 neuronas
- Número de pesos: 65,536 × 1,024 = **67,108,864 parámetros**
- Parámetros de sesgo: 1,024
- **Total para UNA capa: ~67 millones de parámetros**

Y esto es solo la primera capa. Para una red con 3 capas ocultas de 512 neuronas cada una:

Total ≈ 65,536 × 512 + 512 × 512 + 512 × 512 + 512 × 256 ≈ **35 millones de parámetros**

**Red Convolucional (CNN):**
- Filtro 3×3: 9 pesos
- Si aplicamos 16 filtros: 16 × 9 = **144 parámetros**
- Resultado: 65,536 × 144 = **Reducción de 465,000 veces en la primera capa**

Este es el poder de las convoluciones.

### Tres Ideas Clave que Definen las CNNs

Las CNNs resuelven el problema de parámetros mediante tres principios elegantes:

**1. Receptores Locales (Local Receptive Fields)**

En lugar de conectar cada píxel a cada neurona, dividimos la imagen en pequeñas ventanas (típicamente 3×3 o 5×5 píxeles). Cada neurona en la primera capa convolucional solo "ve" estos píxeles locales. Una neurona aprende a detectar patrones locales: bordes, esquinas, texturas.

**2. Compartición de Pesos (Weight Sharing)**

El mismo filtro 3×3 se aplica en *toda* la imagen, deslizándose sobre ella. Esto significa que si aprendemos a detectar un borde horizontal en una esquina, ese detector funciona igual en el centro de la imagen. Un patrón es un patrón, sin importar dónde esté.

**3. Equivarianza Traslacional**

Si un patrón se desplaza un poco en la imagen, la activación también se desplaza (pero mantiene su forma). Esto hace que las CNNs sean naturalmente robustas a pequeñas traslaciones.

| Característica | Red FC | CNN |
|---|---|---|
| Parámetros (256×256 → 512) | 33.6 millones | 1,440 |
| Capaz de compartición de pesos | No | Sí |
| Equivarianza traslacional | No | Sí |
| Memoria requerida | Alta | Baja |
| Generalización a imágenes nuevas | Pobre | Excelente |

## 2. Anatomía de una CNN

### Capas Convolucionales: Los Detectores Aprendidos

Una capa convolucional aplica K filtros aprendibles a una entrada. Si la entrada tiene forma H × W × C (alto, ancho, canales) y aplicamos K filtros de tamaño f × f × C, la salida tiene forma:

$$(H - f + 1) \times (W - f + 1) \times K$$

(asumiendo stride 1 y sin padding)

**Operación matemática:**

$$Y[i,j,k] = \sum_{a=0}^{f-1} \sum_{b=0}^{f-1} \sum_{c=0}^{C-1} W[a,b,c,k] \cdot X[i+a, j+b, c] + b_k$$

donde **W** es el tensor de pesos y **b_k** es el sesgo del k-ésimo filtro.

### Funciones de Activación: Inyectando No-Linealidad

Sin funciones de activación no lineales, una red neuronal es solo una composición de transformaciones lineales, que equivale a una sola transformación lineal. Las funciones de activación rompen esta linealidad.

| Activación | Fórmula | Rango | Cuándo Usar |
|---|---|---|---|
| **ReLU** | $f(x) = \max(0, x)$ | [0, ∞) | Capas ocultas (estándar) |
| **Leaky ReLU** | $f(x) = x$ si $x > 0$, $\alpha x$ si $x \leq 0$ | ℝ | Capas ocultas (evita neurona muerta) |
| **GELU** | $f(x) = x \cdot \Phi(x)$ | ≈ ℝ | Transformers, arquitecturas modernas |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | (0, 1) | Salida binaria |
| **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | Ocasionalmente en salidas |
| **Softmax** | $f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | (0, 1), suma=1 | Salida multiclase |

**ReLU es el estándar** por su simplicidad y porque resuelve el problema del desvanecimiento de gradientes que afligía a redes más antiguas con Sigmoid.

### Capas de Pooling: Compresión y Abstraccíón

Las capas de pooling reducen las dimensiones espaciales, manteniendo la información más importante. El **max-pooling** es el más común.

#### EJEMPLO NUMÉRICO DETALLADO: Max-Pooling 2×2

Supongamos que tenemos una matriz 4×4:

$$X = \begin{bmatrix}
1 & 3 & 2 & 4 \\
5 & 7 & 1 & 2 \\
2 & 4 & 9 & 1 \\
3 & 8 & 2 & 6
\end{bmatrix}$$

Aplicamos max-pooling con ventana 2×2 y stride 2. Dividimos la imagen en 4 cuadrantes 2×2 no superpuestos:

**Cuadrante 1 (arriba-izquierda):**
$$\begin{bmatrix} 1 & 3 \\ 5 & 7 \end{bmatrix} \rightarrow \max = 7$$

**Cuadrante 2 (arriba-derecha):**
$$\begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix} \rightarrow \max = 4$$

**Cuadrante 3 (abajo-izquierda):**
$$\begin{bmatrix} 2 & 4 \\ 3 & 8 \end{bmatrix} \rightarrow \max = 8$$

**Cuadrante 4 (abajo-derecha):**
$$\begin{bmatrix} 9 & 1 \\ 2 & 6 \end{bmatrix} \rightarrow \max = 9$$

**Resultado:**
$$Y = \begin{bmatrix}
7 & 4 \\
8 & 9
\end{bmatrix}$$

Hemos reducido de 4×4 (16 valores) a 2×2 (4 valores), manteniendo los máximos locales que suelen corresponder a características visuales importantes.

### Normalización de Lotes (Batch Normalization)

Batch normalization normaliza los valores en cada lote durante el entrenamiento:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

donde:
- $\mu_B$ es la media del lote
- $\sigma_B^2$ es la varianza del lote
- $\epsilon$ es una pequeña constante para evitar división por cero

Intuitivamente: mantiene los valores en un rango razonable, acelera el entrenamiento y actúa como regularizador.

### Dropout: Prevención del Sobreajuste

Durante el entrenamiento, dropout desactiva aleatoriamente una fracción **p** de las neuronas. Esto obliga a la red a desarrollar representaciones redundantes. Durante la inferencia, todas las neuronas están activas pero se escalan por (1-p).

## 3. El Problema de la Segmentación Semántica

### Clasificación vs. Segmentación

**Clasificación:** Una imagen de un tumor cerebral recibe una etiqueta única: "tumor presente" o "sano". Un número: **1 etiqueta → 1 imagen**.

**Segmentación:** Cada píxel de la imagen recibe su propia etiqueta. Una imagen de 256×256 recibe 65,536 etiquetas (una por píxel). **1 etiqueta → 1 píxel**.

| Aspecto | Clasificación | Segmentación |
|---|---|---|
| Entrada | Imagen H × W × C | Imagen H × W × C |
| Salida | Vector de K clases | Mapa H × W × K |
| Interpretación | "¿Qué hay?" | "¿Dónde está cada parte?" |
| Métrica | Precisión (%) | Dice, IoU (%) |
| Complejidad | Media | Alta |

### El Problema del "Colapso Vectorial" (Vector Collapse)

Las redes completamente conectadas clasificadoras típicamente hacen esto:

1. Aplican convoluciones para extraer características → Forma: 8×8×128
2. **Aplanan (flatten)** el resultado → Forma: 8,192 valores
3. Pasan por capas FC → Salida: vector de K clases

El aplanamiento **destruye permanentemente la información espacial**. Si queremos recuperar la ubicación de cada píxel, es demasiado tarde: los píxeles fueron mezclados en un vector 1D.

#### EJEMPLO NUMÉRICO: Cómo el Aplanamiento Destruye la Estructura

Consideremos una matriz diagonal 3×3 (representa una característica en la diagonal principal):

$$X = \begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 3
\end{bmatrix}$$

La estructura diagonal es clara: hay valores en (0,0), (1,1), (2,2). Esta es información **topológica**: la conectividad de los píxeles.

Cuando aplanamos:

$$\text{flatten}(X) = [1, 0, 0, 0, 2, 0, 0, 0, 3]$$

La estructura espacial se perdió. Una red FC sabe que hay valores 1, 2, 3, pero ha olvidado que formaban una diagonal. Si la salida necesita ser una máscara con la estructura diagonal recuperada, la red debe aprender esta estructura desde cero (difícil) o adivinarlo (malo).

**Para segmentación, necesitamos una arquitectura que preserve la estructura espacial.**

## 4. La Arquitectura U-Net

### Contexto Histórico

En 2015, Olaf Ronneberger, Philipp Fischer y Thomas Brox publicaron "U-Net: Convolutional Networks for Biomedical Image Segmentation" (arxiv.org/abs/1505.04597). El trabajo fue revolucionario porque:

1. Introdujo el camino "expandente" (decoder) para recuperar resolución
2. Introdujo **skip connections** para recuperar detalles finos
3. Mostró que estructuras en forma de U funcionan excepcionalmente bien
4. Fue entrenado con datos limitados y aún así generalizó bien

Hoy, U-Net es la arquitectura de referencia para segmentación médica.

### La Forma "U": Una Estructura Elegante

```
INPUT (256x256)
    ↓
[Conv] (256x256x64)
    ↓
[MaxPool] (128x128x64)
    ↓
[Conv] (128x128x128)
    ↓
[MaxPool] (64x64x128)
    ↓
[Conv] (64x64x256) ← BOTTLENECK
    ↓
[UpSample] (128x128x256)
    ↓
[Conv] (128x128x128)
    ↓
[UpSample] (256x256x128)
    ↓
[Conv] (256x256x1) ← OUTPUT
```

La forma de U viene de:
- **Lado izquierdo (Encoder):** camino de contracción, aumenta profundidad, reduce resolución
- **Fondo (Bottleneck):** compresión máxima
- **Lado derecho (Decoder):** camino de expansión, recupera resolución, reduce profundidad

| Componente | Función | Operación Matemática |
|---|---|---|
| **Convolución** | Detecta características locales | $Y = \sigma(W * X + b)$ |
| **Max-Pooling** | Reduce resolución | $Y_{i,j} = \max(X[2i:2i+2, 2j:2j+2])$ |
| **Convolución Transpuesta** | Aumenta resolución | $Y = \sigma(W^T * X + b)$ |
| **Skip Connection** | Preserva detalles finos | Concatenación: $[Y_{\text{decoder}}, X_{\text{encoder}}]$ |

### EJEMPLO NUMÉRICO: Rastreo de Dimensiones en U-Net Mini

Construyamos un U-Net pequeño y sigamos las dimensiones de un tensor a través de la red.

**Entrada:** Imagen 64×64 con 1 canal (imagen en escala de grises)
```
Forma: (64, 64, 1)
```

**Encoder - Lado izquierdo:**

```
BLOQUE 1:
  Input: (64, 64, 1)
  → Conv 3×3, 16 filtros, ReLU: (64, 64, 16)
  → Conv 3×3, 16 filtros, ReLU: (64, 64, 16)
  → MaxPool 2×2, stride 2: (32, 32, 16)

BLOQUE 2:
  Input: (32, 32, 16)
  → Conv 3×3, 32 filtros, ReLU: (32, 32, 32)
  → Conv 3×3, 32 filtros, ReLU: (32, 32, 32)
  → MaxPool 2×2, stride 2: (16, 16, 32)

BLOQUE 3:
  Input: (16, 16, 32)
  → Conv 3×3, 64 filtros, ReLU: (16, 16, 64)
  → Conv 3×3, 64 filtros, ReLU: (16, 16, 64)
  → MaxPool 2×2, stride 2: (8, 8, 64)
```

**Bottleneck - Centro de la U:**

```
BLOQUE CENTRAL:
  Input: (8, 8, 64)
  → Conv 3×3, 128 filtros, ReLU: (8, 8, 128)
  → Conv 3×3, 128 filtros, ReLU: (8, 8, 128)
```

**Decoder - Lado derecho:**

```
BLOQUE 4 (Expansión):
  Input: (8, 8, 128)
  → ConvTranspuesta 2×2, stride 2, 64 filtros: (16, 16, 64)
  → Concatenar con skip de BLOQUE 3: (16, 16, 64+64) = (16, 16, 128)
  → Conv 3×3, 64 filtros, ReLU: (16, 16, 64)

BLOQUE 5 (Expansión):
  Input: (16, 16, 64)
  → ConvTranspuesta 2×2, stride 2, 32 filtros: (32, 32, 32)
  → Concatenar con skip de BLOQUE 2: (32, 32, 32+32) = (32, 32, 64)
  → Conv 3×3, 32 filtros, ReLU: (32, 32, 32)

BLOQUE 6 (Expansión):
  Input: (32, 32, 32)
  → ConvTranspuesta 2×2, stride 2, 16 filtros: (64, 64, 16)
  → Concatenar con skip de BLOQUE 1: (64, 64, 16+16) = (64, 64, 32)
  → Conv 3×3, 16 filtros, ReLU: (64, 64, 16)

SALIDA:
  Input: (64, 64, 16)
  → Conv 1×1, 1 filtro (máscara de segmentación): (64, 64, 1)
```

**Resumen visual del flujo de dimensiones:**

```
(64,64,1) → (32,32,16) → (16,16,32) → (8,8,64) → (8,8,128)
                                           ↑
                                    Bottleneck
                                           ↓
           (64,64,1) ← (32,32,32) ← (16,16,64) ← (8,8,64)

Skip connections:
  (64,64,16) concatenados → (64,64,32)
  (32,32,32) concatenados → (32,32,64)
  (16,16,64) concatenados → (16,16,128)
```

## 5. Max-Pooling y Upsampling en Detalle

### Max-Pooling como Operador de Restricción

En teoría de multigrid, la **restricción** (restriction) es la transferencia de información de una cuadrícula fina a una más gruesa. Max-pooling juega este papel: resume una región 2×2 en un único valor.

La analogía:
- **Cuadrícula fina:** resolución original (256×256)
- **Cuadrícula gruesa:** resolución reducida (128×128)
- **Operador de restricción:** max-pooling

Esta es la razón por la que U-Net tiene forma de V con múltiples niveles: es implementar un **V-ciclo multigrid** como red neuronal.

#### EJEMPLO NUMÉRICO: Pooling y "Unpooling" con Indices

Realizaremos max-pooling 2×2 en una matriz 4×4, guardando los índices (crucial para SegNet):

**Entrada X:**
```
X = [[1,  3, 2, 4],
     [5,  7, 1, 2],
     [2,  4, 9, 1],
     [3,  8, 2, 6]]
```

**Max-Pooling 2×2 con índices:**

Cuadrante superior-izquierdo [1,3; 5,7]:
- Máximo: 7
- Índice: (1, 1) dentro del cuadrante = (1, 1) global

Cuadrante superior-derecho [2,4; 1,2]:
- Máximo: 4
- Índice: (0, 1) dentro del cuadrante = (0, 3) global

Cuadrante inferior-izquierdo [2,4; 3,8]:
- Máximo: 8
- Índice: (1, 1) dentro del cuadrante = (3, 1) global

Cuadrante inferior-derecho [9,1; 2,6]:
- Máximo: 9
- Índice: (0, 0) dentro del cuadrante = (2, 2) global

**Salida del pooling Y:**
```
Y = [[7, 4],
     [8, 9]]
```

**Mapa de índices M (para unpooling):**
```
M = [[(1,1), (0,3)],
     [(3,1), (2,2)]]
```

Ahora, si queremos hacer **unpooling** (reverso), usamos estos índices. Supongamos que queremos expandir:

```
Z = [[10, 20],
     [15, 30]]
```

Creamos una matriz 4×4 vacía (ceros) y colocamos los valores usando los índices:

```
Resultado = [[0,  0,  0, 20],
             [10, 0,  0,  0],
             [0,  0, 15,  0],
             [0, 30,  0,  0]]
```

Observe: el valor 10 va a posición (1,1), 20 a (0,3), 15 a (2,2), 30 a (3,1).

Esta técnica de index unpooling es exactamente lo que hace **SegNet**, reduciendo significativamente los parámetros comparado con U-Net estándar.

### Convolución Transpuesta para Upsampling

La **convolución transpuesta** (también llamada deconvolución, aunque el término es impreciso) es el "inverso" de la convolución.

#### EJEMPLO NUMÉRICO: Convolución Transpuesta de 2×2 a 4×4

Entrada (2×2):
```
X = [[1, 2],
     [3, 4]]
```

Kernel (2×2) de convolución transpuesta:
```
K = [[1, 2],
     [3, 4]]
```

La convolución transpuesta "desdobla" cada valor de entrada y lo multiplica por el kernel:

Para valor 1 en (0,0):
```
Contribución:
[[1*1, 1*2],
 [1*3, 1*4]]
= [[1, 2],
   [3, 4]]
```

Para valor 2 en (0,1):
```
Contribución (desplazada 1 píxel a la derecha):
[[2*1, 2*2],
 [2*3, 2*4]]
= [[2, 4],
   [6, 8]]
Desplazado: posición (0,1)
```

Para valor 3 en (1,0):
```
Contribución (desplazada 1 píxel abajo):
[[3*1, 3*2],
 [3*3, 3*4]]
= [[3, 6],
   [9, 12]]
Desplazado: posición (1,0)
```

Para valor 4 en (1,1):
```
Contribución (desplazada 1 píxel abajo y derecha):
[[4*1, 4*2],
 [4*3, 4*4]]
= [[4, 8],
   [12, 16]]
Desplazado: posición (1,1)
```

**Sumamos todas las contribuciones:**

```
Posición (0,0): 1
Posición (0,1): 2+2 = 4
Posición (0,2): 4
Posición (0,3): 8

Posición (1,0): 3+3 = 6
Posición (1,1): 4+2+6+4 = 16
Posición (1,2): 8+6 = 14
Posición (1,3): 8+12 = 20

Posición (2,0): 9
Posición (2,1): 6+12 = 18
Posición (2,2): 12+16 = 28
Posición (2,3): 16

Posición (3,0): 12
Posición (3,1): 9+16 = 25
Posición (3,2): 24
Posición (3,3): 16
```

**Salida (4×4):**
```
[[1,  4,  4,  8],
 [6, 16, 14, 20],
 [9, 18, 28, 16],
 [12, 25, 24, 16]]
```

Hemos expandido de 2×2 a 4×4, y el contenido de información se ha multiplicado (algunos píxeles vecinos comparten contribuciones, de ahí los valores duplicados).

### Interpolación Bilineal como Alternativa

La interpolación bilineal es un método más simple (pero menos entrenable) que la convolución transpuesta:

Para expandir un píxel en posición (x, y) a una cuadrícula más fina, interpolamos usando los 4 píxeles vecinos más cercanos:

$$Y[i,j] = (1-\alpha)(1-\beta) X[x,y] + \alpha(1-\beta) X[x+1,y] + (1-\alpha)\beta X[x,y+1] + \alpha\beta X[x+1,y+1]$$

donde $\alpha$ y $\beta$ son las fracciones de distancia.

| Método Upsampling | Mecanismo | Ventajas | Desventajas |
|---|---|---|---|
| **Convolución Transpuesta** | Kernel aprendible | Aprende upsampling óptimo | Más parámetros |
| **Interpolación Bilineal** | Pesos fijos lineales | Rápido, sin parámetros | No aprende, salida borrosa |
| **Interpolación Nearest Neighbor** | Repite píxeles | Muy rápido | Artefactos visuales |
| **Max-Unpooling (SegNet)** | Índices almacenados | Recupera estructura exacta | Requiere almacenamiento de índices |

## 6. Skip Connections: El Ingrediente Secreto

### ¿Por Qué el Decoder Solo Produce Resultados Borrosos?

Un decodificador sin skip connections tiene un problema fundamental:

El encoder comprime la información:
- 256×256 → 128×128 → 64×64 → 32×32 → 16×16

En cada paso, se pierde información (especialmente en los bordes y detalles finos). El max-pooling selecciona solo el máximo en cada región 2×2, descartando el 75% de los valores. Incluso si el bottleneck contiene toda la información relevante (dudoso), el decodificador no puede recuperar los detalles que fueron descartados.

El resultado: un mapa de segmentación borroso con bordes suavizados, que es inaceptable en segmentación médica donde los bordes precisos del tumor son críticos.

### Concatenación vs. Suma

En U-Net, usamos **concatenación**: el decoder ascendente se concatena (channel-wise) con el feature map del encoder en el mismo nivel.

```
Decoder input: (32, 32, 64) [de upsampling]
Skip connection: (32, 32, 32) [encoder]
Concatenated: (32, 32, 96) [64+32=96 canales]
```

Por qué concatenación y no suma:

- **Concatenación preserva toda la información:** ambas características están presentes, la red puede elegir qué usar
- **Suma pierde información:** si una característica es pequeña, es aplastada por la otra

### Interpretación Matemática: Corrección de Error de Interpolación

Podemos ver las skip connections como un **corrector de errores**:

Sea $U_i$ la salida del decodificador en el nivel $i$, e $I_i$ la entrada del encoder en ese nivel.

Sin skip connection: la salida es solo $U_i$.

Con skip connection: la salida es $\text{Conv}(\text{Concat}(U_i, I_i))$, que puede aprender a reconstruir los detalles que faltaban en $U_i$.

Matemáticamente:

$$Y_i = \text{Conv}(\text{Concat}(U_i, I_i)) \approx U_i + \underbrace{f(I_i)}_{\text{Error correction}}$$

donde $f$ es una función aprendida que restaura los detalles interpolados.

#### EJEMPLO NUMÉRICO: Skip Connection Restaura Detalles

Codificar un vector 1D como ejemplo:

**Vector original (4 elementos):**
```
X = [1, 5, 2, 8]
```

**Encoder (max-pool 2×2):**
```
Y_encoded = [max(1,5), max(2,8)] = [5, 8]
Información perdida: los valores 1 y 2 fueron descartados
```

**Decoder sin skip connection (upsampling simple):**
```
Y_upsample = [5, 5, 8, 8]  (repetir cada valor)
Diferencia del original: [4, 0, -6, 0]  ← Error significativo
```

**Con skip connection:**

Concatenamos el decoder upsample con el encoder original:
```
Concatenado = [[5, 1], [5, 5], [8, 2], [8, 8]]
(valor decoder, valor skip)
```

Una red neuronal puede aprender a corregir:

```
Y_corregido ≈ [1, 5, 2, 8]  ← Recuperación casi perfecta
```

El error de interpolación ha sido corregido por la información del encoder.

## 7. U-Net como Método Multigrid (V-Cycle)

### Sistemas Dinámicos y Redes Neuronales

Una perspectiva moderna ve las redes neuronales como **sistemas dinámicos**:

$$\frac{dx}{dt} = F(x(t), \theta(t))$$

donde:
- **x(t)** es el estado de la red en el tiempo $t$ (profundidad de la capa)
- **F** es el mapeo de una capa
- **θ(t)** son los parámetros aprendibles

Cada capa profundiza la representación: una entrada neutra se transforma gradualmente en una representación más abstracta.

En este marco, U-Net es un **V-ciclo neural** que:
1. Desciende (encoder): comprime información hacia una representación abstracta
2. Asciende (decoder): recupera información espacial y escala hacia la salida

| Componente Multigrid | U-Net Equivalente | Rol Matemático |
|---|---|---|
| **Suavizador (Smoother)** | Convoluciones iniciales | Reduce componentes de alta frecuencia |
| **Restricción I_h^{2h}** | Max-pooling | Transfiere información de cuadrícula fina a gruesa |
| **Prolongación I_{2h}^h** | Convolución transpuesta | Transfiere información de cuadrícula gruesa a fina |
| **Corrección de error** | Skip connections | Corrige errores de interpolación |

### Principio del Máximo de Pontryagin (Conexión Teórica)

El **Principio del Máximo de Pontryagin** es un resultado de control óptimo que caracteriza trayectorias óptimas en sistemas dinámicos:

$$\mathcal{H} = L(x, u) + \lambda^T F(x, u, \theta)$$

donde:
- **L** es el costo (pérdida)
- **F** es el sistema dinámico
- **λ** es el costate (multiplicador de Lagrange)

U-Net implícitamente resuelve un problema de control óptimo:
- El encoder define una trayectoria hacia una representación comprimida
- El decoder invierte esta trayectoria mientras minimiza una pérdida de segmentación
- Las skip connections garantizan que podamos volver al espacio original

Este es profundo: entrenar una U-Net es resolver un problema de control óptimo donde queremos que la segmentación tenga costo mínimo mientras preservamos información crucial.

### Por Qué Importa Esta Perspectiva

Entender U-Net como un V-ciclo multigrid y un sistema de control óptimo ayuda a:

1. **Diseñar nuevas arquitecturas:** sabemos qué principios importan
2. **Diagnosticar problemas de entrenamiento:** si la red no converge, es porque el "V-ciclo" no es eficiente
3. **Combinar con técnicas matemáticas clásicas:** multigrid, análisis armónico, etc.
4. **Entender convergencia:** la teoría de multigrid garantiza que V-ciclos convergen rápidamente

## 8. SegNet y Seg-UNet: Variantes Arquitectónicas

### SegNet: Almacenamiento Inteligente de Índices

SegNet (2015, Badrinarayanan et al.) introduce una idea brillante: en lugar de almacenar todos los feature maps del encoder, solo almacenamos los **índices de los máximos** del pooling.

**U-Net estándar:**
```
Memoria encoder: 256×256×64 + 128×128×128 + ... = MUCHA memoria

Ejemplo: 256×256×64 = 4.2 MB solo para la primera capa
```

**SegNet:**
```
Memoria encoder: 2×(256×256×2) índices + 128×128×2 índices + ...
                = 256×256×2 bytes + 128×128×2 bytes + ...
                = ~130 KB

Reducción: 4.2 MB → ~1 KB para la primera capa
```

Luego, en el decoder, usa estos índices para unpooling exacto (como el ejemplo anterior).

**Pero SegNet pierde los valores de los feature maps**, solo guarda sus ubicaciones. Esto es eficiente en memoria pero sacrifica algo de precisión.

### Seg-UNet: Lo Mejor de Ambos Mundos

Seg-UNet combina las dos ideas:
- Usa **skip connections** de U-Net (preserva información completa)
- Usa **index unpooling** de SegNet (más eficiente en memoria)

| Arquitectura | Memoria Encoder | Precisión Espacial | Riqueza Semántica | Complejidad |
|---|---|---|---|---|
| **U-Net** | Alta (todos los features) | Muy alta | Alta | Media |
| **SegNet** | Muy baja (solo índices) | Alta | Baja | Baja |
| **Seg-UNet** | Media (índices + algunos features) | Muy alta | Alta | Media-Alta |

### EJEMPLO NUMÉRICO: Unpooling con Índices Almacenados

Continuamos con el ejemplo anterior.

**Encoder max-pooling:**
```
X (4×4) = [[1, 3, 2, 4],
           [5, 7, 1, 2],
           [2, 4, 9, 1],
           [3, 8, 2, 6]]

Pooled Y (2×2) = [[7, 4],
                   [8, 9]]

Stored indices M:
  M[0,0] = (1,1)  ← el 7 vino de posición (1,1)
  M[0,1] = (0,3)  ← el 4 vino de posición (0,3)
  M[1,0] = (3,1)  ← el 8 vino de posición (3,1)
  M[1,1] = (2,2)  ← el 9 vino de posición (2,2)
```

**Decoder: Supongamos que el bottleneck produjo:**
```
Z (2×2) = [[10, 20],
           [15, 30]]
```

**Unpooling con índices:**
```
Result = zeros(4, 4)
Result[M[0,0]] = Z[0,0]  ⟹  Result[1,1] = 10
Result[M[0,1]] = Z[0,1]  ⟹  Result[0,3] = 20
Result[M[1,0]] = Z[1,0]  ⟹  Result[3,1] = 15
Result[M[1,1]] = Z[1,1]  ⟹  Result[2,2] = 30

Resultado final:
[[0,  0,  0, 20],
 [10, 0,  0,  0],
 [0,  0, 30,  0],
 [0, 15,  0,  0]]
```

Compare con upsampling lineal (que hubiera dado [10,10,20,20; 10,10,20,20; 15,15,30,30; 15,15,30,30]): el index unpooling es mucho más selectivo y exacto.

## 9. El Fallo de las Métricas Locales

### Dice Coefficient: Útil pero Ciego a Topología

El **coeficiente Dice** mide solapamiento:

$$\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$$

donde:
- **X** es la máscara predicha
- **Y** es la máscara verdadera (gold standard)
- **|·|** es el tamaño (número de píxeles)

Es una métrica excelente para *tamaño y solapamiento general*. Pero tiene fallos topológicos.

#### EJEMPLO NUMÉRICO 1: Anillo con Un Píxel Roto

**Máscara verdadera:** anillo de 10×10 píxeles (borde de 2 píxeles de espesor, centro hueco)

```
Verdadera:
██████████
██    ██
██    ██
██    ██
██    ██
██    ██
██    ██
██    ██
██████████

Total de píxeles True: 36
```

**Predicción:** El mismo anillo, pero falta UN píxel (rompe la topología)

```
Predicción:
██████████
░█    ██  ← píxel faltante
██    ██
██    ██
██    ██
██    ██
██    ██
██    ██
██████████

Total de píxeles Pred: 35
Píxeles correctos (True AND Pred): 35
```

**Cálculo de Dice:**
$$\text{Dice} = \frac{2 \times 35}{35 + 36} = \frac{70}{71} = 0.9859 \approx 98.59\%$$

**Pero topológicamente:** El anillo está **roto**. Es un error crítico. Un píxel faltante ha convertido un anillo (topología de un círculo) en un arco (sin cierre).

En segmentación de tumores: si el tumor está conectado (topología relevante) y la predicción lo rompe, Dice dice "98.59% correcto" cuando realmente es un fallo cualitativo.

#### EJEMPLO NUMÉRICO 2: Tumor Sólido con Un Agujero Falso

**Máscara verdadera:** tumor sólido 10×10

```
Verdadera:
██████████
██████████
██████████
██████████
██████████
██████████
██████████
██████████
██████████
██████████

Total: 100 píxeles
```

**Predicción:** Mismo tumor, pero con un agujero falso de 4 píxeles en el centro

```
Predicción:
██████████
██████████
██████████
███  ███
███  ███
██████████
██████████
██████████
██████████
██████████

Total Pred: 96 píxeles
Píxeles correctos: 96
```

**Cálculo de Dice:**
$$\text{Dice} = \frac{2 \times 96}{96 + 100} = \frac{192}{196} = 0.9795 \approx 97.95\%$$

Nuevamente, ~98% según Dice, pero hemos **creado una topología falsa** (un agujero que no existe). En seguimiento clínico, esto es un error serious: el tumor parece tener una cavidad cuando no la tiene.

### Alternativas: IoU, Cross-Entropy, Hausdorff

| Métrica | Fórmula | Qué Mide | Qué Pierde |
|---|---|---|---|
| **Dice** | $\frac{2\|X \cap Y\|}{\|X\|+\|Y\|}$ | Solapamiento global | Topología, estructura fina |
| **IoU** | $\frac{\|X \cap Y\|}{\|X \cup Y\|}$ | Intersección sobre unión | Topología |
| **Cross-Entropy** | $-\sum_i y_i \log(\hat{y}_i)$ | Distribución probabilística | Errores grandes localmente |
| **Hausdorff** | $\max(d(X,Y), d(Y,X))$ | Distancia máxima entre bordes | Lentitud computacional |

Todas capturan algo diferente, pero solo **Hausdorff** y métricas topológicas capturan errores topológicos.

### Por Qué Se Necesitan Pérdidas Topológicas

El próximo post (TDA-SegUNet) trata sobre pérdidas topológicas que corrigen estos problemas:

- **TopoLoss:** penaliza cambios en números de Betti
- **Betti Matching Loss:** hace coincidir explícitamente la topología predicha con la verdadera
- **clDice:** Dice consciente de componentes conectadas

Estas pérdidas son computacionalmente más caras pero capturan lo que realmente importa para aplicaciones médicas.

## 10. Funciones de Pérdida Topológicas (Vista Previa)

### Visión General de Pérdidas Topológicas

Las pérdidas topológicas penalizan **cambios en la forma conectada** de la máscara predicha.

#### TABLA: Función de Pérdida Topológica

| Función | Mecanismo | Fortaleza | Limitación |
|---|---|---|---|
| **TopoLoss** | Penaliza cambios en número de componentes conectadas | Previene rotura de anillos | Computacionalmente costosa (TDA en cada iteración) |
| **Betti Matching** | Hace coincidir números de Betti (ciclos, hoyos) | Captura topología completa | Requiere calcular homología (lento) |
| **clDice** | Dice sobre componentes conectadas | Equilibra precisión global y topología | Menos bien estudiada |

### Ejemplo: TopoLoss vs Dice

Volvamos al anillo roto:

**Con Dice solo:**
- Pierde un píxel, Dice = 98.59%
- Red piensa que está bien, sigue adelante
- Resultado: anillo roto

**Con TopoLoss:**
- Número de Betti verdadero: 1 (un agujero)
- Número de Betti predicho: 0 (sin agujero, es un arco)
- TopoLoss penaliza esta diferencia fuertemente
- Red es forzada a aprender a cerrar el anillo

### Entrenamiento por Currículum: Bootstrapping

Hay un problema práctico: si usas solo TopoLoss desde el inicio, la red no aprende nada (el gradiente es ruidoso porque la topología es discreta).

**Solución: Entrenamiento por currículum**

```
Etapa 1 (épocas 1-50):
  Loss = Dice (aprende la forma básica)

Etapa 2 (épocas 51-100):
  Loss = 0.8*Dice + 0.2*TopoLoss (introduce topología gradualmente)

Etapa 3 (épocas 101-150):
  Loss = 0.5*Dice + 0.5*TopoLoss (peso igual)

Etapa 4 (épocas 151+):
  Loss = 0.2*Dice + 0.8*TopoLoss (énfasis en topología)
```

Este esquema permiteque la red primero aprenda la segmentación básica (con Dice), y luego refine la topología.

## 11. Implementación Práctica

### U-Net Mínima en PyTorch (~50 líneas)

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._conv_block(128, 64)  # 128 porque concatenamos

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._conv_block(64, 32)   # 64 porque concatenamos

        # Output
        self.final = nn.Conv2d(32, out_channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)      # (B, 32, H, W)
        p1 = self.pool1(e1)    # (B, 32, H/2, W/2)

        e2 = self.enc2(p1)     # (B, 64, H/2, W/2)
        p2 = self.pool2(e2)    # (B, 64, H/4, W/4)

        # Bottleneck
        b = self.bottleneck(p2) # (B, 128, H/4, W/4)

        # Decoder con skip connections
        u2 = self.upconv2(b)   # (B, 64, H/2, W/2)
        u2 = torch.cat([u2, e2], dim=1)  # Concatenar skip
        d2 = self.dec2(u2)     # (B, 64, H/2, W/2)

        u1 = self.upconv1(d2)  # (B, 32, H, W)
        u1 = torch.cat([u1, e1], dim=1)  # Concatenar skip
        d1 = self.dec1(u1)     # (B, 32, H, W)

        # Output
        return self.final(d1)  # (B, 1, H, W)
```

### Función Dice Loss en PyTorch

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred y target son (B, C, H, W)
        pred = torch.sigmoid(pred)  # Convertir a [0,1]

        # Aplanar
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        # Dice = 2*intersection / union
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice  # Retornar pérdida (queremos minimizar)
```

### Loop de Entrenamiento Simple

```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dice_loss = DiceLoss()

# Entrenar
for epoch in range(100):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward
        outputs = model(images)
        loss = dice_loss(outputs, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### Rastreo de Dimensiones de Tensores

Para debugging, es útil imprimir formas de tensores:

```python
def forward_debug(self, x):
    print(f"Input: {x.shape}")

    e1 = self.enc1(x)
    print(f"After enc1: {e1.shape}")  # (B, 32, H, W)

    p1 = self.pool1(e1)
    print(f"After pool1: {p1.shape}")  # (B, 32, H/2, W/2)

    e2 = self.enc2(p1)
    print(f"After enc2: {e2.shape}")   # (B, 64, H/2, W/2)

    p2 = self.pool2(e2)
    print(f"After pool2: {p2.shape}")  # (B, 64, H/4, W/4)

    b = self.bottleneck(p2)
    print(f"Bottleneck: {b.shape}")    # (B, 128, H/4, W/4)

    u2 = self.upconv2(b)
    print(f"After upconv2: {u2.shape}") # (B, 64, H/2, W/2)

    u2 = torch.cat([u2, e2], dim=1)
    print(f"After cat (skip): {u2.shape}") # (B, 128, H/2, W/2)

    # ... rest of forward pass
```

## 12. Resumen y Conexiones

### De lo Básico a la Vanguardia

Hemos recorrido un camino largo:

1. **Perceptrón:** una neurona, una ecuación simple
2. **CNNs:** convoluciones, pooling, capas profundas
3. **Segmentación:** pixel-wise, requiere preservar estructura
4. **U-Net:** encoder-decoder con skip connections
5. **Variantes:** SegNet, Seg-UNet, optimizaciones
6. **Pérdidas:** desde Dice (géométrico) a topológicas (structural)

### Tabla de Conexiones: Componente U-Net → Fundamento Matemático → Rol en Tesis

| Componente | Fundamento Matemático | Rol en Thesis |
|---|---|---|
| **Convolución** | Operador lineal con kernel finito | Extrae características visuales (tumores) |
| **ReLU** | Activación no-lineal (max(0,x)) | Aprende decisiones no-lineales |
| **Max-Pooling** | Restricción en multigrid (I_h^{2h}) | Compresión, cálculo eficiente |
| **Bottleneck** | Representación comprimida | Cuello de botella, fuerza la abstracción |
| **Convolución Transpuesta** | Prolongación en multigrid (I_{2h}^h) | Recupera resolución original |
| **Skip Connection** | Corrección de error de interpolación | **Crítico:** preserva detalles finos del tumor |
| **Dice Loss** | Medida F₁ (solapamiento) | Métrica de entrenamiento estándar |
| **TopoLoss** | Homología persistente (números de Betti) | **Próximo:** fuerza topología correcta |

### Puente a la Próxima Entrega: TDA-SegUNet

El siguiente artículo introducirá:

- **Análisis Topológico de Datos (TDA):** homología persistente, números de Betti
- **TDA-SegUNet:** combinación de U-Net con pérdidas topológicas
- **Resultados cuantitativos:** cómo TDA mejora la detección de tumores
- **Ejemplos reales:** fallos de Dice que TopoLoss corrige

En segmentación de tumores cerebrales, la topología importa porque:

- Un tumor conectado debe permanecer conectado en la predicción
- Los bordes deben ser lisos (no rotos)
- Las cavidades internas no deben ser creadas falsamente

La combinación de geometría (CNN/U-Net) y topología (TDA) es lo que hace que nuestro enfoque TDA-SegUNet sea superior a métodos estándar.

---

## Referencias y Lecturas Adicionales

- Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI.
- Badrinarayanan, V., Kendall, A., & Cipolla, R. (2017). "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." IEEE TPAMI.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep Learning." Nature.
- Munkres, J. (2018). "Elements of Algebraic Topology." Revised Edition.
- Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). "A Multigrid Tutorial." SIAM.
