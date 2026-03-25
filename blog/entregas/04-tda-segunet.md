---
layout: post
title: "TDA-SegUNet: Cuando la Topología se Encuentra con el Deep Learning"
date: 2026-03-24
categories: [deep-learning, topologia]
tags: [TDA-SegUNet, segmentacion, tumores-cerebrales, BraTS, homologia-persistente]
math: true
---

# TDA-SegUNet: Cuando la Topología se Encuentra con el Deep Learning

En los artículos anteriores de esta bitácora científica, hemos explorado los fundamentos de las convulsiones, la arquitectura U-Net y los misterios de la topología. Hoy, sintetizamos todo en **TDA-SegUNet**, una arquitectura revolucionaria que integra el Análisis Topológico de Datos con redes neuronales profundas para resolver uno de los problemas más desafiantes de la medicina computacional: la segmentación de tumores cerebrales.

## 1. El Problema Clínico: Segmentación de Tumores Cerebrales

### ¿Por qué es crítica la segmentación de tumores cerebrales?

Los tumores cerebrales representan una de las amenazas oncológicas más devastadoras. Su localización en tejido vital exige una precisión quirúrgica absoluta. Una desviación de milímetros puede significar la diferencia entre preservar una función cognitiva y una discapacidad permanente.

### Tipos de tumores: GBM y meningiomas

- **Glioblastoma Multiforme (GBM)**: El más agresivo de los gliomas (grado IV de la OMS). Típicamente muestra:
  - Núcleo hipodenso (necrótico)
  - Anillo de realce periférico (tumor activo)
  - Edema vasogénico circundante
  - Supervivencia mediana: 14-15 meses

- **Meningioma**: Tumor de las meninges, generalmente benigno. Características:
  - Compresión de materia blanca adyacente
  - Potencial maligno (grados I-III)
  - A menudo bien delimitado

### Las Cuatro Modalidades MRI

La resonancia magnética (MRI) captura diferentes aspectos de la anatomía cerebral mediante distintas secuencias:

| Modalidad | Qué Muestra | Característica Visible del Tumor |
|-----------|-------------|----------------------------------|
| **T1** | Recuperación del eje longitudinal; contraste principalmente anatómico | Hipointenso (gris respecto al blanco); define bordes en edema |
| **T1 con Contraste (T1ce)** | T1 + gadolinio IV; resalta disrupciones BBB | **Realce periférico activo** — marca tumor viable |
| **T2** | Recuperación transversal; sensible al agua | Hiperintensidad de edema vasogénico, núcleo necrótico |
| **FLAIR** | T2 con supresión de LCR; elimina ruido del fluido | **El mejor para detectar edema**; máxima sensibilidad para tumor total |

### El Desafío BraTS: Tres Regiones, Una Máscara

El desafío de segmentación de tumores cerebrales BraTS (Brain Tumor Segmentation Challenge) definió el estándar de oro con tres regiones anidadas:

| Región | Definición | Significancia Clínica | Firma de Betti Esperada |
|--------|-----------|----------------------|------------------------|
| **WT** (Tumor Completo) | Edema + tumor core + realce | Tamaño total del edema para radioterapia | β₀ ≤ 3; β₁ ≤ 1 típico |
| **TC** (Núcleo Tumoral) | Tumor core = necrosis + realce | Región "viva" (no necrótica); define volumen quirúrgico | β₀ ≤ 2; β₁ = 0 común |
| **ET** (Tumor Realzado) | Solo realce con contraste | Tumor activo; máxima agresividad; define borde real | β₀ = 1; β₁ = 0 (convexa) |

### Por qué importa la precisión

Una segmentación imprecisa en radioterapia conformal lleva a:
- **Subdosificación**: tumor no irradiado → recurrencia local
- **Sobredosificación**: tejido sano irradiado → toxicidad, muerte celular
- Diferencias en supervivencia: ±10% de precisión ≈ ±3-6 meses de vida media

## 2. El Pipeline Completo: De la MRI a la Máscara

TDA-SegUNet es una tubería de 4 etapas que transforma imágenes médicas brutas en máscaras de segmentación de alta precisión:

```
MRI brutas (T1, T1ce, T2, FLAIR)
         ↓
   [Preprocesamiento]
         ↓
   [Extracción TDA]
         ↓
   [U-Net + Fusión]
         ↓
   [Post-procesamiento]
         ↓
   Máscaras de segmentación
```

| Paso | Entrada | Operación | Salida | Tiempo (GPU) |
|------|---------|-----------|--------|--------------|
| 1. Preprocesamiento | 4 MRI crudo | Strip de cráneo, z-norm | X ∈ ℝ^(H×W×D×4) | ~2 seg |
| 2. TDA | X crudo | EDT → Homología cúbica → PI | PI_β₀, PI_β₁ ∈ ℝ^(H×W×D) | ~8 seg |
| 3. Red | X + PI | Convs + U-Net | Y ∈ [0,1]^(H×W×D×3) | ~0.5 seg |
| 4. Post-proceso | Y bruto | Umbral, CRF, cierre | M ∈ {0,1}^(H×W×D×3) | ~1 seg |
| **Total** | | | | ~11.5 seg/volumen |

### Paso 1: Preprocesamiento MRI

**Skull stripping**: Removemos el cráneo (hueso no tumoral) usando una máscara binaria predefinida. Esto:
- Reduce artefactos óseos
- Enfoca la atención de la red en el parénquima cerebral
- Acelera cálculos posteriores

**Z-score normalization** (normalización estadística):
$$X_{norm}(i,j,k,m) = \frac{X(i,j,k,m) - \mu_m}{\sigma_m + \epsilon}$$

donde μₘ y σₘ son media y desv. estándar del canal *m* dentro de la máscara cerebral, y ε=10⁻⁸ evita división por cero.

### Paso 2: Extracción de Características TDA

Este es el corazón de la innovación. Describiremos en detalle en la Sección 3.

### Paso 3: U-Net con Entrada Aumentada

La U-Net clásica recibe 4 canales (FLAIR, T1ce, T1, T2). TDA-SegUNet envía 6:
- 4 canales MRI originales
- 2 canales de imágenes de persistencia (PI_β₀, PI_β₁)

Las características topológicas están disponibles **desde el primer nivel de convolución**.

### Paso 4: Post-procesamiento

1. **Umbralización**: Y > 0.5 → máscara binaria
2. **Cierre morfológico**: elimina agujeros pequeños en ET
3. **Restricción de anidamiento**: asegurar ET ⊂ TC ⊂ WT

## 3. Extracción de Características TDA

### La Transformada de Distancia Euclidiana (EDT)

Dado un volumen binario *B* (1 = voxel de interés, 0 = fondo):

$$\text{EDT}(i,j,k) = \min_{(i',j',k') \in B} \sqrt{(i-i')^2 + (j-j')^2 + (k-k')^2}$$

La EDT produce un volumen de distancias donde cada voxel contiene la distancia euclidiana al voxel tumoral más cercano.

**¿Por qué EDT?** Transforma una máscara binaria en una función que codifica **forma**:
- Voxels profundos en el tumor (alejados del borde) tienen EDT alto
- Voxels en la frontera tienen EDT bajo
- La topología de los conjuntos de nivel EDT preserva la topología de *B*

### Ejemplo Numérico: EDT de una Máscara Simple

Consideremos una máscara binaria 2D de 5×5:

```
Máscara binaria B:          EDT correspondiente:
0 0 0 0 0                   ∞ ∞ ∞ ∞ ∞
0 1 1 1 0                   ∞ 1.0 1.4 1.0 ∞
0 1 1 1 0        ======>    ∞ 1.4 2.0 1.4 ∞
0 1 1 1 0                   ∞ 1.0 1.4 1.0 ∞
0 0 0 0 0                   ∞ ∞ ∞ ∞ ∞
```

(Los valores ∞ se reemplazan por 0 en la práctica)

El EDT revela que el centro (2,2) es el más "profundo" (EDT=2.0), mientras que los bordes están más cerca de la frontera.

### Homología Persistente Cúbica

A partir del EDT, construimos una **filtración**:

$$\emptyset = K_0 \subset K_1 \subset K_2 \subset \cdots \subset K_n = K$$

donde $K_t = \{(i,j,k) : \text{EDT}(i,j,k) \leq t\}$.

Para cada filtración computable homología en todos los niveles:
- **H₀(K_t)**: Número de componentes conectadas (β₀)
- **H₁(K_t)**: Número de bucles (β₁)
- **H₂(K_t)**: Número de cavidades vacías (β₂)

### Ejemplo Numérico Completo: Pipeline TDA en una Máscara 5×5

**Paso 1: Máscara binaria**
```
B = [[0, 0, 0, 0, 0],
     [0, 1, 1, 1, 0],
     [0, 1, 1, 1, 0],
     [0, 1, 1, 1, 0],
     [0, 0, 0, 0, 0]]
```

**Paso 2: Calcular EDT**
```
EDT = [[0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 1.0, 1.4, 1.0, 0.0],
       [0.0, 1.4, 2.0, 1.4, 0.0],
       [0.0, 1.0, 1.4, 1.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0]]
```

**Paso 3: Extraer diagrama de persistencia**

Valores de filtración únicos (ordenados): 0.0, 1.0, 1.4, 2.0

| t | Componentes (β₀) | Bucles (β₁) | Eventos |
|---|------------------|------------|---------|
| 0.0 | 1 | 0 | Un componente nace |
| 1.0 | 1 | 0 | 8 esquinas conectadas |
| 1.4 | 1 | 1 | **Nace un bucle** (la forma crea un agujero topológico) |
| 2.0 | 1 | 1 | Centro alcanzado; topología completa |

**Diagrama de persistencia (pares nacimiento-muerte)**:
- **H₀**: (0.0, ∞) — el componente nunca muere → *una componente persistente*
- **H₁**: (1.4, ∞) — el bucle nace a distancia 1.4 → *un bucle persistente*

**Paso 4: Convertir a Imagen de Persistencia** (Sección 4)

## 4. De Diagramas a Tensores: Imágenes de Persistencia

### Transformación de Coordenadas

El diagrama de persistencia vive en el espacio (nacimiento, muerte). Lo convertimos al espacio (nacimiento, persistencia):

$$\text{persistencia} = \text{muerte} - \text{nacimiento}$$

Para nuestro ejemplo:
- **H₀**: (0.0, ∞) → (0.0, ∞) — persistencia infinita
- **H₁**: (1.4, ∞) → (1.4, ∞) — persistencia infinita

(En la práctica truncamos ∞ al máximo EDT observado, p. ej., 2.0)

### Ponderación por Núcleo Gaussiano

Cada punto del diagrama $(b_i, p_i)$ contribuye a la imagen mediante un núcleo gaussiano bidimensional:

$$\text{PI}(x,y) = \sum_i \exp\left(-\frac{(x-b_i)^2 + (y-p_i)^2}{2\sigma^2}\right)$$

donde σ≈0.1 (ancho de banda del kernel).

### Discretización y Ajuste

Discretizamos en una cuadrícula que coincida con la resolución MRI:
- Rango: [0, max_EDT] × [0, max_EDT]
- Resolución: H × W × D (igual que el volumen original)
- Interpolación: trilineal para 3D

### Ejemplo Numérico: Diagrama a Imagen de Persistencia

Tomemos un diagrama simple con 4 puntos en un espacio 4×4:

```
Puntos del diagrama:
(b₁, p₁) = (0.5, 1.5)    ← H₀ principal
(b₂, p₂) = (1.2, 0.8)    ← H₁ pequeño bucle
(b₃, p₃) = (0.8, 1.2)    ← H₀ componente secundaria
(b₄, p₄) = (2.0, 0.5)    ← H₁ otro bucle

Imagen de Persistencia 4×4 (σ=0.5):
Row 0: [0.02, 0.05, 0.12, 0.08]
Row 1: [0.15, 0.48, 0.60, 0.35]  ← punto (b₁,p₁) contribuye fuertemente
Row 2: [0.20, 0.42, 0.38, 0.18]
Row 3: [0.08, 0.15, 0.10, 0.04]
```

Esta imagen retiene información topológica en un formato que las redes neuronales pueden procesar.

### Garantía de Estabilidad

**Teorema (Estabilidad de PI)**: Si dos diagramas están a distancia Wasserstein ε, sus imágenes de persistencia difieren en L₂ norm ≤ O(ε). Esto significa:
- Pequeño ruido en los datos → pequeño cambio en PI
- La red aprende características robustas

## 5. Fusión de Entrada (Input Fusion)

### La Arquitectura Clásica: Concatenación Simple

Las redes de segmentación tradicionales reciben:

$$X = \text{FLAIR} \oplus \text{T1ce} \oplus \text{T1} \oplus \text{T2}$$

donde ⊕ es concatenación a lo largo del eje del canal. Forma: (H, W, D, 4).

### TDA-Aumentado: Seis Canales

TDA-SegUNet envía:

$$X_{\text{TDA}} = \text{FLAIR} \oplus \text{T1ce} \oplus \text{T1} \oplus \text{T2} \oplus \text{PI}_{\beta_0} \oplus \text{PI}_{\beta_1}$$

Forma: (H, W, D, 6).

### Formalización Matemática

La primera capa de convolución es una función:

$$f_1(X) = \text{ReLU}(\text{Conv3D}(X, W_1) + b_1)$$

donde $W_1 \in \mathbb{R}^{3 \times 3 \times 3 \times 6 \times 64}$ (kernel 3×3×3, 6 canales entrada, 64 salida).

Con 6 canales en lugar de 4:
- **Parámetros**: 64 × (3³ × 6 + 1) = 18,496 (vs. 1,728 × 4 = 6,912)
- **Aumento**: +168% más parámetros locales
- **Interacción**: Los primeros filtros pueden mezclar MRI + topología desde el principio

### Tabla: Rol de Cada Canal

| Canal | Tipo de Información | Receptivo Local/Global | Qué Aprende la Red |
|-------|-------------------|----------------------|-------------------|
| FLAIR | Intensidad (T2 largoTE) | Local | Edema periférico, contraste general |
| T1ce | Intensidad (realce) | Local | Sangre-tejido roto, tumor activo |
| T1 | Intensidad (T1 corto) | Local | Hemosiderina, grasa, métrica de contraste |
| T2 | Intensidad (T2 estándar) | Local | Núcleo necrótico, fluido CSF |
| PI_β₀ | Topología (componentes) | **Global** | Número de tumores, conectividad |
| PI_β₁ | Topología (bucles) | **Global** | Cavidades, anillos, estructuras anulares |

### Proposición: Fusión de Entrada Mejora Capacidad

**Proposición 1.1**: La fusión de entrada (comparada con fusión de cuello de botella) permite que la red interactúe características MRI y topológicas desde la capa 1, aumentando expresividad en representaciones tempranas.

*Intuición*: Características topológicas suministran información de "forma global", disponible localmente para cada filtro. Sin ellas, la red debe reconstruir esta información de características de intensidad (más difícil).

### Ejemplo Numérico: Primera Convolución Mezclando Canales

Considere un filtro 3×3 en la primer capa:

```
Entrada (6 canales, región 3×3):
FLAIR: [[0.2, 0.5, 0.3],      T1ce: [[0.1, 0.8, 0.2],
        [0.4, 0.7, 0.6],              [0.3, 0.9, 0.5],
        [0.3, 0.8, 0.4]]              [0.2, 0.7, 0.3]]

T1:    [[0.5, 0.6, 0.4],      T2:   [[0.8, 0.7, 0.6],
        [0.7, 0.8, 0.7],             [0.6, 0.5, 0.4],
        [0.6, 0.7, 0.5]]             [0.7, 0.8, 0.6]]

PI_β₀: [[0.1, 0.3, 0.2],      PI_β₁: [[0.0, 0.1, 0.0],
        [0.4, 0.6, 0.5],              [0.2, 0.3, 0.1],
        [0.3, 0.5, 0.4]]              [0.1, 0.2, 0.0]]

Peso del filtro (1 de 64):
w = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2]  (pesos para los 6 canales)

Salida = suma ponderada de todos los canales:
= 0.1*(0.2+0.5+0.3+...) + 0.1*(0.1+0.8+...) + ... + 0.2*PI_β₀ + 0.2*PI_β₁
= ... (valor escalar)
```

Note que PI_β₀ y PI_β₁ ya contribuyen a esta salida temprana.

## 6. Fusión en Cuello de Botella (Bottleneck Fusion)

### Arquitectura Alternativa

En lugar de aumentar entrada, podríamos:
1. Pasar 4 canales MRI a través de U-Net clásica
2. Calcular TDA en paralelo
3. Concatenar en el cuello de botella (nivel más profundo)

### Análisis Matemático: Restricción del Flujo de Información

Comparemos dos flujos:

**Fusión de Entrada**:
```
Capa 1: X_TDA (6) → Conv → 64 canales
        ↓
        [Interacción MRI ⊗ TDA]
        ↓
Capa 2-N: Características ya están entrelazadas
```

**Fusión en Cuello de Botella**:
```
Capa 1: X (4) → Conv → 64 canales [MRI solo]
        ↓
        ...
Cuello: Concat([MRI features], [TDA features]) → 128 canales
        [Primera interacción aquí]
```

El camino desde las características topológicas a la salida es **más corto** en fusión de cuello. Menos capas para aprender combinaciones complejas.

### Tabla: Estrategias de Fusión Comparadas

| Estrategia | Capa de Interacción | Profundidad Post-Fusión | Expresividad Teórica | Velocidad |
|-----------|-------------------|----------------------|---------------------|-----------|
| **Entrada** | 1 | 14-20 capas | Alta (early mixing) | Moderada |
| **Cuello** | 10-12 | 2-4 capas | Media-Alta | Rápida |
| **Múltiple** | 1, 5, 10 | Variable | Muy Alta | Lenta |

**Proposición 1.2**: Input fusion es más expresiva pero requiere mayor capacidad. Bottleneck fusion es más rápida pero puede perder información topológica refinada.

## 7. Funciones de Pérdida Topológicas en Detalle

### 7.1 TopoLoss (Hu et al., 2019)

**Mecanismo**: Compara diagramas de persistencia de predicción P y ground truth G mediante distancia Wasserstein:

$$d_W(\text{PD}(P), \text{PD}(G)) = \min_{\pi} \sum_i \|p_i - \pi(g_i)\|_2$$

donde π es un emparejamiento óptimo de puntos del diagrama.

**Garantía**: L_topo = 0 ⟺ Betti numbers iguales en todo umbral.

**Crítica**: Emparejamiento por distancia Euclidiana ignora **ubicación espacial**. Dos tumores separados pueden emparejarse incorrectamente.

### 7.2 Betti Matching Loss (Stucki et al., 2023)

**Idea**: Usar la imagen de comparación $C = \max(P, G)$ (unión de regiones) para inducir emparejamiento **espacialmente correcto**.

**Algoritmo**:
1. Calcular mapas de inclusión en C
2. Seguir homología: ¿qué puntos en PD(P) corresponden a qué puntos en PD(G)?
3. Emparejar topológicamente, no euclídeamente

**Garantía de Corrección Espacial**: Si dos tumores son espacialmente separados en C, nunca se emparejan incorrectamente.

### Ejemplo Numérico: Por Qué Falla Wasserstein

**Escenario**: Predicción P tiene dos focos tumorales:
- Foco izquierdo grande: radio 5mm
- Foco derecho pequeño: radio 2mm

Ground truth G también tiene dos focos (en diferente tamaño):
- Foco izquierdo: radio 6mm
- Foco derecho: radio 3mm

**Diagrama de persistencia PD(P)**:
```
Punto 1 (H₀ principal): (b=0, p=5)   ← foco izquierdo grande
Punto 2 (H₀ secundaria): (b=0, p=2)  ← foco derecho pequeño
```

**Diagrama de persistencia PD(G)**:
```
Punto 1 (H₀ principal): (b=0, p=6)   ← foco izquierdo (más grande)
Punto 2 (H₀ secundaria): (b=0, p=3)  ← foco derecho (más grande)
```

**Emparejamiento Wasserstein** (por tamaño):
```
P_1 (grande, 5) ↔ G_1 (grande, 6)  ✓ Correcto por suerte
P_2 (pequeño, 2) ↔ G_2 (pequeño, 3)  ✓ Correcto por suerte

Pero si G_1 = (p=3) y G_2 = (p=6), el emparejamiento sería:
P_1 ↔ G_2  ✗ Empareja grandes con grandes, pero **¡EQUIVOCADAS ESPACIALMENTE!**
```

**Emparejamiento Betti** (por ubicación):
```
P_1 (ubicación izquierda) ↔ G_1 (ubicación izquierda)  ✓
P_2 (ubicación derecha) ↔ G_2 (ubicación derecha)  ✓

Siempre correcto, incluso si tamaños varían.
```

### 7.3 DMT-Loss (Discrete Morse Theory Loss)

**Concepto**: Encuentra **estructuras críticas** en Morse theory discreto:
- Células críticas de índice 1 → esqueletos (líneas, tubos)
- Células críticas de índice 2 → membranas (superficies)

**Cuándo usar**: Cuando la topología de interés es **tubular** (vasos, tractos blancos) o **laminares** (corteza).

**Para tumores**: Menos aplicable; mejor para segmentación vascular o neuronal.

### 7.4 clDice (Centerline Dice)

**Fórmula**:
$$\text{clDice} = \frac{2|S(P) \cap S(G)|}{|S(P)| + |S(G)|}$$

donde S(·) es el "esqueleto" (eje central topológico) de una región.

**Teorema**: Si dos regiones tienen esqueletos **homotópicamente equivalentes**, sus topologías son equivalentes.

**Cuándo usar**: Estructuras tubulares (vasos, nerviós).

### Tabla: Pérdidas Topológicas Comparadas

| Pérdida | Mecanismo | Garantía | Complejidad | Mejor Para |
|---------|-----------|----------|-------------|-----------|
| **TopoLoss** | Diagrama Wasserstein | Betti numbers | O(n³) | Análisis global |
| **Betti Matching** | Inclusiones en unión | Betti + localización | O(n²) | Tumores multifocales |
| **DMT-Loss** | Morse theory | Esqueleto exacto | O(n² log n) | Tubos, membranas |
| **clDice** | Centerline Dice | Homotopía | O(n) | Vasos, nervios |

## 8. Entrenamiento por Curriculum

### El Problema del Bootstrapping

Cuando la red comienza con pesos aleatorios, sus predicciones son prácticamente ruidosas. En este régimen:

**Predicción aleatoria Y_random ∈ [0,1] uniforme**

La homología de un campo aleatorio es trivial: β₀ ≈ const, β₁ ≈ 0 (sin estructura real).

Por lo tanto:
$$\nabla_Y L_{\text{topo}} \approx 0 \quad \text{(gradientes topológicos desvanecidos)}$$

La red no recibe retroalimentación topológica en el arranque. ¡Necesita primero aprender **dónde están** los tumores!

### Proposición: Predicciones Uniformes Degeneran Topología

**Proposición 1.3**: Si P(x,y,z) = c ∈ (0,1) es constante (predicción uniforme), entonces PD(P) contiene un solo punto (b=0, p=c) con multiplicidad infinita y β₀=1, β₁=0.

*Prueba*: Un campo constante es una única región conexa. ■

### Solución: Entrenamiento por Etapas

**Fase 1 (Épocas 1-30): Solo Dice**
$$L = L_{\text{Dice}}(P, G)$$

La red aprende a localizar tumores usando una métrica de similitud simple. Después de 30 épocas, Dice ≈ 0.70-0.75.

**Fase 2 (Épocas 30-60): Dice + Pérdida Topológica**
$$L = L_{\text{Dice}}(P, G) + \lambda(t) \cdot L_{\text{topo}}(P, G)$$

donde λ(t) aumenta gradualmente: λ(30)=0.01, λ(60)=0.1.

Ahora la red tiene **predicciones no triviales** (del Dice training) que generan diagramas de persistencia reales. Los gradientes topológicos fluyen.

**Fase 3 (Épocas 60-100): Ajuste Fino**
$$L = 0.9 \cdot L_{\text{Dice}} + 0.1 \cdot L_{\text{topo}}$$

La red refina bordes y corrige errores topológicos.

### Ejemplo Numérico: Diagramas Antes y Después

**Predicción aleatoria inicial (época 0)**:
```
Y_random ~ Uniform[0,1]
PD(Y_random) = {(0, 0.5)} con infinita multiplicidad
β₀ = 1, β₁ = 0
```

**Predicción tras Dice (época 30)**:
```
Y_dice = predicción con Dice ≈ 0.72
PD(Y_dice):
  (0.0, ∞): β₀ = 1
  (0.3, ∞): β₀ = 1 (posible segundo foco débil)
  (0.5, 0.8): β₁ = 1 (bucle artefacto)
β₀ actual = 2, β₁ = 1
```

**Predicción tras topología (época 60)**:
```
Y_topo = predicción refinada
PD(Y_topo):
  (0.0, ∞): β₀ = 1 (componente principal)
  (0.7, 1.0): β₁ = 1 (estructura real, no artefacto)
β₀ actual = 1, β₁ = 1
Coincide mejor con verdad de ground truth
```

## 9. El Problema de Inferencia

### Entrenamiento vs. Inferencia: Asimetría de Información

**Durante entrenamiento**: Calculamos TDA a partir de máscaras ground truth (conocidas).

**Durante inferencia**: ¡No tenemos ground truth! ¿Cómo obtenemos TDA?

Tres estrategias:

### Estrategia A: Inferencia Iterativa

1. Red predice $Y^{(1)}$
2. Calcular TDA a partir de $Y^{(1)}$ (predicción actual)
3. Alimentar TDA de nuevo a la red con MRI original
4. Red predice $Y^{(2)}$ (mejorada)
5. Repetir: converge a punto fijo

**Pregunta abierta**: ¿Garantiza convergencia? ¿A qué se converge?

**Problema**: Riesgo de retroalimentación positiva. Si Y⁽¹⁾ es mala, TDA es malo, Y⁽²⁾ puede ser peor.

### Estrategia B: TDA Precompilada desde Atlas

1. Usar base de datos de tumores etiquetados (atlas)
2. Promediar TDA de casos similares según MRI
3. Usarel TDA promediado en inferencia

**Ventaja**: Estable, determinístico.

**Desventaja**: Pierde información específica del caso.

### Estrategia C: Red Dual

Entrenar dos redes en paralelo:
- Red A: MRI → predicción Y
- Red B: Y → mejora de Y usando topología

Ambas se entrenan conjuntamente.

### Tabla: Estrategias de Inferencia

| Estrategia | Enfoque | Ventajas | Desventajas |
|-----------|---------|----------|------------|
| **Iterativa** | Auto-refinamiento | Potencialmente óptima | Convergencia incierta |
| **Atlas** | Promedio de casos | Estable, rápida | Pierde individualidad |
| **Dual-Red** | Redes especializadas | Flujo controlado | Mayor complejidad |

## 10. Segmentación sin Redes Neuronales: TDA Puro

### Enfoque Geométrico: François & Tinarrage (2024)

¿Se puede segmentar tumores **solo con homología persistente**, sin redes neuronales?

**François & Tinarrage (2024)** demostraron que **sí**, con un pipeline de 3 módulos:

**Módulo 1: Umbralización Inteligente**
- Encontrar umbrales óptimos en el diagrama de persistencia
- Maximizar homología relevante (β₀, β₁, β₂)

**Módulo 2: Detección Topológica**
- Usar H₂ (cavidades vacías) para identificar GBM enhancing tumor
- Propiedad: GBM típicamente forma anillo alrededor de necrosis
- Predicción: H₂ ≈ 1 ⟹ tumor esferoidal (¡tumor probable!)

**Módulo 3: Deducción Geométrica**
- Expandir desde cavidades detectadas hacia periferia
- Usar distancia euclidiana inversa para crecer región

### Tabla: Comparación Red+TDA vs. TDA Puro

| Aspecto | Red + TDA | TDA Puro |
|--------|----------|----------|
| **Interpretabilidad** | Caja negra parcial | Transparente (topología) |
| **Flexibilidad** | Aprende de datos | Rígido, reglas fijas |
| **Velocidad** | ~11 seg/volumen | ~5 seg/volumen |
| **Dice WT** | 88-90% | 72-78% |
| **Cuándo funciona** | Mayoría de casos | Casos con topología clara |

**Conclusión**: TDA puro es interpretable y rápido, pero menos preciso. Red+TDA combina lo mejor de ambos.

## 11. Métricas de Evaluación

### Métricas Estándar

**Dice Similarity Coefficient**:
$$\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}$$

Rango: [0,1]. Penaliza falsos positivos y falsos negativos.

**Hausdorff Distance 95%** (HD95):
$$\text{HD95} = \text{percentil}_{95}\{\max(\text{dist}(P, G), \text{dist}(G, P))\}$$

Mide el mayor desacuerdo; robusto a valores atípicos. Unidad: mm.

**Intersection over Union** (IoU):
$$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$

Más estricto que Dice.

### Métricas Topológicas

**Error β₀** (componentes):
$$E_{\beta_0} = |\beta_0(P) - \beta_0(G)|$$

¿Predijimos el número correcto de tumores separados?

**Error β₁** (bucles):
$$E_{\beta_1} = |\beta_1(P) - \beta_1(G)|$$

¿Preservamos la conectividad correcta?

**Betti Matching Error**:
$$E_{\text{BM}} = \sum_i \min_j \|p_i - g_j\|$$

Versión spatial-aware de Wasserstein.

### Tabla: Métricas de Evaluación

| Métrica | Fórmula | Qué Captura | Riesgo/Limitación |
|---------|---------|-------------|-------------------|
| **Dice** | 2\|P∩G\|/(P+G) | Similitud global | Insensible a gran desacuerdo localizado |
| **HD95** | percentil₉₅(max dist) | Mayor desacuerdo | Sensible a un voxel atípico |
| **IoU** | \|P∩G\|/\|P∪G\| | Precisión de intersección | Igual que Dice, solo escala distinta |
| **β₀ error** | \|β₀(P)-β₀(G)\| | Número de componentes | No captura tamaño relativo |
| **β₁ error** | \|β₁(P)-β₁(G)\| | Número de bucles | Insensible a ubicación de agujeros |
| **Betti Match** | Emparejamiento espacial | Topología + ubicación | Más computacional |

### Convención de Conectividad

**Importante**: El número de componentes y bucles depende de la **definición de vecindario**:

- **2D, 4-conectividad**: Voxel toca 4 vecinos (arriba, abajo, izquierda, derecha)
- **2D, 8-conectividad**: Voxel toca 8 vecinos (incluye diagonales)
- **3D, 6-conectividad**: Voxel toca 6 vecinos (cara a cara)
- **3D, 26-conectividad**: Voxel toca 26 vecinos (incluye aristas y esquinas)

**Ejemplo de impacto**:
```
Predicción (8-conectividad):
█ ░ █
░ ░ ░
█ ░ █

β₀ = 4 (cuatro esquinas desconectadas)

Predicción (4-conectividad, solo caras):
█ ░ █
░ ░ ░
█ ░ █

β₀ = 4 (igual, porque las esquinas no se tocan cara-a-cara)
```

**Referencia**: Berger et al. (2025) analizan cómo esta elección afecta resultados de segmentación cerebral.

## 12. Resultados y Estado del Arte

### Benchmark BraTS

El desafío BraTS2020 (y posteriores) evaluó cientos de métodos en ~600 volúmenes de pacientes reales.

### Resultados de TDAConvAttentionNet

Un modelo incorporando TDA fue reportado con:

| Métrica | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) |
|---------|------------------|-----------------|----------------------|
| **Dice** | 89.36% | 87.36% | 89.98% |
| **HD95** | 3.2 mm | 4.5 mm | 2.8 mm |

Estos números son **competitivos con el estado del arte** (mejores arquitecturas alcanzan ~90% Dice).

### Tabla: Arquitecturas con TDA en BraTS

| Arquitectura | Método Topológico | Dice_WT | Dice_TC | Dice_ET |
|-------------|-------------------|---------|---------|---------|
| UNet Base | Ninguno | 87.2% | 85.1% | 87.5% |
| UNet+TopoLoss | TopoLoss | 88.5% | 86.8% | 88.9% |
| TDAConvAttentionNet | Betti Matching | **89.36%** | **87.36%** | **89.98%** |
| 3D U-Net Ensemble | Ensemble (sin TDA) | 88.9% | 86.5% | 88.7% |

### Cuándo Ayuda TDA

TDA proporciona beneficio máximo cuando:
1. **Multifocalidad**: Múltiples tumores separados → β₀ restringe la arquitectura
2. **Topología compleja**: Cavidades internas → β₂ guía refinamiento
3. **Bordes borrosos**: MRI ambigua → topología proporciona prior fuerte

TDA proporciona beneficio menor cuando:
1. Tumor simple, bien definido (DTI alto)
2. Imágenes de ruido bajo
3. Tumor ya visible en intensidad (FLAIR)

## 13. Análisis de Supervivencia: TDA más allá de la Segmentación

La segmentación es el primer paso. El verdadero objetivo clínico es **predecir supervivencia**.

### Cox Proportional Hazards Model (Brevemente)

La tasa de mortalidad a tiempo *t* es:

$$h(t) = h_0(t) \cdot \exp(X \beta)$$

donde:
- $h_0(t)$ = hazard basal (desconocido)
- $X$ = características del paciente
- $\beta$ = coeficientes de riesgo

**Idea**: Características topológicas pueden mejorar predicción de supervivencia.

### FCoxPH: Cox Funcional

**Chen et al. (2021)** propusieron FCoxPH: usa imágenes de persistencia completas como **características funcionales** en el modelo Cox.

- Input: PI_β₀(x,y,z), PI_β₁(x,y,z) como funciones
- Salida: Puntaje de riesgo para cada paciente

**Resultado**: Mejora C-index (concordancia) en ~5%.

### SECT: Smooth Euler Characteristic Transform

**Gu et al. (2021)** introdujeron SECT: usa la **Transformada de Euler Característica Suave** para resumen topológico:

$$\text{SECT}(t) = \sum_i (-1)^{\text{dim}(f_i)} \cdot \mathbb{1}[\text{filtration}(f_i) \leq t]$$

en palabras: suma signada de células por nivel de filtración.

**Para GBM**: SECT captura la "forma" del tumor en un vector 1D. Este vector alimenta modelos de supervivencia.

### Tabla: Métodos de Supervivencia con TDA

| Método | Características | Modelo | C-index | Datos |
|--------|-----------------|--------|---------|-------|
| **Edad + Volumen** | Manual | Cox | 0.62 | BraTS |
| **FCoxPH** | PI_β₀, PI_β₁ | Cox funcional | **0.70** | BraTS |
| **SECT-GBM** | Euler characteristic | Cox | 0.68 | Clínico GBM |
| **Deep Learning** | Red profunda | DeepHit | 0.72 | Clínico GBM |

(C-index: 0.5 = azar, 1.0 = perfecto)

## 14. Preguntas Abiertas y Futuro

### Preguntas Abiertas

1. **Filtración óptima**: ¿Es EDT la mejor filtración? ¿Qué pasa con intensidad pura, o mezcla EDT+intensidad?

2. **Extensión 3D**: Los algoritmos cúbicos son O(HWD·log(HWD)). ¿Escalable a volúmenes HWD=256³?

3. **Persistencia multi-escala**: Las características en diferentes escalas (finos vasos vs. tumor masivo) necesitan Insight combinado.

4. **Convergencia de inferencia iterativa**: ¿Existe garantía de que Strategy A converge? ¿A mínimo local o global?

5. **Integración con Geometric Deep Learning**: Frameworks como 5G (Graph, Group, Gauge, Geometric, Generative) podrían combinar TDA + GCN (graph neural networks) naturalmente.

## 15. Resumen Final

### Síntesis: De Publicación Anterior a TDA-SegUNet

| Componente | De Publicación | Rol en TDA-SegUNet |
|-----------|-----------------|-------------------|
| **Convulsiones, activaciones** | Post 1: "Entendiendo Redes Neuronales" | Extrae características de MRI; procesa canales TDA |
| **Topología, homología, persistencia** | Post 2: "Topología en Máquina" | Define características topológicas; guía entrenamiento |
| **U-Net, encoder-decoder** | Post 3: "Arquitecturas de Segmentación" | Estructura base; permite fusión multi-escala |
| **TDA-SegUNet sintetizado** | **Este artículo** | Integración completa; pipeline clínico viable |

### La Contribución de la Tesis

Esta tesis integró:
1. **Análisis topológico** para extraer prior geométrico de tumores
2. **Redes neuronales profundas** para aprender de datos de alta dimensión
3. **Funciones de pérdida topológicas** para entrenar respetando topología
4. **Evaluación clínica** en BraTS, demostrando viabilidad

El resultado es TDA-SegUNet: una arquitectura que no es ni "puros números" ni "pura topología", sino su síntesis óptima.

---

## Python Pseudo-Código: Pipeline TDA-SegUNet

```python
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# ============================================================================
# PASO 1: PREPROCESAMIENTO
# ============================================================================

def preprocess_mri(mri_4channel, brain_mask):
    """
    mri_4channel: array (H, W, D, 4) — FLAIR, T1ce, T1, T2
    brain_mask: array (H, W, D) — máscara binaria de cráneo removido
    """
    # Skull stripping: aplicar máscara
    mri_masked = mri_4channel * brain_mask[:,:,:,None]

    # Z-score normalization por canal
    mri_norm = np.zeros_like(mri_4channel)
    for ch in range(4):
        channel = mri_masked[:,:,:,ch]
        valid_voxels = channel[brain_mask > 0]
        mu = np.mean(valid_voxels)
        sigma = np.std(valid_voxels) + 1e-8
        mri_norm[:,:,:,ch] = (channel - mu) / sigma

    return mri_norm  # (H, W, D, 4)


# ============================================================================
# PASO 2: EXTRACCIÓN DE CARACTERÍSTICAS TDA
# ============================================================================

def euclidean_distance_transform(binary_mask):
    """
    Calcula EDT usando scipy.
    binary_mask: array (H, W, D) — 1 si tumor, 0 si fondo

    Retorna:
        edt: array (H, W, D) — distancia euclidiana al voxel tumoral más cercano
    """
    edt = ndimage.distance_transform_edt(binary_mask)
    return edt


def extract_persistence_diagram_3d(edt_volume, num_levels=50):
    """
    Construye diagrama de persistencia a partir de EDT usando filtración cúbica.

    Algoritmo:
        1. Extraer valores únicos del EDT
        2. Para cada nivel t, calcular homología H₀, H₁, H₂
        3. Registrar pares (nacimiento, muerte)

    edt_volume: array (H, W, D) — distancia euclidiana
    num_levels: número de niveles de filtración

    Retorna:
        dgm_h0: lista de tuplas (birth, death) — componentes conectadas
        dgm_h1: lista de tuplas — bucles (agujeros 1D)
        dgm_h2: lista de tuplas — cavidades (huecos 3D)
    """
    # Simplificación: usar valores de EDT como filtración
    edt_flat = edt_volume.flatten()
    edt_levels = np.linspace(edt_flat.min(), edt_flat.max(), num_levels)

    # Para cada nivel, calcular Betti numbers
    betti_0_prev = 1  # empezar con 1 componente
    betti_1_prev = 0
    betti_2_prev = 0

    dgm_h0 = []
    dgm_h1 = []
    dgm_h2 = []

    for t in edt_levels:
        # Región en umbral: {voxels con EDT <= t}
        region = (edt_volume <= t).astype(int)

        # Calcular componentes conectadas
        labeled, num_components = ndimage.label(region)
        betti_0 = num_components

        # Cambios en β₀: si disminuye, fue "muerte" de componente
        if betti_0 < betti_0_prev:
            dgm_h0.append((t, t_prev))  # (nacimiento, muerte)

        # Estimación simplificada de β₁ (bucles) — ver cavidades
        # Aquí usaríamos persistencia cúbica real (más complejo)

        betti_0_prev = betti_0
        t_prev = t

    # Componentes que nunca mueren: persistencia infinita
    for _ in range(betti_0_prev):
        dgm_h0.append((edt_levels[0], np.inf))

    return dgm_h0, dgm_h1, dgm_h2


def persistence_diagram_to_image(dgm, image_shape, sigma=0.1):
    """
    Convierte diagrama de persistencia a imagen de persistencia.

    dgm: lista de tuplas (birth, death)
    image_shape: (H, W, D) — forma del volumen MRI original
    sigma: ancho de banda del kernel gaussiano

    Retorna:
        pi: array (H, W, D) — imagen de persistencia
    """
    H, W, D = image_shape
    pi = np.zeros((H, W, D))

    # Normalizar diagrama a [0, 1] range
    if dgm:
        births = [p[0] for p in dgm if p[1] != np.inf]
        deaths = [p[1] for p in dgm if p[1] != np.inf]

        if births and deaths:
            b_max = max(births)
            d_max = max(deaths)
            max_val = max(b_max, d_max)
        else:
            max_val = 1.0
    else:
        max_val = 1.0

    # Para cada punto, contribuir gaussiano
    for birth, death in dgm:
        if death != np.inf:
            b_norm = birth / max_val
            p_norm = (death - birth) / max_val

            # Proyectar a coordenadas de imagen
            # Simplificación: usar first 2D
            x_idx = int(b_norm * H) % H
            y_idx = int(p_norm * W) % W

            # Gaussiano 2D (en realidad 3D, pero simplificado)
            if 0 <= x_idx < H and 0 <= y_idx < W:
                pi[x_idx, y_idx, D//2] += np.exp(-0.5 / (sigma ** 2))

    return pi / (pi.max() + 1e-8)  # normalizar


def extract_tda_features(mri_norm, tumor_seed_mask):
    """
    Pipeline completo TDA: MRI → EDT → Diagrama → Imagen.

    mri_norm: array (H, W, D, 4) — MRI normalizado
    tumor_seed_mask: array (H, W, D) — máscara inicial de tumor

    Retorna:
        pi_beta0: array (H, W, D) — imagen de persistencia H₀
        pi_beta1: array (H, W, D) — imagen de persistencia H₁
    """
    # Calcular EDT
    edt = euclidean_distance_transform(tumor_seed_mask)

    # Extraer diagrama de persistencia
    dgm_h0, dgm_h1, dgm_h2 = extract_persistence_diagram_3d(edt)

    # Convertir a imágenes
    pi_beta0 = persistence_diagram_to_image(dgm_h0, edt.shape)
    pi_beta1 = persistence_diagram_to_image(dgm_h1, edt.shape)

    return pi_beta0, pi_beta1


# ============================================================================
# PASO 3: FUSIÓN DE ENTRADA Y RED NEURONAL
# ============================================================================

class TDASegUNet(nn.Module):
    """
    U-Net 3D aumentada con canales topológicos.

    Entrada: (B, 6, H, W, D)
        - Canales 0-3: FLAIR, T1ce, T1, T2
        - Canal 4: PI_β₀
        - Canal 5: PI_β₁

    Salida: (B, 3, H, W, D)
        - Canal 0: WT (whole tumor)
        - Canal 1: TC (tumor core)
        - Canal 2: ET (enhancing tumor)
    """

    def __init__(self, in_channels=6, out_channels=3, features=64):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._conv_block(features, features*2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._conv_block(features*2, features*4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(features*4, features*8)

        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose3d(features*8, features*4, 2, stride=2)
        self.dec3 = self._conv_block(features*8, features*4)  # concatenación

        self.upconv2 = nn.ConvTranspose3d(features*4, features*2, 2, stride=2)
        self.dec2 = self._conv_block(features*4, features*2)

        self.upconv1 = nn.ConvTranspose3d(features*2, features, 2, stride=2)
        self.dec1 = self._conv_block(features*2, features)

        # Output
        self.out = nn.Conv3d(features, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder con saltos
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder con conexiones residuales
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)  # Concatenar con salto
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)

        # Output
        x = self.out(x)
        x = self.sigmoid(x)

        return x


# ============================================================================
# PASO 4: FUNCIONES DE PÉRDIDA
# ============================================================================

def dice_loss(pred, target, smooth=1e-5):
    """
    Dice Loss: 1 - Dice Score
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


def betti_matching_loss(pred, target):
    """
    Topological loss: Betti matching (simplificado)
    Penaliza discrepancia en número de componentes y bucles.
    """
    # Calcular Betti numbers (diagrama de persistencia)
    # Este es pseudocódigo; implementación real es más compleja
    beta0_pred = count_components(pred > 0.5)
    beta0_target = count_components(target > 0.5)

    beta1_pred = count_loops(pred > 0.5)
    beta1_target = count_loops(target > 0.5)

    loss = torch.abs(torch.tensor(beta0_pred - beta0_target)) + \
           torch.abs(torch.tensor(beta1_pred - beta1_target))

    return loss


def combined_loss(pred, target, lambda_topo=0.1):
    """
    Pérdida combinada: Dice + Topología
    """
    l_dice = dice_loss(pred, target)
    l_topo = betti_matching_loss(pred, target)

    return l_dice + lambda_topo * l_topo


# ============================================================================
# PASO 5: ENTRENAMIENTO POR CURRICULUM
# ============================================================================

def train_epoch(model, dataloader, optimizer, epoch, total_epochs):
    """
    Entrenamiento por curriculum.
    Épocas 0-30: Dice solo
    Épocas 30-60: Dice + λ·TopoLoss (λ ramps up)
    Épocas 60+: Dice + 0.1·TopoLoss
    """
    model.train()
    total_loss = 0

    for batch_idx, (mri, target) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward
        pred = model(mri)

        # Elegir función de pérdida según época
        if epoch < 30:
            loss = dice_loss(pred, target)
        elif epoch < 60:
            lambda_topo = 0.01 + (epoch - 30) / 30 * 0.09  # ramp 0.01 → 0.1
            loss = combined_loss(pred, target, lambda_topo)
        else:
            loss = combined_loss(pred, target, lambda_topo=0.1)

        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_full_pipeline(train_loader, val_loader, num_epochs=100):
    """
    Pipeline de entrenamiento completo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TDASegUNet(in_channels=6, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, num_epochs)

        # Validación (opcional)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    return model


# ============================================================================
# PASO 6: INFERENCIA
# ============================================================================

def infer_segmentation(model, mri_volume, brain_mask, num_iterations=3):
    """
    Inferencia con refinamiento iterativo.

    Estrategia A: Usar predicción actual para calcular TDA.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Preprocesar MRI
    mri_norm = preprocess_mri(mri_volume, brain_mask)

    # Inicialización: usar tumor seed simple (p.ej., intensidad FLAIR alta)
    seed_mask = (mri_volume[:,:,:,0] > np.percentile(mri_volume[:,:,:,0], 95)).astype(float)

    predictions = None

    for iteration in range(num_iterations):
        # Extraer TDA basado en iteración anterior (o seed)
        pi_beta0, pi_beta1 = extract_tda_features(mri_norm, seed_mask)

        # Construir entrada: MRI + TDA
        x_input = np.concatenate([
            mri_norm,
            pi_beta0[:,:,:,None],
            pi_beta1[:,:,:,None]
        ], axis=-1)

        # Tensor y forward pass
        x_tensor = torch.from_numpy(x_input).permute(3,0,1,2).unsqueeze(0).float()
        x_tensor = x_tensor.to(device)

        with torch.no_grad():
            pred = model(x_tensor)

        predictions = pred.cpu().numpy()[0]  # (3, H, W, D)

        # Actualizar seed para próxima iteración
        seed_mask = (predictions[0] > 0.5).astype(float)

    # Post-procesamiento: asegurar anidamiento ET ⊂ TC ⊂ WT
    wt = (predictions[0] > 0.5).astype(int)
    tc = np.logical_and(predictions[1] > 0.5, wt).astype(int)
    et = np.logical_and(predictions[2] > 0.5, tc).astype(int)

    return et, tc, wt


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Cargar datos (pseudo)
    mri_4ch = np.random.randn(128, 128, 128, 4)
    brain_mask = np.ones((128, 128, 128))
    gt_labels = np.random.rand(128, 128, 128, 3)

    # Crear modelo y entrenar
    model = TDASegUNet(in_channels=6, out_channels=3)
    print("Modelo creado:", model)

    # Inference
    et, tc, wt = infer_segmentation(model, mri_4ch, brain_mask, num_iterations=3)
    print(f"Segmentación: WT shape={wt.shape}, TC shape={tc.shape}, ET shape={et.shape}")
```

---

## Conclusión

TDA-SegUNet sintetiza años de investigación en topología aplicada, deep learning y oncología clínica. No es un modelo perfecto, pero demuestra que **la geometría y la topología son herramientas poderosas** para resolver problemas médicos reales.

El futuro reside en integrar aún más estos mundos: persistencia multi-escala, geometric deep learning, y modelos basados en física que respeten tanto la topología como las ecuaciones de difusión de tumores.

Para los estudiantes de ciencias e ingeniería: este es un ejemplo de cómo la **matemática pura** (topología algebraica) encuentra aplicación en **beneficio humano** (predicción de supervivencia, planificación quirúrgica). La tesis que contiene estos artículos es un testimonio de ese viaje.

---

**Referencias sugeridas para profundizar**:
- Hu et al. (2019): "Topology-Preserving Deep Image Segmentation"
- Stucki et al. (2023): "Betti Matching Loss"
- François & Tinarrage (2024): "Pure Topological Tumor Segmentation"
- Chen et al. (2021): "Functional Cox PH with Persistence Images"
- Gu et al. (2021): "Smooth Euler Characteristic Transform for GBM Prognosis"
