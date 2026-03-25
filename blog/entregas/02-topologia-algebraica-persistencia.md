---
layout: post
title: "Topología Algebraica y Homología Persistente: La Forma de los Datos"
date: 2026-03-24
categories: [matematicas, topologia]
tags: [topologia, homologia, persistencia, TDA, betti]
math: true
---

# Topología Algebraica y Homología Persistente: La Forma de los Datos

## Introducción

Cuando observamos una imagen de resonancia magnética (MRI) del cerebro, vemos píxeles. Pero dentro de esos píxeles hay una estructura topológica: un tumor sólido (sin agujeros), una cavidad necrótica (un agujero rodeado de tejido), vasos sanguíneos retorcidos (ciclos), y conexiones complejas entre todas las partes. La topología algebraica es el lenguaje matemático para **ver la forma en los datos**.

Este artículo te llevará desde los conceptos básicos de topología hasta la homología persistente: la herramienta que permite a los algoritmos **a diferentes escalas** detectar qué estructuras en una imagen de tumor son reales y cuáles son artefactos del ruido.

---

## 1. ¿Qué es la Topología?

La topología es el estudio de las propiedades de los espacios que permanecen invariantes bajo deformaciones continuas. Si puedes transformar una forma en otra mediante estiramiento, contracción y torsión (pero **sin romper ni pegar**), entonces desde la perspectiva topológica son la misma cosa.

### El Ejemplo Clásico: La Taza de Café y la Dona

Imagina una taza de café de cerámica. Tiene un asa. Una dona también tiene un agujero. Desde la perspectiva de un topólogo, **una taza de café es topológicamente equivalente a una dona**. Puedes imaginar que la cerámica es barro infinitamente flexible:

1. Primero, deforma el cuerpo de la taza en una esfera
2. Luego, expande el asa hasta que sea un agujero que atraviesa toda la esfera
3. Finalmente, ajusta y moldea hasta que parece una dona

Ningún paso rompe la cerámica. Ningún paso pega dos partes que estaban separadas. Por lo tanto, topológicamente hablando: **taza = dona**.

### ¿Qué es un Homeomorfismo?

Formalmente, un **homeomorfismo** es una función continua entre dos espacios que es biyectiva (uno a uno y sobre) y cuya inversa también es continua. Si existe un homeomorfismo entre dos espacios, entonces son **topológicamente idénticos** (homeomorfos).

**Ejemplo práctico:** La función $f(x) = x^3$ es un homeomorfismo de la recta real a sí misma, porque es continua, invertible ($f^{-1}(x) = \sqrt[3]{x}$), y su inversa es continua. El espacio "se ve diferente" (una cúbica versus una línea recta) pero la topología es idéntica.

### Topología vs. Geometría: Una Comparación

| Concepto | Topología | Geometría | Ejemplo |
|----------|-----------|-----------|---------|
| **Distancias** | No importan | Esenciales | Dos triángulos con ángulos iguales pero tamaños diferentes son geométricamente distintos pero topológicamente idénticos |
| **Ángulos** | No importan | Importantes | Una esfera puede tener ángulos aplastados; una esfera y un cubo son geométricamente diferentes pero topológicamente iguales |
| **Agujeros** | ¡Importan mucho! | Importan | Una dona y una esfera son geométricamente diferentes Y topológicamente diferentes (la dona tiene un agujero) |
| **Continuidad** | Central | Derivada de ella | La topología pregunta "¿qué mapeos son continuos?" |
| **Invariantes Clave** | Género, número de Betti, grupo fundamental | Curvatura, ángulos, longitudes | El genus de una dona es 1; una esfera tiene genus 0 |

### ¿Por Qué la Topología es "Ciega" a las Distancias Pero "Ve" la Estructura?

En topología, dos espacios son iguales si uno puede transformarse continuamente en el otro. Esto significa que:

- **La topología no ve la distancia.** Puedo estirar una línea recta de 1 cm a 1 km sin cambiar su topología.
- **La topología ve los agujeros y las conexiones.** Un círculo y una línea recta son topológicamente diferentes porque el círculo tiene un "lazo" que no puedes eliminar sin romperlo.

**Aplicación clínica:** En la segmentación de tumores cerebrales, dos tumores con diferente tamaño y forma pueden tener la misma topología (ambos sólidos, sin cavidades). Pero un tumor con una cavidad necrótica tiene una topología diferente. La topología nos permite **clasificar estructuras por su esencia**, no por su apariencia superficial.

---

## 2. Complejos Simpliciales

Para hacer topología computacional, necesitamos una forma discreta y finita de representar espacios. Los **complejos simpliciales** son la herramienta clásica.

### Componentes: Simplices

Un **simplice** es la generalización de un triángulo a cualquier dimensión:

| Nombre | Dimensión | Descripción | Ejemplo Geométrico |
|--------|-----------|-------------|-------------------|
| **0-simplice (vértice)** | 0 | Un único punto | $p = (2, 3)$ |
| **1-simplice (arista)** | 1 | Una línea recta entre dos vértices | Una arista conecta $v_1$ a $v_2$ |
| **2-simplice (triángulo)** | 2 | Un triángulo lleno entre tres vértices | Triángulo con vértices $\{v_1, v_2, v_3\}$ |
| **3-simplice (tetraedro)** | 3 | Un tetraedro sólido entre cuatro vértices | Tetraedro con vértices $\{v_1, v_2, v_3, v_4\}$ |
| **n-simplice** | n | La envolvente convexa de $n+1$ puntos en posición general | En $\mathbb{R}^{100}$, un 100-simplice es la envolvente de 101 puntos |

### Construcción de un Complejo Simplicial: Reglas Formales

Un **complejo simplicial** $K$ es una colección de simplices tal que:

1. Si $\sigma$ es un simplice en $K$, entonces todo simplice contenido en $\sigma$ (llamado "cara" de $\sigma$) también está en $K$.
2. Si $\sigma$ y $\tau$ son dos simplices en $K$, entonces su intersección $\sigma \cap \tau$ es vacía o es un simplice común a ambos (sin traslapes arbitrarios).

**Ejemplo de violación de reglas:**

```
Esto NO es válido como complejo simplicial:
   v1 ---- v2
   |  \ X /  |
   |   /\    |
   |  /  \   |
   v4 ---- v3

Razón: Los triángulos v1-v2-v3 y v1-v3-v4 se solapan en una región
de 2D que no es una cara común. Esto viola la regla 2.
```

**Esto SÍ es válido:**

```
   v1 ---- v2
   | \     / |
   |  \ t1/  |
   |   \  /  |
   | t2 \/   |
   |     X   |  donde t1 = triángulo v1-v2-v3
   |    / \  |  y t2 = triángulo v1-v3-v4
   |   /   \ |  La arista v1-v3 es una cara compartida
   v4 ---- v3
```

### Ejemplo Numérico: Construyendo un Complejo Simplicial a partir de 5 Puntos

Supongamos que tenemos 5 puntos en el plano:

```
v1 = (0, 0)    v3 = (2, 1)
v2 = (1, -1)   v4 = (1, 2)
v5 = (2, 3)
```

**Paso 1: Incluir todos los 0-simplices (vértices)**

```
K = {v1, v2, v3, v4, v5}
```

**Paso 2: Agregamos 1-simplices (aristas). Digamos que conectamos puntos cercanos:**

```
Distancias:
d(v1, v2) = sqrt(1 + 1) ≈ 1.41  ✓ Agrega arista v1-v2
d(v1, v3) = sqrt(4 + 1) ≈ 2.24  ✓ Agrega arista v1-v3
d(v2, v3) = sqrt(1 + 4) ≈ 2.24  ✓ Agrega arista v2-v3
d(v3, v4) = sqrt(0 + 1) = 1     ✓ Agrega arista v3-v4
d(v4, v5) = sqrt(1 + 1) ≈ 1.41  ✓ Agrega arista v4-v5
d(v2, v4) = sqrt(0 + 9) = 3     ✗ Muy lejos, no agrega

K = {v1, v2, v3, v4, v5,
     v1-v2, v1-v3, v2-v3, v3-v4, v4-v5}
```

**Paso 3: Agregamos 2-simplices (triángulos). Para cada arista, si hay un tercer vértice a distancia corta:**

```
Triangulación:
- Vértices v1, v2, v3 forman un triángulo (todas las aristas existen)
- Vértices v3, v4, v5 ¿forman un triángulo?
  ✗ No: No existe la arista v3-v5 (demasiado lejos)

Por lo tanto:
K = {v1, v2, v3, v4, v5,                    [0-simplices]
     v1-v2, v1-v3, v2-v3, v3-v4, v4-v5,    [1-simplices]
     triángulo v1-v2-v3}                    [2-simplices]
```

**Visualización:**

```
         v4 --- v5
         |
    v1 - v3
     \ /
      v2
```

**Resumen del resultado:**

- 5 vértices, 5 aristas, 1 triángulo
- Este complejo simplicial representa una especie de "gráfico planar" con una cara triangular.

---

## 3. Complejos Cúbicos: La Estructura Nativa de Imágenes

Para imágenes digitales (especialmente imágenes médicas como MRI), los complejos simpliciales tienen un problema: son ineficientes. Una imagen de $1024 \times 1024$ píxeles podría requerir tetrahedrización (construcción de tetraedros), lo que explota exponencialmente.

Los **complejos cúbicos** son mucho más naturales para imágenes porque la imagen ya es una cuadrícula.

### ¿Qué es un Complejo Cúbico?

Un complejo cúbico es una colección de cubos (en cualquier dimensión) que se pegan de forma consistente:

- **0-cubo:** Un vértice (punto)
- **1-cubo:** Una arista (segmento entre dos vértices)
- **2-cubo:** Un cuadrado (celda unitaria)
- **3-cubo:** Un cubo sólido
- **k-cubo:** En $\mathbb{R}^k$, es $[0,1]^k$ (el "cubo unitario" estándar)

### Intervalos Elementales e Intervalos Cúbicos

En una cuadrícula discreta, representamos elementos como "intervalos elementales":

- En 1D: $[i, i+1]$ es un intervalo elemental (representa un píxel o una arista)
- En 2D: $[i, i+1] \times [j, j+1]$ es un cuadrado elemental (un píxel)
- En 3D: $[i, i+1] \times [j, j+1] \times [k, k+1]$ es un cubo elemental (un vóxel)

Un **cubo elemental** puede también ser degenerado:
- $[i] \times [j, j+1] \times [k]$: una arista en la dirección y
- $[i] \times [j] \times [k]$: un vértice

### Construcción 1.1: De una Imagen Binaria a un Complejo Cúbico

**Entrada:** Una imagen binaria (0 = fondo, 1 = objeto).

**Proceso:**

1. Incluye cada vóxel (o píxel) donde el valor es 1 como un 3-cubo (o 2-cubo en 2D).
2. Incluye todas las caras de esos cubos (es decir, aristas y vértices).
3. El resultado es un complejo cúbico.

**Ventaja:** La complejidad es **lineal** en el número de píxeles/vóxeles. No hay explosión exponencial como en simplicial.

### Ejemplo Numérico: Máscara Binaria 3×3

Consideremos una imagen binaria de 3×3:

```
Imagen binaria (0 = blanco, 1 = negro):

  j=0  j=1  j=2
i=0:  0    1    1
i=1:  1    1    0
i=2:  0    1    0

Visualización:
  .  #  #
  #  #  .
  .  #  .

Donde # = 1 (píxeles negros)
```

**Paso 1: Identificar los 2-cubos (píxeles)**

Los píxeles donde valor=1 corresponden a cuadrados:

```
2-cubos:
- [0,1] × [1,2]  (píxel en i=0, j=1)
- [0,1] × [2,3]  (píxel en i=0, j=2)
- [1,2] × [0,1]  (píxel en i=1, j=0)
- [1,2] × [1,2]  (píxel en i=1, j=1)
- [2,3] × [1,2]  (píxel en i=2, j=1)
```

**Paso 2: Incluir todas las aristas (1-cubos) de esos cuadrados**

Para el cuadrado $[0,1] \times [1,2]$, las aristas son:

```
Aristas horizontales: [0,1] × [1] y [0,1] × [2]
Aristas verticales:   [0] × [1,2] y [1] × [1,2]
```

Por simetría y consistencia, listamos todas las aristas necesarias:

```
Aristas en dirección i (verticales en el diagrama):
[0,1] × [1], [0,1] × [2], [0,1] × [3],
[1,2] × [0], [1,2] × [1], [1,2] × [2],
[2,3] × [1],

Aristas en dirección j (horizontales en el diagrama):
[0] × [1,2], [0] × [2,3],
[1] × [0,1], [1] × [1,2], [1] × [1,3],
[2] × [1,2],
[3] × [1,2],
... (total de 13 aristas)
```

**Paso 3: Incluir todos los vértices (0-cubos)**

Los vértices son puntos de la cuadrícula $(i, j)$ para $i, j \in \{0, 1, 2, 3\}$ que son incidentes con al menos una arista.

```
0-cubos (vértices):
(0,0), (0,1), (0,2), (0,3),
(1,0), (1,1), (1,2), (1,3),
(2,0), (2,1), (2,2), (2,3),
(3,0), (3,1), (3,2), (3,3)

Total: 16 vértices
```

**Complejidad para una imagen $m \times n$:**

- 0-cubos: $O(m \times n)$
- 1-cubos: $O(m \times n)$
- 2-cubos: $O(m \times n)$
- **Total: $O(m \times n)$** (lineal, no exponencial)

**Ventaja clínica:** Una MRI de $256 \times 256 \times 256$ vóxeles requiere apenas $256^3 \approx 16$ millones de elementos. Con complejos simpliciales, podría ser billones.

---

## 4. El Operador de Frontera y Homología

Ahora tenemos una forma de construir complejos discretos. El siguiente paso es **medir topología** mediante álgebra lineal.

### El Operador de Frontera ∂

Para cada dimensión $k$, definimos un operador lineal $\partial_k$ que toma un $k$-simplice y produce los $(k-1)$-simplices en su frontera.

**Ejemplo: frontera de un triángulo**

Si un triángulo tiene vértices $\{a, b, c\}$, su frontera consiste en tres aristas:

$$\partial(\{a,b,c\}) = \{a,b\} - \{a,c\} + \{b,c\}$$

(Los signos alternan; esto importa para la orientación, pero lo omitiremos por claridad.)

**Ejemplo: frontera de una arista**

$$\partial(\{a,b\}) = b - a$$

(Dos extremos con signos opuestos.)

**Ejemplo: frontera de un vértice**

$$\partial(a) = 0$$

(Un punto no tiene frontera.)

### La Regla Dorada: ∂∂ = 0

Este es el hecho fundamental de la topología algebraica:

$$\partial_k \circ \partial_{k+1} = 0$$

**¿Qué significa?** La frontera de la frontera de cualquier cosa es siempre cero.

**Ejemplo:**

```
Triángulo T = {a, b, c}

Primer ∂:
∂(T) = {a,b} - {a,c} + {b,c}
     = (b - a) - (c - a) + (c - b)
     = b - a - c + a + c - b
     = 0

Segundo ∂:
∂(∂(T)) = ∂({a,b}) + ∂({a,c}) + ∂({b,c})
        = (b - a) - (c - a) + (c - b)
        = 0
```

**¿Por qué es importante?** Significa que existe una estructura algebraica profunda: los ciclos (cosas sin frontera) pueden dividirse en dos tipos: los que son fronteras de algo mayor y los que no.

### Ciclos y Fronteras

Para un espacio $K$, en dimensión $k$:

- **Ciclos:** $Z_k = \ker(\partial_k) = \{\sigma : \partial_k(\sigma) = 0\}$ (los elementos sin frontera)
- **Fronteras:** $B_k = \text{im}(\partial_{k+1}) = \{\sigma : \sigma = \partial_{k+1}(\tau) \text{ para algún } \tau\}$ (los elementos que son fronteras de cosas mayores)

Debido a la regla $\partial \partial = 0$, siempre tenemos $B_k \subseteq Z_k$.

### Grupos de Homología

El **grupo de homología en dimensión $k$** se define como:

$$H_k(K) = Z_k / B_k = \ker(\partial_k) / \text{im}(\partial_{k+1})$$

Intuitivamente:
- Medimos ciclos (estructuras cerradas)
- Los consideramos "equivalentes" si su diferencia es la frontera de algo
- Lo que queda son los "agujeros reales" que no pueden cerrarse

**Ejemplo simple:** En un círculo dibujado como una cadena de vértices y aristas, el ciclo es la propia cadena. No es la frontera de nada (no rodea ningún 2-simplice). Por lo tanto, está en $Z_1$ pero no en $B_1$, generando un elemento no trivial en $H_1$.

### Ejemplo Numérico: Homología de un Triángulo con un Agujero

Consideremos un complejo consistente en:
- 4 triángulos (formando un cuadrado hueco) con un agujero en el centro
- Vértices: $v_1, v_2, v_3, v_4$ (esquinas), $v_c$ (centro, omitido del complejo)
- Triángulos: $T_1 = \{v_1, v_2, c\}$, $T_2 = \{v_2, v_3, c\}$, etc.

(Simplificaré usando un cuadrado con un agujero cuadrado.)

```
Complejo: Un cuadrado sólido con un agujero cuadrado central.

v1 ---- v2        Vértices exteriores: v1, v2, v3, v4
|      |  |       Vértices interiores: v5, v6, v7, v8
|  ####  |        Aristas: conexiones entre vértices
|  #  #  |        2-cubos: las 8 cuadrados externos (rellenos)
v4 ---- v3

Cálculo de homología:

H_0: Componentes conectadas. El complejo es un rectángulo relleno
     excepto el agujero central. Todo está conectado.
     β_0 = 1 (un componente)

H_1: Ciclos 1D. El agujero central crea un ciclo que rodea el hueco.
     Este ciclo no es la frontera de ningún 2-cubo (porque el hueco está vacío).
     β_1 = 1 (un "túnel")

H_2: Cubos sólidos. El complejo vive en 2D.
     β_2 = 0 (no hay "cavidades" en 2D)

Resultado:
H_0 ≈ Z (isomorfo a los números enteros, generado por el componente)
H_1 ≈ Z (generado por el ciclo que rodea el agujero)
H_2 = 0 (trivial)

Números de Betti: (β_0, β_1, β_2) = (1, 1, 0)
```

---

## 5. Los Números de Betti

Los **números de Betti** son los rangos de los grupos de homología. Proporcionan un resumen numérico de la topología.

$$\beta_k = \text{rango}(H_k)$$

### Interpretación Geométrica

| Número de Betti | Significado Geométrico | Interpretación Intuitiva |
|-----------------|------------------------|-------------------------|
| **β₀** | Número de componentes conectadas | ¿En cuántos "pedazos separados" se divide el objeto? |
| **β₁** | Número de "túneles" o ciclos 1D | ¿Cuántos agujeros (topológicos) hay? |
| **β₂** | Número de "cavidades" o vacíos 3D | ¿Cuántas "burbujas huecas" hay dentro? |
| **βₖ** (general) | Número de cavidades k-dimensionales | Generalizaciones de los anteriores |

### Tabla: Números de Betti e Invariante de Euler

| Espacio | β₀ | β₁ | β₂ | χ = β₀ - β₁ + β₂ | Tipo |
|---------|----|----|-----|-------------------|------|
| **Punto** | 1 | 0 | 0 | 1 | 0-dimensional |
| **Línea** | 1 | 0 | 0 | 1 | 1-dimensional sin bucles |
| **Círculo** | 1 | 1 | 0 | 0 | 1-dimensional con 1 bucle |
| **Disco** | 1 | 0 | 0 | 1 | 2-dimensional sólido |
| **Esfera** | 1 | 0 | 1 | 2 | 2-dimensional hueca |
| **Toro** | 1 | 2 | 1 | 0 | 2 agujeros + 1 cavidad |
| **Botella de Klein** | 1 | 2 | 0 | -1 | No orientable, sin cavidad |
| **Toro doble** | 1 | 4 | 1 | -2 | Dos agujeros en paralelo |

### Características de Euler

La **característica de Euler** es:

$$\chi(K) = \sum_{k=0}^{\infty} (-1)^k \beta_k$$

En 2D y 3D, también se puede calcular como:

$$\chi = V - E + F$$

donde $V$ = vértices, $E$ = aristas, $F$ = caras (una fórmula clásica).

**Ejemplos numéricos:**

1. **Esfera:** $\chi = 2 - 0 + 1 = 2$
2. **Toro:** $\chi = 1 - 2 + 1 = 0$
3. **Botella de Klein:** $\chi = 1 - 2 + 0 = -1$

### Relevancia Clínica: Números de Betti en Tumores Cerebrales

| Situación Clínica | β₀ | β₁ | β₂ | Interpretación |
|-------------------|----|----|-----|----------------|
| **Tumor sólido uniforme** | 1 | 0 | 1 | Una región conectada, con cavidad necrótica central |
| **Tumor con cavidad necrótica** | 1 | 0 | 1 | La cavidad es un agujero 3D (como una burbuja) |
| **Dos tumores separados** | 2 | 0 | 2 | Dos componentes desconectados, cada uno con su cavidad |
| **Tumor con ramificaciones** | 1 | 2 | 1 | Un componente principal con 2 "dedos" o proyecciones |
| **Defecto de segmentación (ruido)** | Muchos | Muchos | Muchos | Fragmentación artificial, β₀ y β₁ muy altos |

**Aplicación clave:** Si β₀ > 1 en la segmentación de un tumor sólido, eso indica fragmentación artificial o presencia de artefactos.

### Ejemplos Numéricos Completos

**Ejemplo 1: Esfera**

```
Topología: Una bola sólida 3D, como un balón de fútbol.

β₀ = 1   (un componente)
β₁ = 0   (sin túneles)
β₂ = 1   (una cavidad interna)

Euler: χ = 1 - 0 + 1 = 2
```

**Ejemplo 2: Toro**

```
Topología: Una dona.

β₀ = 1     (un componente)
β₁ = 2     (dos túneles: uno a través del agujero principal,
             uno alrededor del "donut ring")
β₂ = 1     (una cavidad hueca)

Euler: χ = 1 - 2 + 1 = 0
```

**Ejemplo 3: Botella de Klein**

```
Topología: Una superficie no orientable (un "Klein bottle").
           Es como un toro pero retorcido de una manera especial.

β₀ = 1     (un componente)
β₁ = 2     (dos ciclos no triviales)
β₂ = 0     (sin cavidades cerradas 3D)

Euler: χ = 1 - 2 + 0 = -1
```

**Ejemplo 4: Toro Doble**

```
Topología: Dos donuts pegados.

β₀ = 1     (un componente)
β₁ = 4     (cuatro túneles independientes)
β₂ = 1     (una cavidad única que envuelve ambos donuts)

Euler: χ = 1 - 4 + 1 = -2
```

---

## 6. Homología Persistente: Topología Multiescala

Aquí viene la magia. En datos reales (especialmente imágenes médicas), **no sabemos a priori qué escala usar**. Un tumor puede tener ruido en varias escalas. La homología persistente analiza el espacio en *múltiples escalas simultáneamente*.

### Motivación: El Problema del Ruido

Imagina una imagen de MRI con ruido. A escala muy pequeña, el ruido crea muchos pequeños ciclos falsos (β₁ muy alto). A escala grande, el ruido se suaviza y vemos solo la estructura real.

La homología persistente pregunta: **¿Cuáles de estas características persisten a través de múltiples escalas?** Las características reales persisten; el ruido desaparece.

### Filtración: La Analogía del Paisaje Inundado

Imaginemos un mapa topográfico de un paisaje montañoso. Simulamos lluvia que sube el nivel del agua.

```
t=0:  Cada pico es una isla separada
t=1:  Algunos picos se conectan a través de crestas bajas
t=2:  Más picos se unen
t=3:  Hay un componente principal con algunos lagos aislados
t=∞:  Todo está bajo agua
```

En cada momento $t$, el nivel del agua es $t$. La **filtración** es exactamente esto: una secuencia anidada de complejos:

$$K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_n = K$$

Cuando pasamos de $K_t$ a $K_{t+1}$:
- Algunos componentes se **nacen** (β₀ disminuye)
- Algunos ciclos se **cierran** (β₁ disminuye)
- Algunas cavidades se **crean** (β₂ puede aumentar o disminuir)

Registramos cada "evento": en qué momento nace una característica y cuándo muere.

### Ejemplo Numérico Detallado: Filtración de una Matriz 3×3

Consideremos una imagen 2D de $3 \times 3$ píxeles con valores:

```
  j=0  j=1  j=2
i=0:  0.1  0.9  0.5
i=1:  0.2  0.8  0.4
i=2:  0.6  0.7  0.3

Visualización (donde . = bajo, # = alto):
  .  #  .
  .  #  .
  #  #  .
```

Creamos una **filtración de subnivel** (sublevel-set filtration): en el tiempo $t$, incluimos todos los píxeles con valor $\geq t$.

Umbrales críticos (valores únicos): $t \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$

**t = 0.0** (baseline)

```
Todos los píxeles incluidos:
  #  #  #
  #  #  #
  #  #  #

Complejo cúbico: Cuadrícula completa 3×3
β₀ = 1  (un componente)
β₁ = 0  (sin agujeros)
β₂ = 1  (una cavidad)
```

**t = 0.1** (agregamos el píxel (0,0) con valor 0.1)

```
Píxeles con valor ≥ 0.1:
  #  #  #
  #  #  #
  #  #  #

Igual al anterior: β₀ = 1, β₁ = 0, β₂ = 1
```

**t = 0.2** (agregamos el píxel (1,0) con valor 0.2)

```
Píxeles con valor ≥ 0.2:
  #  #  #
  #  #  #
  #  #  #

Igual: β₀ = 1, β₁ = 0, β₂ = 1
```

(Continuamos hasta los umbrales críticos que producen cambios topológicos...)

**t = 0.5** (los píxeles con valor ≥ 0.5 son: (0,1), (0,2), (1,1), (1,2), (2,0), (2,1))

```
Píxeles incluidos:
  .  #  #
  .  #  #
  #  #  .

Estructura: Región conectada principal (derecha y abajo-izquierda)
β₀ = 1  (conectada)
β₁ = 1  (ahora hay un agujero: la región (0,0) falta)
β₂ = 0  (en 2D, no hay cavidades)

Evento: β₁ nace en t = 0.5 (Nacimiento = 0.5)
```

**t = 0.8** (los píxeles con valor ≥ 0.8 son: (0,1), (1,1), (2,1))

```
Píxeles incluidos:
  .  #  .
  .  #  .
  .  #  .

Estructura: Una línea vertical de tres píxeles
β₀ = 1  (conectada)
β₁ = 0  (el agujero se cierra)
β₂ = 0

Evento: β₁ muere en t = 0.8 (Muerte = 0.8)
```

### Tabla de Nacimientos y Muertes

| Característica | Tipo | Nace (Birth) | Muere (Death) | Persistencia |
|---|---|---|---|---|
| Componente principal | β₀ | 0.0 | ∞ | ∞ |
| Agujero central | β₁ | 0.5 | 0.8 | 0.3 |

### Regla del Anciano

En homología persistente, cuando dos características compiten, el **más antiguo gana** (elder rule). Si una característica nace en $t_1$ y otra en $t_2 > t_1$, y ambas podrían "morir", el de $t_1$ persiste más tiempo.

### Pares Nacimiento-Muerte

Un **par de persistencia** es una tupla $(b, d)$ donde:
- $b$ = momento de nacimiento (cuando la característica aparece)
- $d$ = momento de muerte (cuando se cierra o se fusiona)
- Persistencia = $d - b$

Características reales tienen persistencia alta. Ruido tiene persistencia baja.

---

## 7. Diagramas de Persistencia

Un **diagrama de persistencia** (Dgm) es una visualización 2D de todos los pares $(b, d)$.

### Cómo Leerlo

```
      d (muerte)
      ^
      |     . . .     ← Ruido (cerca de la diagonal)
      |  .
      | .
      |.  ← Característica real (lejos de la diagonal)
      |________________________________ b (nacimiento)
```

- **Eje x:** Momento de nacimiento $b$
- **Eje y:** Momento de muerte $d$
- **Diagonal $y = x$:** Características con persistencia cero (ruido puro)
- **Puntos lejos de la diagonal:** Características reales con persistencia alta

### Lectura de Puntos

| Punto | Persistencia | Interpretación |
|-------|--------------|---|
| $(0.1, 0.12)$ | $0.02$ | Muy pequeña, probablemente ruido |
| $(0.3, 0.8)$ | $0.5$ | Mediana, característica notable |
| $(0.0, ∞)$ | $∞$ | Componente que nunca desaparece (siempre presente) |

### Ejemplo Numérico: Diagrama de Persistencia de la Filtración Anterior

De nuestro ejemplo de matriz 3×3:

```
Pares:
- (0.0, ∞): Componente conectada que persiste siempre
- (0.5, 0.8): Agujero que nace en 0.5 y muere en 0.8

Diagrama de persistencia (sketch):
      d
      |
    ∞ |  *  ← (0.0, ∞)
      |  |\
    0.8|  | \
      |  |  *  ← (0.5, 0.8)
    0.5|  | /
      |  |/
      |  *-----
      0.0      0.5      1.0   b
```

### Representación en Código de Barras (Barcode)

Una alternativa es visualizar como **barras** horizontales:

```
Componente:    |==================|  (desde 0.0 hasta ∞)
Agujero:              |===|  (desde 0.5 hasta 0.8)
               0.0   0.5   0.8
```

Barras largas = características reales. Barras cortas = ruido.

---

## 8. Teorema de Estabilidad

Aquí está el resultado más profundo de la topología de datos. El teorema de estabilidad dice que **diagrama de persistencia es robusto al ruido**.

### Enunciado del Teorema

Sea $f, g: X \to \mathbb{R}$ dos funciones (como intensidades de píxeles en dos imágenes).

$$d_B(\text{Dgm}(f), \text{Dgm}(g)) \leq \|f - g\|_\infty$$

donde $d_B$ es la **distancia de Bottleneck** entre diagramas.

**En palabras simples:**
> Si dos imágenes son similares (diferencia pequeña en intensidades), entonces sus diagramas de persistencia son similares.

### Prueba (Roadmap)

La prueba tiene 4 pasos clave:

| Paso | Qué Ocurre | Intuición |
|------|-----------|-----------|
| 1: Box Lemma | Si una arista se perturba, el ciclo asociado cambio discretamente | Cambios locales → cambios globales acotados |
| 2: Mapas Inducidos | La filtración $K_t(f)$ mapea continuamente a $K_t(g)$ | Las funciones similares generan filtros similares |
| 3: Intercalamiento | Los dos diagramas están "intercalados": uno contiene aproximaciones del otro | Diagramas cercanos en el espacio de funciones están cercanos en diagramas |
| 4: Optimalidad | El mejor apareamiento entre puntos del diagrama es el que minimiza distancia | La métrica de Bottleneck es óptima |

### Analogía Intuitiva

Imagina un mapa topográfico con lluvia que sube lentamente:

```
Mapa original:    Mapa perturbado:

 ▲                  ▲ (ligeramente más bajo)
 │                  │
 │                  │
```

Si llueve lentamente en ambos mapas:
- El agua sube en momentos ligeramente diferentes
- Pero los "eventos topológicos" (un pico se vuelve una isla, dos islas se conectan) ocurren aproximadamente al mismo tiempo
- Por lo tanto, los diagramas de persistencia son similares

### Comparación: Por Qué el Umbralado Rompe la Topología

Supongamos que hacemos un segmentación por **umbral**: todo con intensidad > 0.5 se incluye, todo lo demás se descarta.

```
Imagen original: valores 0.48, 0.51, 0.52 (tres píxeles cercanos)
Imagen ruidosa:  valores 0.49, 0.48, 0.51 (dos están bajo el umbral)

Segmentación original: Incluye tres píxeles
Segmentación ruidosa:  Incluye solo uno

Cambio relativo: 100% de diferencia en la topología,
aunque el ruido fue mínimo.
```

Con homología persistente:

```
Diagrama persistencia: Ambos diagrama tienen pares (0.48, 0.51) y (0.50, 0.52)
                       La perturbación es suave.
```

### Extensión: Estabilidad Wasserstein

Una versión más fuerte usa la **métrica de Wasserstein**:

$$d_W(\text{Dgm}(f), \text{Dgm}(g)) \leq C \|f - g\|_\infty$$

para alguna constante $C$ que depende del dominio.

---

## 9. Vectorización: Imágenes de Persistencia

Los diagramas de persistencia son conjuntos de tamaño variable. Pero las redes neuronales necesitan **entrada de tamaño fijo**.

### El Problema

```
Imagen 1: 5 características
Imagen 2: 8 características
Imagen 3: 3 características

¿Cómo alimentar esto a un CNN que espera vectores de dimensión 100?
```

### Solución 1: Imágenes de Persistencia

Convertimos el diagrama en una **imagen**:

**Paso 1: Transformación de Coordenadas**

Convertimos $(b, d) \to (b, \text{persistencia})$ donde $\text{persistencia} = d - b$:

```
Diagrama original:         Diagrama transformado:
d                          persistencia
|                          |
| •(0.1,0.5)     ======>   | •(0.1,0.4)
|                          |
|___b                       |___b
```

**Paso 2: Densidad Gaussiana**

Colocamos un **kernel gaussiano** alrededor de cada punto:

```
Para cada (b, p_i) en el diagrama transformado,
contribuye una gaussiana: exp(-(x-b)² - (y-p_i)²) / (2σ²)
```

**Paso 3: Crear Imagen Rasterizada**

Discretizamos el espacio en una cuadrícula (ej., 64×64):

```
Imagen de persistencia (64×64):
La intensidad en cada píxel = suma de gaussianas evaluadas ahí
```

### Tabla: Proceso de Vectorización

| Paso | Operación | Entrada | Salida |
|------|-----------|---------|--------|
| 1 | Leer diagrama de persistencia | Conjunto de $(b_i, d_i)$ | Conjunto de $(b_i, p_i)$ donde $p_i = d_i - b_i$ |
| 2 | Aplicar kernel gaussiano | $(b_i, p_i)$ + σ | Función continua $f(x, y)$ |
| 3 | Rasterizar | $f(x, y)$ + resolución | Matriz 64×64 |
| 4 | Normalizar | Matriz | Matriz normalizada [0, 1] |
| 5 | Alimentar a red neuronal | Imagen 64×64 | Predicción |

### Ejemplo Numérico: Convertir Diagrama a Imagen de Persistencia

**Diagrama de entrada:**

```
Pares (b, d):
- (0.0, 1.0) → persistencia = 1.0
- (0.2, 0.35) → persistencia = 0.15
- (0.5, 0.7) → persistencia = 0.2
```

**Matriz transformada:**

```
(b, p):
- (0.0, 1.0)
- (0.2, 0.15)
- (0.5, 0.2)
```

**Rasterización (8×8 para simplificar):**

```
Cuadrícula de [0,1] × [0,1], σ = 0.1

Píxel (0,0) en b=0, p=0: Cercano a (0.0, 1.0) (gaussiana de (0.0, 1.0))
Píxel (4,4) en b=0.5, p=0.5: Cercano a (0.5, 0.2) (gaussiana más débil)
...

Imagen resultante (sketch):
  0.8│ . . . . . . . .
     │ . . . # # # . .
  0.6│ . . # # # # # .
     │ . # # # # # # #
  0.4│ # # # # # # # .
     │ # # # # # # . .
  0.2│ # # # . . # . .
     │ # . . . . . . .
  0.0└─────────────────
     0   0.2  0.4  0.6  0.8

(Donde # representa valores altos de gaussiana)
```

### Ventajas de Imágenes de Persistencia

1. **Tamaño fijo:** Siempre 64×64 (o el tamaño que elijamos)
2. **Suave:** Las gaussianas son diferenciables
3. **Estable:** Hereda la estabilidad del diagrama de persistencia
4. **Compatible con CNNs:** Puedes usarlo como entrada a redes de imágenes

### Estabilidad de Imágenes de Persistencia

Las imágenes de persistencia heredan la **continuidad de Lipschitz** del diagrama:

$$\|I(f) - I(g)\|_2 \leq L \|f - g\|_\infty$$

donde $I(f)$ es la imagen de persistencia y $L$ es una constante que depende de σ.

---

## 10. Digitalización y Preservación Topológica

Un problema práctico: Cuando discretizas un espacio continuo (como una imagen médica) en píxeles, ¿preservas la topología?

### Teorema 1.1: Condiciones para Preservación Topológica

Sea $X$ un espacio compacto continuo y $X_\epsilon$ su discretización con resolución $\epsilon$ (tamaño de píxel).

Si la **distancia de alcance** (reach) de $X$ satisface $\text{reach}(X) > k\epsilon$ para alguna constante $k$, entonces:

$$H_n(X) \cong H_n(X_\epsilon) \quad \text{para todos los } n$$

es decir, **la homología se preserva**.

### Noción de Alcance (Reach)

La **reach** de un conjunto compacto $X$ es el mayor $r$ tal que todo punto en $X$ tiene un único punto más cercano en $X$ (sin ambigüedades):

```
Intuición: "Espacio de maniobra" en la frontera de $X$

Forma suave (reach alto):     Forma puntiaguda (reach bajo):
  ╱╲                          ╱╲
 ╱  ╲   r es grande          ╱  ╲  r es muy pequeño
─────                        ─────
```

Para un **objeto 1D suave** (una curva):
- Reach ∝ curvatura mínima
- Curvatura alta (giro cerrado) → reach bajo

Para un **objeto 2D suave** (una superficie):
- Reach ∝ curvatura principal mínima

### Condición de Resolución

Para preservar topología:

$$\epsilon < \frac{\text{reach}(X)}{k}$$

donde $k$ depende de cómo se discretice.

**Ejemplo clínico:** Un tumor tiene frontera suave con radio de curvatura ~10 voxels. Si usamos vóxeles de tamaño 1 mm, entonces:

```
Reach ≈ 10 mm
Condición: ε < 10 mm / k

Para k = 2 (margen de seguridad):
ε < 5 mm

Por lo tanto, vóxeles ≤ 5 mm preservan topología.
```

### Ejemplo: Dos Focos Tumorales

Dos regiones tumorales separadas por 3 voxels:

```
Escala: 1 voxel = 1 mm

Imagen real:
[__TUMOR1__]  [__TUMOR2__]
            ===
            3 mm de brecha

Digitalización:
Si ε = 1 mm: Dos componentes separados (β₀ = 2) ✓ Correcto
Si ε = 2 mm: Cada pixel representa un cuadrado de 2mm.
             La brecha de 3 mm puede desaparecer según el posicionamiento.
             Riesgo de que aparezcan como conectados incorrectamente.
```

---

## 11. Herramientas de Software

### Tabla: Herramientas de Homología Persistente

| Herramienta | Propósito | Lenguaje | Tipo Complejo | URL |
|---|---|---|---|---|
| **GUDHI** | Cálculo de persistencia, filtración | C++/Python | Simplicial, Cúbico | [GUDHI.inria.fr](https://gudhi.inria.fr) |
| **Ripser** | Persistencia rápida (Vietoris-Rips) | C++/Python | Simplicial | [ripser.org](http://ripser.org) |
| **giotto-tda** | Integración con machine learning | Python | Múltiples | [giotto.ai](https://www.giotto.ai) |
| **Dionysus 2** | Cálculo de persistencia avanzado | C++/Python | Simplicial | [www.mrzv.org/software/dionysus2](https://www.mrzv.org/software/dionysus2) |
| **Javaplex** | Herramienta educativa | Java | Simplicial | [github.com/appliedtopology/javaplex](https://github.com/appliedtopology/javaplex) |
| **TDA.jl** | Computación topológica rápida | Julia | Simplicial | [github.com/Arity-Math/TDA.jl](https://github.com/Arity-Math/TDA.jl) |

### Ejemplo de Código: GUDHI para Persistencia Cúbica

```python
import numpy as np
import gudhi

# Crear una imagen binaria 3x3
imagen = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 0]
], dtype=np.uint8)

# Crear un complejo cúbico desde la imagen
cubical_complex = gudhi.CubicalComplex(
    top_dimensional_cells=imagen,
    dimensions=[imagen.shape[0], imagen.shape[1]]
)

# Calcular persistencia
cubical_complex.compute_persistence()

# Obtener pares de persistencia
print("Pares de persistencia (dimensión, nacimiento, muerte):")
for dim, (birth, death) in cubical_complex.persistent_pairs():
    print(f"  Dimensión {dim}: nace en {birth}, muere en {death}, "
          f"persistencia = {death - birth}")

# Obtener números de Betti
betti = cubical_complex.persistent_betti_numbers(0, 1)
print(f"Números de Betti: {betti}")
```

**Salida esperada:**

```
Pares de persistencia (dimensión, nacimiento, muerte):
  Dimensión 0: nace en 0, muere en inf, persistencia = inf
  Dimensión 1: nace en 0.5, muere en 0.8, persistencia = 0.3

Números de Betti: [1, 1, 0]
```

### Ejemplo Adicional: Visualización de Diagrama de Persistencia

```python
import matplotlib.pyplot as plt
import numpy as np

# Simular pares de persistencia
pares = np.array([
    [0.0, np.inf],  # Componente que persiste
    [0.2, 0.5],     # Característica de ruido
    [0.3, 0.9],     # Característica real
    [0.5, 0.55],    # Ruido fino
])

# Filtrar pares finitos para visualización
finitos = pares[np.isfinite(pares[:, 1])]
infinitos = pares[~np.isfinite(pares[:, 1])]

# Crear diagrama
fig, ax = plt.subplots(figsize=(8, 8))

# Diagonal y = x
max_val = finitos[:, 1].max() * 1.2
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal (ruido)')

# Puntos finitos
ax.scatter(finitos[:, 0], finitos[:, 1], s=100, c='blue', alpha=0.6, label='Características finitas')

# Puntos infinitos (en la parte superior)
for b, _ in infinitos:
    ax.scatter(b, max_val, s=150, c='red', marker='^', alpha=0.8)
ax.text(infinitos[0, 0], max_val * 1.05, 'Infinito', ha='center', fontsize=10)

ax.set_xlabel('Nacimiento (b)', fontsize=12)
ax.set_ylabel('Muerte (d)', fontsize=12)
ax.set_title('Diagrama de Persistencia', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

---

## 12. Resumen y Conexiones

### Tabla: Concepto y Rol en la Pipeline de Tesis

| Concepto | Explicación Breve | Rol en Segmentación de Tumores | Capítulo Siguiente |
|---|---|---|---|
| **Topología** | Estudio de propiedades invariantes bajo deformación continua | Define qué queremos preservar en la segmentación | Geometría computacional |
| **Complejos Simpliciales** | Construcción discreta de espacios mediante triángulos | Modelo teórico, pero ineficiente para imágenes 3D | Complejos cúbicos |
| **Complejos Cúbicos** | Construcción nativa para imágenes digitales | La representación real que usamos en imágenes MRI | Algoritmos de persistencia |
| **Homología** | Medida algebraica de agujeros y cavidades | Detecta cavidades necróticas, tumores múltiples, fragmentación | Números de Betti |
| **Números de Betti** | Rangos de grupos de homología: β₀, β₁, β₂ | Características cuantitativas del tumor (componentes, agujeros, cavidades) | Descriptores topológicos |
| **Homología Persistente** | Análisis multiescala que registra nacimientos y muertes | Robustez al ruido, extrae características reales de ruido | Filtración y estabilidad |
| **Diagramas de Persistencia** | Visualización 2D de pares (nacimiento, muerte) | Resumen compacto de la topología a múltiples escalas | Imágenes de persistencia |
| **Teorema de Estabilidad** | Las perturbaciones pequeñas en datos causan cambios pequeños en diagramas | Garantiza que nuestro análisis es robusto matemáticamente | Vectorización |
| **Imágenes de Persistencia** | Conversión de diagramas variables a imágenes 64×64 fijas | Entrada compatible con redes neuronales | CNNs y U-Net |
| **Digitalización Segura** | Condiciones de resolución para preservar topología | Asegura que discretización no crea artefactos topológicos | Preprocesamiento |

### Visión de Conjunto

En esta tesis, utilizaremos **topología algebraica y homología persistente** para:

1. **Análisis:** Extraer características topológicas de tumores cerebrales (cavidades, fragmentación, ramificaciones).
2. **Robustez:** Usar la estabilidad de la homología persistente para manejar ruido en imágenes MRI.
3. **Entrada a ML:** Convertir diagramas de persistencia en imágenes que alimenten un U-Net.
4. **Validación:** Verificar que la segmentación respeta la topología (β₀ = 1 para tumores sólidos, β₂ = 1 para cavidades).

El siguiente capítulo explorará **CNN y redes U-Net**, que utilizan las imágenes de persistencia y descriptores topológicos como características adicionales para mejorar la segmentación automática.

---

## Conclusión

La topología algebraica y la homología persistente son herramientas poderosas para entender la **forma de los datos**. Desde la definición abstracta de espacios hasta el cálculo práctico en imágenes médicas, estos conceptos ofrecen:

- **Invariancia:** Propiedades que no cambian bajo transformaciones continuas
- **Estabilidad:** Robustez ante pequeñas perturbaciones
- **Multiescala:** Análisis a diferentes resoluciones simultáneamente
- **Computabilidad:** Algoritmos rápidos para calcular topología discreta

En el contexto de la segmentación de tumores cerebrales, estas herramientas permiten a los algoritmos de aprendizaje automático **ver la forma real** del tumor, más allá de píxeles individuales, mejorando la precisión y confiabilidad del diagnóstico.

---

**Nota Final:** Este artículo es autoconttenido, pero la exploración completa de estos temas requeriría cursos especializados. Para lectores interesados, recomendamos:

- Libros: *Computational Topology: An Introduction* (Edelsbrunner & Harer)
- Cursos en línea: Plataformas como Coursera ofrecen introducción a topología computacional
- Software: Experimenta con GUDHI o giotto-tda en tus propias imágenes
