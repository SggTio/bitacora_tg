# Razonamiento para Visualizaciones HTML Interactivas

## Filosofía General

Las visualizaciones interactivas complementan los posts de blog proporcionando experiencias pedagógicas dinámicas que permiten a los lectores:

- **Experimentar**: modificar parámetros y observar cambios en tiempo real
- **Intuir**: visualizar conceptos abstractos (topología, convoluciones, filtración)
- **Validar**: comparar resultados matemáticos con representaciones gráficas
- **Transferir**: aprender principios que se aplican a problemas reales (segmentación de órganos)

### Principios de Diseño

- **Autonomía**: Cada archivo HTML es auto-contenido (HTML + CSS + JS inline)
- **Accesibilidad**: CDN públicos (D3.js, Three.js, Plotly.js, Chart.js, p5.js)
- **Responsividad**: Funcionan en desktop y móvil
- **Claridad**: Etiquetas en español, código comentado en español
- **Encapsulación**: Embebibles en Jekyll markdown vía `<iframe>` o `<script src="...">`

---

## Documento 1: Teoría de Convoluciones

### 1.1 Convolución 1D Interactiva

**Concepto a visualizar**: Operación de convolución discreta (g * f)[n] = Σ g[m] f[n - m] mediante ventana deslizante

**Tipo de interacción**:
- Slider horizontal controla posición de la ventana
- Visualización lado-a-lado: señal original + kernel + resultado
- Tooltip muestra cálculo en tiempo real: multiplicación elemento-a-elemento y suma

**Descripción técnica**:
- Lienzo SVG con tres gráficas: f(x) en azul, kernel g en rojo, convolución resultado en verde
- Animación de la ventana deslizante con rectángulo de selección
- Tabla dinámica mostrando valores multiplicados y suma acumulada
- Botones para usar diferentes kernels (promediador, diferenciador, Sobel 1D)

**Prompt detallado para Claude Code**:
```
Crea un archivo HTML interactivo que demuestre la convolución 1D discreta.
Incluye:
1. Una señal f[n] discreta (ej: [1,2,3,4,5,3,2,1]) graficada como barras azules
2. Un kernel g[n] (ej: [0.25, 0.5, 0.25]) como barras rojas superpuestas
3. Un slider que controla el índice de la ventana deslizante (rango 0 a length(f)-length(g))
4. Al mover el slider, dibuja un rectángulo que rodea la ventana actual
5. Calcula y muestra en una tabla: multiplicaciones elemento-a-elemento, suma parciales, resultado final
6. Gráfica adicional abajo mostrando la convolución completa en verde (acumulada)
7. Botones para cambiar kernel: "Promediador", "Diferenciador", "Suavizador Gaussiano"
8. Velocidad de animación ajustable
Usa Canvas o SVG. Mantén es­pacios en blanco para claridad. Etiquetas en español.
```

**Librería**: Canvas/SVG vanilla JavaScript
**Complejidad**: Media

---

### 1.2 Teorema de Convolución de Fourier

**Concepto a visualizar**: Convolución en dominio espacial = multiplicación en dominio de frecuencia (FFT)

**Tipo de interacción**:
- Usuario dibuja una señal 1D (o selecciona preset)
- Selector para elegir kernel
- Visualización cuádruple: señal original + FFT (magnitud) | señal convolucrada + FFT (magnitud)
- Parámetros: escala de frecuencias, rango de amplitud

**Descripción técnica**:
- Entrada: usuario dibuja puntos en un canvas (o carga signal preset)
- FFT usando bibliotéca JavaScript (ej: Goertzel o FFT.js)
- Dos columnas: izquierda (entrada), derecha (convolución)
- Cada una con dos filas: dominio espacial y magnitud de frecuencia
- Líneas discontinuas indican correspondencia visual
- Escala logarítmica opcional para magnitudes

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva del Teorema de Convolución de Fourier.
Incluye:
1. Cuatro gráficas en disposición 2x2:
   - (Arriba-Izq) Señal original f(x) en dominio espacial (Canvas o SVG, max 256 puntos)
   - (Arriba-Der) Magnitud de FFT{f(x)} en escala lineal o logarítmica
   - (Abajo-Izq) Convolución f(x) * g(x) en dominio espacial
   - (Abajo-Der) Magnitud de FFT{f(x) * g(x)} = FFT{f} · FFT{g}
2. Botones/radio buttons para presets de f: rampa, gaussiana, rectangular, senoidal
3. Botones para kernel g: blur (Hann), diferenciador, sharpen
4. Slider para escala logarítmica en frecuencia (checkbox)
5. Usa FFT.js o implementa FFT simple (Cooley-Tukey para 256 puntos)
6. Líneas punteadas conectan gráficas correspondientes para guiar la lectura
7. Tooltip: al pasar mouse, muestra valor en ambos dominios
8. Leyenda clara: "Dominio Espacial" vs "Dominio de Frecuencia"
Nota: demostración clara de equivalencia espacial=frecuencia.
```

**Librería**: FFT.js (CDN), Canvas, SVG
**Complejidad**: Alta

---

### 1.3 Ecuación del Calor / Difusión Gaussiana

**Concepto a visualizar**: Convolución con gaussiana en tiempo = solución de ecuación del calor en 2D

**Tipo de interacción**:
- Usuario especifica condición inicial (mapa de calor 2D o dibuja puntos calientes)
- Sliders para: tiempo (de 0 a T), parámetro σ de la gaussiana
- Animación en tiempo real del campo de calor
- Mapa de color: azul (frío) → rojo (caliente)

**Descripción técnica**:
- Grid 2D (ej: 64×64 píxeles)
- Convolución 2D con kernel gaussiano G(σ) = exp(-(x²+y²)/(2σ²)) / (2πσ²)
- Canvas principal mostrando temperatura como mapa de color
- Slider temporal avanza mediante repetidas convoluciones (∂u/∂t ≈ Δu)
- Perfil 1D adicional: corte vertical del campo 2D actual
- Opción: "Pulso instantáneo" vs "Fuente continua"

**Prompt detallado para Claude Code**:
```
Crea una visualización HTML5 Canvas interactiva de la ecuación del calor 2D.
Incluye:
1. Canvas principal (400×400px mínimo) mostrando campo de temperatura como mapa de color
   - Azul oscuro = 0K (frío), rojo = máxima temperatura
2. Inicializaciones predefinidas:
   - "Punto central": calor concentrado en centro
   - "Doble punto": dos focos de calor
   - "Línea vertical": calor distribuido en columna central
   - "Dibuja tú": usuario hace click para agregar calor en mouse position
3. Slider de tiempo (0 a 100 pasos) que ejecuta pasos de difusión
   - Cada paso: convoluciona con kernel gaussiano 5×5 (aproxima ∂u/∂t = Δu)
4. Slider de parámetro σ (0.5 a 3.0) ajusta ancho de gaussiana (velocidad de difusión)
5. Botón "Animar" que ejecuta auto-stepping
6. Gráfica 1D adicional (Chart.js): corte horizontal/vertical del campo, actualiza con cada paso
7. Display numérico: temperatura máxima, temperatura media, tiempo simulado
8. Paleta de colores (viridis o similar) dibujada como leyenda lateral
Objetivo: intuir que convoluciones suavizan superficies (difusión).
```

**Librería**: Canvas HTML5, Chart.js (para gráfica 1D)
**Complejidad**: Alta

---

### 1.4 Convolución 2D Deslizante (CNN-style)

**Concepto a visualizar**: Operación de convolución 2D kernel-over-image en una capa CNN (forward pass)

**Tipo de interacción**:
- Imagen pequeña (ej: 5×5 o 7×7) mostrada como grid
- Usuario elige kernel (edge detection, blur, sharpen)
- Botón "Paso siguiente" que desliza kernel una posición
- Visualización: imagen → kernel overlay → valor computado → feature map parcial

**Descripción técnica**:
- Grid 2D (imagen) con valores 0-255 en escala de grises
- Kernel overlay con colores diferenciados (positivos=rojo, negativos=azul)
- Tabla mostrando multiplicaciones y suma final
- Feature map parcial se llena gradualmente (gris para no-procesados, naranja para actuales)
- Stride y padding ajustables
- Opción de animación automática

**Prompt detallado para Claude Code**:
```
Crea una visualización paso-a-paso de convolución 2D (similar a CNNs).
Requisitos:
1. Imagen de entrada 7×7 píxeles (valores 0-255, mostradi como grid con colores de escala de grises)
2. Kernel 3×3 (valores -1 a +1, mostrado con colores rojo/azul)
3. Visualización overlay: kernel posicionado sobre imagen con cuadrante de selección
4. Slider "posición de ventana" (rango: 0 a (7-3)² = 25 posiciones)
5. Para cada posición:
   - Tabla de 3×3 con multiplicaciones (imagen[i,j] × kernel[i,j])
   - Suma total con fondo resaltado
   - Valor añadido a feature map
6. Feature map resultante 5×5 mostrado a la derecha, actualizado dinámicamente
7. Presets de kernel: "Borde Vertical", "Borde Horizontal", "Blur", "Sharpen", "Identidad"
8. Sliders para: Stride (1-2) y Padding (0-1)
9. Display: "Dimensión de salida: 5×5" (formula (N-K+2P)/S + 1)
10. Botón "Autoplay" que anima el deslizamiento
Utiliza Canvas o tabla HTML con estilos. Clarity sobre intuición.
```

**Librería**: Canvas/SVG, Bootstrap/CSS Grid
**Complejidad**: Media-Alta

---

### 1.5 Separabilidad de Filtros (Convolución Depthwise-Separable)

**Concepto a visualizar**: Reducción de parámetros comparando convolución estándar vs separable

**Tipo de interacción**:
- Sliders para: canales de entrada C_in, canales de salida C_out, tamaño de kernel K
- Gráfica dinámica: barras mostrando total de parámetros
- Comparación lado-a-lado de arquitecturas
- Factorización visual

**Descripción técnica**:
- Estándar: C_in × K × K × C_out parámetros
- Separable: C_in × K × K + K × K × C_out parámetros
- Gráfico de barras actualizándose en tiempo real
- Tabla mostrando desglose: pesos convolucionales, pesos pointwise
- Fórmula matemática renderizada (MathJax)
- Comparativa: "Reducción: X%"

**Prompt detallado para Claude Code**:
```
Crea una herramienta interactiva que compare parámetros de convoluciones.
Características:
1. Tres sliders horizontales:
   - "Canales de Entrada (C_in)": 1 a 64 (paso 1)
   - "Canales de Salida (C_out)": 1 a 64 (paso 1)
   - "Tamaño del Kernel (K)": 1 a 7 (paso 2, impares)
2. Dos diagramas lado-a-lado:
   - IZQUIERDA: "Convolución Estándar"
     - Bloque visual mostrando entrada [C_in], kernel [K×K], salida [C_out]
     - Contador: "Parámetros = C_in × K² × C_out"
   - DERECHA: "Convolución Depthwise-Separable"
     - Dos bloques: Depthwise [C_in × K²] + Pointwise [K² × C_out]
     - Contador: "Parámetros = C_in × K² + K² × C_out"
3. Gráfica de barras horizontales comparando total de parámetros
4. Línea roja mostrando "Estándar", línea azul "Separable"
5. Texto dinámico: "Reducción: X%" y "Relación: 1:Y"
6. Fórmula matemática clara usando notación LaTeX (o HTML equivalente)
7. Ejemplos predefinidos: "MobileNet", "SqueezeNet", "ResNet"
Usa Plotly.js para gráficas o Canvas. Enfoque: eficiencia computacional.
```

**Librería**: Plotly.js (o Chart.js), MathJax (LaTeX)
**Complejidad**: Media

---

## Documento 2: Topología Algebraica y Persistencia

### 2.1 Deformaciones Topológicas (Homeomorfismo)

**Concepto a visualizar**: Equivalencia topológica: taza de café ↔ rosquilla (torus)

**Tipo de interacción**:
- Slider continuo de 0 a 1 (parámetro t de deformación)
- Vista 3D rotable (mouse drag)
- Información: número de agujeros conservados

**Descripción técnica**:
- Three.js para renderizado 3D
- Malla parametrizada para torus
- Interpolación lineal de vértices: torus(1-t) + copa(t)
- Materiales: wireframe o shaded
- Orientación rotable con controles de mouse

**Prompt detallado para Claude Code**:
```
Crea una visualización 3D interactiva mostrando homeomorfismo entre rosquilla y taza.
Requisitos:
1. Renderizado 3D con Three.js (cámara perspectiva, iluminación básica)
2. Objeto 3D inicial: torus (dona) con mayor eje R=2, menor eje r=0.8
3. Objeto 3D final: taza (aproximación con cilindro+ asa modelada como torus pequeño)
4. Slider horizontal "Deformación" (0 a 1):
   - En t=0: torus puro
   - En t=1: taza pura
   - 0 < t < 1: interpolación suave (morphing)
5. Rotación automática (puede pausarse)
6. Controles de mouse: click+drag para rotación manual
7. Botones: "Reproducir Animación", "Pausar", "Reset"
8. Información textual:
   - "Número de agujeros: 1 (invariante topológico)"
   - "Esta deformación preserva la topología"
9. Vista wireframe+shaded (toggle)
10. Escena con fondo neutral (gradiente gris)
Enfoque pedagógico: homeomorfismo, invariantes topológicos.
```

**Librería**: Three.js
**Complejidad**: Alta

---

### 2.2 Complejo Simplicial Interactivo (Rips/Delaunay)

**Concepto a visualizar**: Construcción progresiva de complejo simplicial a partir de puntos nube

**Tipo de interacción**:
- Usuario coloca puntos (click en canvas)
- Slider epsilon ajusta radio de conectividad
- Visualización en tiempo real: puntos → aristas → triángulos

**Descripción técnica**:
- Canvas 2D mostrando puntos como círculos
- Slider epsilon (0 a distancia máxima)
- Aristas conectan puntos a distancia < epsilon (colores suaves)
- Triángulos rellenos cuando 3 puntos mutuamente conectados
- Tabla: número de 0-símplices, 1-símplices, 2-símplices
- Botones: "Limpiar", "Ejemplo Predefinido", "Automático"

**Prompt detallado para Claude Code**:
```
Crea una herramienta HTML5 para construir complejos simpliciales (Rips).
Componentes:
1. Canvas principal (600×600px) con fondo gris claro
2. Modo "Agregar Puntos": click coloca puntos azules (radio 5px)
3. Slider "Radio de Conexión (ε)": 10 a 300 píxeles
4. Dinámicamente:
   - Para cada par de puntos con distancia < ε, dibuja arista gris
   - Para cada trío de puntos mutuamente conectados, dibuja triángulo relleno (transparencia 0.3, color cian)
5. Tabla/contadores:
   - "0-símplex (vértices)": número de puntos
   - "1-símplex (aristas)": número de conexiones
   - "2-símplex (triángulos)": número de caras 2D
6. Botones:
   - "Limpiar puntos"
   - "Ejemplo 1: 5 puntos en círculo"
   - "Ejemplo 2: Nube aleatoria (20 puntos)"
7. Animación: cuando cambia ε, actualizar visualmente durante 300ms
8. Opcional: mostrar número de Euler V - E + F
Usa Canvas. Enfoque: topología computacional, complejos.
```

**Librería**: Canvas HTML5, JavaScript vanilla
**Complejidad**: Media

---

### 2.3 Filtración de Subnivel

**Concepto a visualizar**: Aumento gradual del conjunto de subnivel de una función, cambios topológicos

**Tipo de interacción**:
- Slider de altura/threshold (0 a max valor función)
- Visualización 2D: contorno de subnivel + puntos nacimiento/muerte
- Gráfica 1D: corte vertical de función

**Descripción técnica**:
- Función 2D renderizada como mapa de altura (colores por valor)
- Contorno rojo muestra límite subnivel actual
- Marcadores verdes = nacimiento componente, rojos = muerte
- Gráfica lado-a-lado: corte 1D de función, con línea horizontal indicando threshold
- Animación automática opcional

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva de filtración de subnivel.
Especificaciones:
1. Canvas 2D (400×400px) mostrando función f(x,y) = altura en 3D proyectada a mapa de color
   - Colores: azul (bajo) → verde → amarillo → rojo (alto)
   - Función sugerida: f(x,y) = sin(x) + cos(y) + gauss(x,y)
2. Slider "Threshold" (0 a max(f)):
   - Dibuja región donde f(x,y) ≤ threshold como contorno rojo grueso
   - Dentro del contorno, shading más oscuro
3. En el mapa:
   - Puntos verdes = nuevo nacimiento de componente (cuando threshold cruza mínimo local)
   - Puntos rojos = muerte de componente (cuando threshold cruza máximo local)
4. Panel lateral con gráfica 1D:
   - Eje vertical = f, eje horizontal = posición x (o integrada)
   - Línea horizontal azul = threshold actual
   - Marcas de nacimiento/muerte alineadas
5. Tabla: "Componentes activas: N", "Agujeros activos: M"
6. Botón "Animar Filtración" que barre threshold automáticamente
7. Información topológica: β₀ (componentes), β₁ (agujeros) en tiempo real
Enfoque: homología persistente, cambios topológicos en filtración.
```

**Librería**: Canvas, Chart.js (para gráfica 1D)
**Complejidad**: Alta

---

### 2.4 Diagrama de Persistencia Interactivo

**Concepto a visualizar**: Nacimientos y muertes de características topológicas como puntos en plano 2D

**Tipo de interacción**:
- Usuario modifica función subyacente (slider de parámetros)
- Diagrama de persistencia actualiza en tiempo real
- Zoom, hover para ver coordenadas exactas
- Comparación con diagrama diagonal (ruido topológico)

**Descripción técnica**:
- Plano 2D: eje x = tiempo nacimiento, eje y = tiempo muerte
- Cada punto = característica topológica (color según tipo: componente/agujero)
- Distancia a diagonal = persistencia (lifetimee de característica)
- Slider para función parameter → cambios en diagram
- Línea diagonal punteada como referencia
- Zoom y pan interactivos

**Prompt detallado para Claude Code**:
```
Crea un visualizador interactivo de diagrama de persistencia.
Requisitos:
1. Plano cartesiano 2D (400×400px mínimo):
   - Eje X: "Nacimiento (tiempo/altura)"
   - Eje Y: "Muerte (tiempo/altura)"
   - Rango: 0 a 100 en ambos ejes
2. Línea diagonal y=x dibujada punteada (gris claro) - separación ruido
3. Puntos del diagrama mostrados como círculos:
   - Azul = componentes conectadas (H₀)
   - Rojo = agujeros/cavidades (H₁)
   - Tamaño proporcional a persistencia
4. Sliders para modificar función subyacente:
   - Slider "Parámetro de Amplitud" (1 a 5)
   - Slider "Número de Picos" (1 a 5)
   - Cada cambio recalcula la filtración y actualiza diagrama
5. Hover en punto: tooltip con "(nacimiento, muerte, persistencia=muerte-nacimiento)"
6. Zoom: scroll del mouse amplía alrededor del cursor
7. Pan: click+drag desplaza vista
8. Botón "Reset Vista" vuelve a escala inicial
9. Tabla inferior con estadísticas:
   - "Puntos H₀", "Puntos H₁", "Persistencia máxima", "Persistencia promedio"
10. Ejemplo predefinido: función de gaussianas múltiples
Enfoque: compresión topológica, identificación de ruido vs características.
```

**Librería**: D3.js (o Plotly.js con interactive features)
**Complejidad**: Alta

---

### 2.5 Números de Betti en Acción

**Concepto a visualizar**: Conteo de componentes, agujeros, cavidades (β₀, β₁, β₂)

**Tipo de interacción**:
- Usuario dibuja forma en canvas (libre o selecciona presets)
- Cálculo automático de Betti numbers
- Visualización: forma coloreada + números de Betti

**Descripción técnica**:
- Canvas 2D editable (usuario dibuja con mouse)
- Análisis topológico en tiempo real (conectividad, ciclos)
- Tres contadores: β₀ (componentes), β₁ (agujeros 1D), β₂ (cavidades 3D)
- Código pseudo mostrando algoritmo (DFS para componentes, ciclos)
- Presets: círculo, anillo, figura-8, letra con agujeros, etc.

**Prompt detallado para Claude Code**:
```
Crea una herramienta interactiva para visualizar números de Betti.
Características:
1. Canvas principal (600×600px):
   - Modo dibujo: usuario dibuja formas negras sobre fondo blanco
   - Herramientas: pincel (grosor ajustable), borrador, limpiar
2. Análisis topológico automático (cada cambio):
   - β₀ = número de componentes conectadas (1 para forma única)
   - β₁ = número de agujeros topológicos (1 para anillo, 2 para figura-8)
   - β₂ = cavidades 3D (0 en 2D)
   - Fórmula de Euler: χ = β₀ - β₁ + β₂
3. Display de resultados:
   - Tres cuadros grandes mostrando β₀, β₁, β₂ con colores verde/naranja
   - Fórmula de Euler χ = [resultado]
4. Presets (botones):
   - "Círculo" (β₀=1, β₁=0)
   - "Anillo" (β₀=1, β₁=1)
   - "Figura-8" (β₀=1, β₁=2)
   - "Dos círculos" (β₀=2, β₁=0)
   - "Anillo + Círculo" (β₀=2, β₁=1)
5. Panel educativo con pseudocódigo:
   - "Algoritmo para calcular β₀: DFS/BFS sobre píxeles conectados"
   - "Algoritmo para calcular β₁: análisis de ciclos"
6. Tooltip: explicación de qué es cada número de Betti
Usa Canvas HTML5 + algoritmos de análisis de imagen simple (conectividad).
```

**Librería**: Canvas HTML5, JavaScript vanilla
**Complejidad**: Media-Alta

---

### 2.6 Imagen de Persistencia (PI)

**Concepto a visualizar**: Transformación de diagrama de persistencia a imagen fija mediante kernels Gaussianos

**Tipo de interacción**:
- Usuario ajusta parámetro de bandwidth σ
- Slider de resolución de imagen
- Visualización lado-a-lado: diagrama → imagen de persistencia

**Descripción técnica**:
- Diagrama de persistencia como entrada (puntos)
- Kernel Gaussiano 2D aplicado a cada punto
- Imagen 2D resultante con valores intensidad 0-255
- Sliders: σ (ancho), resolución (píxeles)
- Colormap viridis o similar

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva de Imágenes de Persistencia (PI).
Especificaciones:
1. Dos paneles lado-a-lado (cada uno 300×300px):
   - IZQUIERDA: Diagrama de persistencia (puntos)
     - Fondo blanco, ejes etiquetados
     - Puntos azules representan H₀, rojos H₁
     - Diagonal y=x punteada
   - DERECHA: Imagen de Persistencia (mapa de calor)
     - Píxeles coloreados según densidad kernel
     - Colormap: viridis (azul oscuro → amarillo)

2. Sliders de control:
   - "Bandwidth σ": 0.5 a 5.0
     - Controla ancho de kernel Gaussiano 2D
     - Σ pequeña: imagen ruidosa, máximos agudos
     - Σ grande: imagen suave, máximos difusos
   - "Resolución": 32×32 a 256×256 píxeles
   - "Escala de valor (min/max intensidad)": log o lineal

3. Cálculo en tiempo real:
   - Para cada píxel (x,y) en imagen, suma gaussianas 2D centradas en puntos persistencia
   - PI(x,y) = Σ exp(-((x-birth[i])² + (y-death[i])²)/(2σ²))
   - Normaliza a rango 0-255

4. Datos de prueba:
   - Conjunto predefinido de puntos persistencia
   - Botón "Cargar ejemplo" que agrega puntos aleatorios

5. Información:
   - Display numérico: "σ = X.X", "Resolución = Y×Y"
   - Leyenda de colores (escala de valor)

Usa Canvas. Enfoque: representación fija-tamaño para comparación topológica.
```

**Librería**: Canvas HTML5, Plotly.js (colorscale)
**Complejidad**: Media

---

## Documento 3: CNNs y U-Net

### 3.1 Neurona Artificial Interactiva

**Concepto a visualizar**: Cálculo de perceptrón: y = σ(w₁x₁ + w₂x₂ + ... + b)

**Tipo de interacción**:
- Sliders para cada peso w₁, w₂, bias b
- Entradas x₁, x₂ fijas o ajustables
- Visualización: cálculo paso-a-paso, curva de activación

**Descripción técnica**:
- Diagrama esquemático: círculos para entradas, líneas etiquetadas con pesos
- Nodo de suma, luego caja de función de activación
- Líneas de números mostrando valores en tiempo real
- Gráfica de σ(z) actualizada
- Selector para función de activación: sigmoid, tanh, ReLU

**Prompt detallado para Claude Code**:
```
Crea una herramienta interactiva para explorar un perceptrón simple.
Componentes:
1. Diagrama visual de neurona:
   - Dos nodos de entrada (x₁, x₂) a la izquierda
   - Líneas etiquetadas con pesos (w₁, w₂)
   - Nodo suma en el medio
   - Entrada sesgo (b) hacia el nodo suma
   - Nodo función de activación σ
   - Salida y a la derecha

2. Sliders de control:
   - "Peso w₁": -2 a +2
   - "Peso w₂": -2 a +2
   - "Sesgo b": -2 a +2
   - "Entrada x₁": 0 a 1
   - "Entrada x₂": 0 a 1

3. Cálculo paso-a-paso mostrado en tabla:
   - z = w₁·x₁ + w₂·x₂ + b
   - y = σ(z) donde σ es la función activación

4. Gráfica derecha: curva de σ(z) sobre rango -5 a +5
   - Punto rojo marcando el z actual y su salida y en la curva
   - Etiquetas: "pre-activación z = X.XX", "salida y = X.XX"

5. Selector función de activación:
   - Radio buttons: Sigmoid, Tanh, ReLU, Linear
   - Gráfica se actualiza dinámicamente

6. Visualización adicional:
   - Malla 2D: regiones donde y > 0.5 vs y ≤ 0.5 (coloración azul/rojo)
   - Línea de decisión (frontera lineal) en malla

7. Información pedagógica:
   - Tooltip: explicación de cada parámetro
   - "Tipo de problema que resuelve": clasificación binaria lineal

Usa Canvas o SVG. Enfoque: intuición sobre transformaciones lineales + no-linealidad.
```

**Librería**: Canvas/SVG, Chart.js
**Complejidad**: Media

---

### 3.2 Max-Pooling y Unpooling

**Concepto a visualizar**: Reducción y reconstrucción de mapas de características mediante pooling

**Tipo de interacción**:
- Usuario crea mapa de características pequeño (editando valores)
- Botón "Max-Pooling" que reduce resolución
- Botón "Unpooling" que reconstruye (con índices guardados)
- Visualización lado-a-lado: original → pooled → unpooled

**Descripción técnica**:
- Grillas 2D mostrando valores numéricos
- Colores indican magnitud (escala de calor)
- Indices de máximos guardados durante pooling
- Unpooling "rellena en blanco" las posiciones no-máximas
- Pool size (2×2 o 3×3) ajustable

**Prompt detallado para Claude Code**:
```
Crea un visualizador interactivo de Max-Pooling y Unpooling.
Requisitos:
1. Tres grillas 2D lado-a-lado (cada una 8×8 para entrada, 4×4 para pooled, 8×8 para unpooled):
   - IZQUIERDA: "Mapa de características original" (editable)
   - CENTRO: "Después de Max-Pooling"
   - DERECHA: "Después de Unpooling"

2. Mapa original editable:
   - Células son inputs (usuario puede escribir números -10 a +10)
   - O botón "Generador aleatorio" para llenar con valores random
   - Colormap: azul oscuro (negativo) → rojo (positivo)

3. Pool Size selector:
   - Radio buttons: 2×2 (reduce 8×8 → 4×4) o 3×3 (reduce 8×8 → 2×2)
   - Stride ajustable

4. Botón "Ejecutar Max-Pooling":
   - Para cada región de 2×2 (o 3×3), encuentra máximo
   - Dibuja en panel CENTRO
   - Registra índices (posición del máximo dentro de cada región)
   - Indices visualizados como números pequeños dentro de celdas

5. Visualización de índices:
   - En el mapa original, dibuja cuadrado alrededor de cada máximo seleccionado

6. Botón "Unpooling (NN o Max)":
   - Copia máximos a sus posiciones originales
   - Todas las otras celdas en 0 (o NaN, mostradas vacías)
   - Panel DERECHA muestra resultado

7. Tabla informativa:
   - "Parámetros:Pool Size=2×2, Stride=2"
   - "Original: 64 valores (8×8)"
   - "Pooled: 16 valores (4×4)"
   - "Unpooled: 64 valores (8×8, ~75% ceros)"

8. Estadísticas:
   - "Máximo global", "Promedio original vs pooled vs unpooled"

Usa HTML table con bordes y estilos CSS para colormaps. Enfoque: pérdida de información.
```

**Librería**: Canvas o HTML Table + CSS, Chart.js (para gráficas)
**Complejidad**: Media

---

### 3.3 Arquitectura U-Net Interactiva

**Concepto a visualizar**: Flujo de datos a través de encoder-bottleneck-decoder con skip connections

**Tipo de interacción**:
- Usuario carga imagen pequeña (o selecciona predefinida)
- Click en capas muestra tensor shape, visualiza feature maps
- Slider para profundidad de la red
- Toggle skip connections on/off

**Descripción técnica**:
- Diagrama de bloques mostrando arquitectura U
- Cada bloque etiquetado con dimensiones [C_in, H, W]
- Click en bloque muestra mapa de características de ejemplo
- Líneas conectando capas, colores para skip connections
- Animación de flujo de datos

**Prompt detallado para Claude Code**:
```
Crea un visualizador interactivo de la arquitectura U-Net.
Especificaciones:
1. Diagrama de arquitectura (SVG o Canvas):
   - Rama izquierda (Encoder): 4-5 bloques convolucionales de arriba a abajo
     - Cada bloque etiquetado: Conv + MaxPool
     - Dimensiones: [64, 256, 256] → [64, 128, 128] → [128, 64, 64] → [256, 32, 32] → [512, 16, 16]
   - Centro (Bottleneck): bloque central [512, 16, 16]
   - Rama derecha (Decoder): 4-5 bloques de abajo a arriba
     - Cada bloque etiquetado: Upsample + Conv
     - Dimensiones inversas de Encoder
   - Skip connections: líneas diagonales conectando capas del encoder al decoder
     - Colores especiales (ej: naranja) para skip connections
     - Etiqueta "Concatenate" en punto de unión

2. Click en cada bloque muestra:
   - Panel popover con detalles:
     - Nombre de capa, dimensiones de entrada y salida
     - Operación (Conv 3×3, MaxPool 2×2, Upsample 2×)
     - Número de parámetros
   - Visualización de feature map ejemplo (miniatura)
     - Si encoder: mostrardatos reales o sintetizados
     - Si decoder: mostrar reconstrucción progresiva

3. Entrada de imagen:
   - Canvas pequeño (128×128 o 256×256) para imagen de entrada
   - Usuario puede cargar imagen o usar presets (natural, patrón)
   - Botón "Propagate" anima flujo a través de red

4. Sliders:
   - "Profundidad de Encoder": 2 a 5 capas
   - "Canales iniciales": 32 a 256

5. Toggle:
   - Checkbox "Mostrar skip connections" (puede desactivarse)
   - Afecta visualmente al diagrama

6. Salida:
   - A la derecha o abajo: imagen de salida segmentada (ejemplo)
   - Comparación: entrada vs salida

7. Estadísticas globales:
   - "Total parámetros", "Profundidad máxima", "skip connections activas"

Usa SVG para diagrama, Canvas para feature maps. Enfoque: arquitectura, shapes, conexiones.
```

**Librería**: SVG (o D3.js), Canvas, HTML+CSS
**Complejidad**: Alta

---

### 3.4 Skip Connections

**Concepto a visualizar**: Impacto de skip connections en reconstrucción de detalles

**Tipo de interacción**:
- Slider para controlar qué capas tienen skip connections
- Dos canvases: segmentación sin/con skip connections
- Métrica de calidad Dice en tiempo real

**Descripción técnica**:
- Imagen de entrada (órgano/estructura)
- Red simplificada en diagrama
- Toggle para activar/desactivar skip connections
- Output visual comparando detalles
- Métrica Dice actualizada dinámicamente

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva mostrando el impacto de skip connections.
Componentes:
1. Interfaz dividida en dos columnas:
   - IZQUIERDA: "Sin Skip Connections"
   - DERECHA: "Con Skip Connections"

2. En cada columna:
   - Diagrama simplificado de U-Net (5 bloques en cada rama)
   - Encoder: rojo, Decoder: azul
   - En columna DERECHA: líneas naranjas indican skip connections
   - En columna IZQUIERDA: sin líneas naranjas

3. Entrada:
   - Canvas común (200×200) con imagen de prueba
     - Imagen sintética: círculo blanco sobre fondo gris (o imagen real médica miniaturizada)
   - Misma entrada se propaga a ambas ramas

4. Salidas (dos canvases 200×200):
   - IZQUIERDA: máscara de salida SIN skip (borrosa, detalles perdidos)
   - DERECHA: máscara de salida CON skip (detalles claros, bordes nítidos)

5. Slider "Profundidad de skip":
   - Controla qué capas en la arquitectura conectan al decoder
   - "Solo última capa", "2+ capas", "Todas"
   - Visualización se actualiza

6. Métrica de calidad (Dice coefficient):
   - "Dice Score SIN skip: X%"
   - "Dice Score CON skip: Y%"
   - Cálculo pseudo-real (simulado)
   - Barra de progreso visual mostrando diferencia

7. Información educativa:
   - Explicación: "skip connections preservan detalles finos del encoder"
   - "Sin ellas, información se pierde en bottleneck"

8. Animación opcional:
   - Botón "Animar propagación": flujo de datos se visualiza mediante
     colores transitorios en el diagrama

Usa Canvas para imágenes, SVG para diagrama. Enfoque: intuición sobre beneficios skip.
```

**Librería**: Canvas, SVG, Chart.js (para gráfica Dice)
**Complejidad**: Media-Alta

---

### 3.5 Dice vs Topología

**Concepto a visualizar**: Paradoja: alta métrica Dice ≠ topología correcta

**Tipo de interacción**:
- Usuario dibuja máscara (verdadera) y predicción en dos canvases
- Cálculo en tiempo real: Dice score + Betti numbers
- Caso de ejemplo: máscara sin agujeros vs predicción con agujero extra

**Descripción técnica**:
- Dos canvases editables: verdad fundamental vs predicción
- Cálculo de Dice score (2|A∩B|/(|A|+|B|))
- Análisis topológico: β₀, β₁ para cada máscara
- Comparación visual: regiones FP, FN, TP
- Conclusión: "Dice alto pero Betti números diferentes = error topológico"

**Prompt detallado para Claude Code**:
```
Crea una herramienta interactiva mostrando desacoplamiento entre Dice y Topología.
Requisitos:
1. Interfaz con tres secciones:
   - ARRIBA: Imagen de entrada (200×200, médica sintética)
   - ABAJO-IZQUIERDA: "Máscara Verdadera (Ground Truth)" - editable
   - ABAJO-DERECHA: "Predicción de Red" - editable

2. Canvases para máscaras (ambos 200×200):
   - Herramientas: pincel para dibujar (blanco=órgano, negro=fondo)
   - Borrador
   - Botón "Limpiar"
   - Presets de máscara:
     - "Círculo simple"
     - "Anillo (1 agujero)"
     - "Dos órganos separados"
     - "Forma compleja"

3. Cálculos en tiempo real (abajo):
   - TABLA de métricas:
     - Dice Score = 2|A∩B|/(|A|+|B|) en %
     - Número de voxeles verdaderos = X
     - Número de voxeles predichos = Y
     - Verdaderos Positivos (TP), Falsos Positivos (FP), Falsos Negativos (FN)
   - Análisis Topológico:
     - β₀(Ground Truth) = N₀, β₁(Ground Truth) = H₀
     - β₀(Predicción) = N₁, β₁(Predicción) = H₁
     - Color ROJO si β valores difieren, VERDE si iguales

4. Visualización de diferencias:
   - Canvas adicional mostrando:
     - Azul = TP (acuerdo)
     - Rojo = FP (predicción extra)
     - Verde = FN (perdido)
     - Este canvas ilustra por qué Dice no captura todo

5. Ejemplo predefinido (botón "Mostrar Paradoja"):
   - Ground Truth: círculo perfecto (β₀=1, β₁=0)
   - Predicción: círculo + punto extra dentro (β₀=2, β₁=0)
   - Calcula Dice: ~90% (parecido bueno)
   - Pero β₀ diferente = error topológico
   - Conclusión: "¡Aunque Dice es alto, la topología es incorrecta!"

6. Explicación textual:
   - "Dice mide solapamiento pixel-wise"
   - "Topología mide estructura global"
   - "Ambas métricas son necesarias para segmentación confiable"

Usa Canvas + JavaScript vanilla para análisis. Enfoque: limitaciones de métricas estándar.
```

**Librería**: Canvas HTML5, JavaScript vanilla
**Complejidad**: Media

---

## Documento 4: TDA-SegUNet

### 4.1 Pipeline TDA Completo

**Concepto a visualizar**: Flujo de datos en el método TDA-SegUNet: MRI → EDT → PH → PI → U-Net → Máscara

**Tipo de interacción**:
- Usuario carga imagen MRI (o usa predefinida)
- Botones "Siguiente" para avanzar por etapas
- Cada etapa muestra resultado intermedio y explicación

**Descripción técnica**:
- Diagrama de flujo con 6 cajas
- Click en caja muestra:
  1. Imagen MRI 2D
  2. Distancia euclidiana (EDT) como mapa de calor
  3. Complejo simplicial / persistencia diagrama
  4. Imagen de persistencia (heatmap)
  5. Feature maps U-Net
  6. Máscara final segmentada
- Animación progresiva con botón "Autoplay"

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva del pipeline TDA-SegUNet.
Especificaciones:
1. Diagrama de flujo horizontal con 6 etapas conectadas:
   Etapa 0: MRI → Etapa 1: EDT → Etapa 2: PH → Etapa 3: PI → Etapa 4: U-Net → Etapa 5: Máscara Final

2. Cada etapa representada como un rectángulo clickeable

3. Entrada (Etapa 0):
   - Canvas 200×200 mostrando imagen MRI 2D pequeña
   - Puede cargarse del usuario o usar imagen predefinida
   - Muestra contraste de grises (ventana nivel)

4. Etapa 1 (EDT - Euclidean Distance Transform):
   - Canvas 200×200 mostrando mapa de distancia
   - Colormap: azul oscuro (fondo, distancia=0) → rojo (interior profundo)
   - Slider para ajustar saturación de colores

5. Etapa 2 (Homología Persistente / Diagrama de Persistencia):
   - Panel 300×300 mostrando diagrama de persistencia
   - Puntos azules (H₀ - componentes)
   - Diagonal y=x de referencia
   - Explicación: "Nacimientos y muertes de características topológicas"

6. Etapa 3 (Imagen de Persistencia - PI):
   - Canvas 200×200 mostrando heatmap
   - Colormap viridis
   - Explicación: "Representación fija de diagrama via gaussianas"
   - Slider para ajustar bandwidth σ

7. Etapa 4 (U-Net Features):
   - Miniatura de arquitectura U-Net
   - 3-4 feature maps de ejemplo mostrados
   - Indicador de "procesamiento en progreso"

8. Etapa 5 (Máscara de Salida):
   - Canvas 200×200 mostrando máscara binaria
   - Overlay: máscara en rojo semitransparente sobre MRI original

9. Interfaz de navegación:
   - Botón "Anterior" (desactivado en etapa 0)
   - Botón "Siguiente" (desactivado en etapa 5)
   - Slider horizontal: posiciona en cualquier etapa (0-5)
   - Botón "Autoplay" que avanza cada 2 segundos

10. Panel de información:
    - Etapa actual: "Nombre y descripción técnica"
    - Fórmula matemática relevante (ej: "EDT[i,j] = min distancia a frontera")
    - Dimensions: entrada y salida de cada etapa

11. Datos de prueba:
    - Imagen MRI pequeña real o sintética (200×200)
    - Presets para diferentes órganos/estructuras

Usa Canvas, SVG para diagrama, HTML+CSS. Enfoque: visualizar el flujo completo.
```

**Librería**: Canvas, SVG, HTML+CSS
**Complejidad**: Muy Alta

---

### 4.2 Fusión de Canales (MRI + PI)

**Concepto a visualizar**: Concatenación de canales MRI e imagen de persistencia, impacto en extracción de características

**Tipo de interacción**:
- Sliders para activar/desactivar cada canal
- Visualización: entrada de MRI + entrada de PI + salida de la primera convolución
- Comparativa visual de feature maps

**Descripción técnica**:
- Tres canvases: MRI (escala grises), PI (heatmap), Feature map salida Conv1
- Toggle on/off para cada canal
- Kernel de convolución visible
- Estadísticas: magnitud promedio, varianza de activaciones

**Prompt detallado para Claude Code**:
```
Crea un visualizador interactivo de fusión de canales MRI+PI.
Requisitos:
1. Interfaz dividida en secciones:
   - IZQUIERDA: Entradas (2 canales)
   - CENTRO: Primera capa convolucional
   - DERECHA: Output (feature map)

2. IZQUIERDA - Canales de Entrada:
   - Canvas A (200×200): "Canal MRI" - imagen en escala de grises
     - Checkbox toggle: "Incluir canal MRI"
     - Slider "Escala de amplitud"
   - Canvas B (200×200): "Canal Imagen de Persistencia" - heatmap viridis
     - Checkbox toggle: "Incluir canal PI"
     - Slider "Escala de amplitud"

3. CENTRO - Operación de Fusión:
   - Diagrama esquemático mostrando concatenación
   - Si ambos canales: entrada Conv1D será [H, W, 2 canales]
   - Si solo MRI: entrada [H, W, 1 canal]
   - Si solo PI: entrada [H, W, 1 canal]
   - Si ninguno: "Ingrese al menos un canal"
   - Kernel de convolución 3×3 × C_in → C_out mostrado en miniatura

4. DERECHA - Salida Feature Map:
   - Canvas (200×200): "Feature map post-Conv1" - activaciones coloreadas
   - Colormap: cool (azul) para negativas, warm (rojo) para positivas
   - Normalización automática a rango visible

5. Estadísticas actualizadas en tiempo real:
   - Tabla mostrando:
     - "Canales de entrada: N" (1 o 2)
     - "Parámetros de Conv1: X"
     - "Magnitud promedio de entrada: μ"
     - "Magnitud promedio de salida: μ_out"
     - "Varianza de activaciones: σ²"

6. Comparativas:
   - Tres gráficos (Chart.js):
     - Histograma de intensidades MRI
     - Histograma de intensidades PI
     - Histograma de activaciones Conv1

7. Botones predefinidos:
   - "Solo MRI" (desactiva PI)
   - "Solo PI" (desactiva MRI)
   - "Fusión óptima" (activa ambos)

8. Observación educativa:
   - "Información topológica (PI) complementa datos de intensidad (MRI)"
   - "Fusión mejora capacidad del modelo de discriminación"

Usa Canvas, Chart.js. Enfoque: complementariedad de modalidades.
```

**Librería**: Canvas, Chart.js
**Complejidad**: Media-Alta

---

### 4.3 Betti Matching vs Wasserstein

**Concepto a visualizar**: Dos estrategias de comparación topológica: matching por número de Betti vs distancia de Wasserstein en diagrama

**Tipo de interacción**:
- Slider para deformar una forma
- Dos mapeos simultáneos: Betti (by count) vs Wasserstein (by diagram location)
- Visualización: formas + diagramas + distancias calculadas

**Descripción técnica**:
- Dos formas 2D (ej: islas en mapa)
- Canvas izquierdo muestra forma A, derecho forma B
- Slider "deformación" modifica forma B
- Diagrama PH: ambas formas representadas
- Líneas de matching Wasserstein visibles como arcos
- Números de Betti mostrados con colores (matching correcto = verde)

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva comparando Betti Matching y Wasserstein.
Especificaciones:
1. Interfaz dividida en secciones:
   - ARRIBA: Dos mapas 2D lado-a-lado (formas/topologías)
   - ABAJO: Dos diagramas de persistencia lado-a-lado

2. ARRIBA-IZQUIERDA - Forma A (referencia):
   - Canvas 250×250 dibujando una forma fija (ej: dos islas)
   - Colores: azul=agua, naranja=tierra
   - Etiqueta: "Forma de referencia"

3. ARRIBA-DERECHA - Forma B (deformable):
   - Canvas 250×250 similar a Forma A
   - Slider horizontal "Deformación" (0 a 1):
     - 0: idéntica a Forma A
     - 0.5: leve distorsión
     - 1: forma significativamente diferente (ej: islas separadas, fusionadas, con agujeros)
   - Etiqueta: "Forma a comparar"

4. ABAJO-IZQUIERDA - Diagrama A:
   - 300×300 mostrando puntos persistencia de Forma A
   - Azul = H₀, Rojo = H₁

5. ABAJO-DERECHA - Diagrama B:
   - 300×300 mostrando puntos persistencia de Forma B
   - Azul = H₀, Rojo = H₁

6. Overlay de Matching (en ambos diagramas):
   - Líneas rectas conectando puntos matcheados
   - Color de línea según tipo de matching:
     - VERDE: matching correcto (Betti numbers coinciden)
     - NARANJA: matching Wasserstein (por proximidad en diagrama)
     - ROJO: mismatch (diferente topología)

7. Panel de comparación (debajo):
   - BETTI MATCHING:
     - β₀(A) = 2, β₀(B) = ? → Verde si coincide, Rojo si no
     - β₁(A) = 0, β₁(B) = ? → Verde si coincide, Rojo si no
     - "Distancia de Betti" = diferencia absoluta en contador

   - WASSERSTEIN MATCHING:
     - Distancia de Wasserstein calculada entre diagramas
     - Mostrada numéricamente (ej: W = 1.23)
     - Visualización: suma de longitudes de arcos

8. Conclusión textual:
   - Si topología igual (verde): "Betti y Wasserstein coinciden"
   - Si topología distinta: "Betti detecta cambio, Wasserstein mide magnitud"
   - "Betti = conteo binario, Wasserstein = distancia métrica"

9. Presets de deformación:
   - Botones: "Separar islas", "Fusionar", "Agregar agujero"

Usa Canvas. Enfoque: métricas topológicas complementarias.
```

**Librería**: Canvas HTML5, Plotly.js (para diagramas)
**Complejidad**: Alta

---

### 4.4 Curriculum Training

**Concepto a visualizar**: Progreso de entrenamiento en fases: bootstrapping → regular → refinement

**Tipo de interacción**:
- Slider de epoch (0 a epochs_totales)
- Tres gráficas: loss, Dice score, topological loss
- Tres imágenes: ejemplos de predicción en cada fase
- Indicador visual del progreso

**Descripción técnica**:
- Gráficas de pérdidas usando Chart.js
- Línea vertical indicando epoch actual
- Ejemplos de segmentación (entrada, verdad, predicción) en tres fases
- Tablas con métricas en cada fase
- Anotaciones de cambios arquitectónicos (ej: "Activa pérdida topológica en epoch 50")

**Prompt detallado para Claude Code**:
```
Crea una visualización interactiva del entrenamiento en curriculum.
Requisitos:
1. Slider principal "Epoch": 0 a 200
   - Control fino (puede escribirse número directamente)

2. ARRIBA: Tres gráficas (Chart.js) lado-a-lado (200px alto cada una):
   - "Pérdida Total": línea azul decreciente
     - Fluctúa en fase bootstrapping, estable en regular, baja en refinement
   - "Dice Score": línea verde ascendente (0% a ~95%)
   - "Pérdida Topológica": línea roja
     - 0 en fase 1, activa desde epoch 50, relevante en fase 2
   - Línea vertical gris indicando epoch actual

3. Anotaciones en gráficas:
   - Fase 1 (Epochs 0-50): "Bootstrapping" (fondo verde claro)
   - Fase 2 (Epochs 50-150): "Entrenamiento Regular" (fondo azul claro)
   - Fase 3 (Epochs 150-200): "Refinement Topológico" (fondo naranja claro)
   - Cambios de arquitectura/loss marcados con "*"

4. MEDIO: Ejemplos de predicción (3 columnas, cada una muestra una fase):
   - Columna 1 (Época ~25):
     - Canvas 100×100: Entrada MRI
     - Canvas 100×100: Verdad
     - Canvas 100×100: Predicción (ruidosa, burda)
   - Columna 2 (Época ~100):
     - Mismo layout
     - Predicción más limpia, boundaries mejor
   - Columna 3 (Época ~180):
     - Mismo layout
     - Predicción topológicamente correcta, bordes nítidos

5. ABAJO: Tabla de métricas por fase:
   | Fase | Epochs | Dice Máx | Betti Acc. | Wasserstein Avg |
   |------|--------|----------|-----------|-----------------|
   | Boot | 0-50   | 0.65     | 50%       | 2.5             |
   | Regu | 50-150 | 0.92     | 85%       | 0.8             |
   | Fine | 150-200| 0.96     | 98%       | 0.3             |

6. Panel Informativo:
   - Texto dinámico explicando qué ocurre en fase actual:
     - "Bootstrapping: el modelo aprende formas básicas sin regularización topológica"
     - "Entrenamiento Regular: optimización conjunta de Dice + topología"
     - "Refinement: enfoque en precisión topológica"

7. Slider adicional "Velocidad de Reproducción": 0.1× a 4×
   - Afecta velocidad de avance automático (si se presiona Play)

8. Botones:
   - "Play" (autoplay de epochs)
   - "Reset" (vuelve a época 0)
   - "Ir a Fase" (combobox: Bootstrapping, Regular, Refinement)

9. Datos numéricos:
   - Usar datos sintetizados pero realistas (pérdidas y métricas plausibles)
   - Simulación o datos de entrenamiento real si disponible

Usa Chart.js para gráficas, Canvas para imágenes. Enfoque: 3 fases, progreso visible.
```

**Librería**: Chart.js, Canvas HTML5
**Complejidad**: Muy Alta

---

## Resumen de Visualizaciones

Tabla consolidada de todas las visualizaciones propuestas:

| # | Documento | Visualización | Librería Principal | Complejidad | Prioridad | Estimado (horas) |
|---|-----------|---------------|-------------------|-------------|-----------|------------------|
| 1.1 | Conv | Convolución 1D Interactiva | Canvas/SVG | Media | Alta | 3-4 |
| 1.2 | Conv | Teorema de Fourier | FFT.js + Canvas | Alta | Alta | 5-6 |
| 1.3 | Conv | Ecuación del Calor 2D | Canvas + Chart.js | Alta | Media | 5-6 |
| 1.4 | Conv | Convolución 2D Deslizante | Canvas | Media-Alta | Alta | 4-5 |
| 1.5 | Conv | Separabilidad Filtros | Plotly.js + MathJax | Media | Media | 3-4 |
| 2.1 | Topología | Deformaciones Topológicas | Three.js | Alta | Baja | 6-8 |
| 2.2 | Topología | Complejo Simplicial | Canvas | Media | Media | 4-5 |
| 2.3 | Topología | Filtración de Subnivel | Canvas + Chart.js | Alta | Alta | 5-6 |
| 2.4 | Topología | Diagrama de Persistencia | D3.js | Alta | Alta | 6-7 |
| 2.5 | Topología | Números de Betti | Canvas | Media-Alta | Alta | 4-5 |
| 2.6 | Topología | Imagen de Persistencia | Canvas + Plotly.js | Media | Media | 4-5 |
| 3.1 | CNNs | Neurona Artificial | Canvas/SVG + Chart.js | Media | Media | 3-4 |
| 3.2 | CNNs | Max-Pooling/Unpooling | Canvas | Media | Media | 3-4 |
| 3.3 | CNNs | Arquitectura U-Net | SVG + Canvas | Alta | Alta | 5-6 |
| 3.4 | CNNs | Skip Connections | Canvas + SVG | Media-Alta | Alta | 4-5 |
| 3.5 | CNNs | Dice vs Topología | Canvas | Media | Alta | 3-4 |
| 4.1 | TDA-SegUNet | Pipeline Completo | Canvas + SVG | Muy Alta | Alta | 8-10 |
| 4.2 | TDA-SegUNet | Fusión de Canales | Canvas + Chart.js | Media-Alta | Media | 4-5 |
| 4.3 | TDA-SegUNet | Betti vs Wasserstein | Canvas + Plotly.js | Alta | Media | 5-6 |
| 4.4 | TDA-SegUNet | Curriculum Training | Chart.js + Canvas | Muy Alta | Alta | 8-10 |

---

## Notas Técnicas Finales

### Librerías Recomendadas (CDN)

- **D3.js**: Visualizaciones interactivas avanzadas, gráficos complejos
- **Three.js**: Gráficos 3D, modelos parametrizados, animaciones
- **Plotly.js**: Gráficas científicas, interactividad, heatmaps
- **Chart.js**: Gráficas simples (líneas, barras, donuts)
- **Canvas HTML5**: Bajo nivel, máximo control, mejor rendimiento
- **SVG**: Gráficos vectoriales, escalables, buenos para diagramas
- **MathJax**: Renderización de ecuaciones LaTeX
- **FFT.js**: Transformada Rápida de Fourier

### Guía de Implementación

1. **Para cada visualización**:
   - Crear archivo `viz_NX.html` (donde N=documento, X=número)
   - Self-contained: todos CSS y JS inline o linked via CDN
   - Responsive: media queries para móvil
   - Comentarios en español explicando código

2. **Testing**:
   - Abrir en navegador moderno (Chrome, Firefox, Safari)
   - Verificar rendimiento (canvas debería ~60 FPS)
   - Validar cálculos con casos conocidos (unit tests)

3. **Integración Jekyll**:
   - Link `<iframe src="/assets/viz/viz_NX.html" width="100%" height="600px"></iframe>`
   - O embed directo en markdown

### Casos de Uso Pedagógico

- **Clase presencial**: proyectar en pantalla, interactuar en tiempo real
- **Tarea**: estudiantes experimentan, varían parámetros, reportan hallazgos
- **Blog**: lector explora a propio ritmo, autoevaluación

---

**Documento compilado**: HTML-Reasoning para TDA-SegUNet thesis blog, 2026-03-24
