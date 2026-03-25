# Razonamiento para Videos Manim: Ilustrando Convoluciones, Topología, CNNs y TDA

## Filosofía General

Este documento contiene **prompts detallados para Claude Code** que generarán videos Manim de alta calidad para acompañar cada uno de los cuatro blog posts de la tesis.

### Principios de Diseño

- **Librería**: Manim Community Edition (manim-ce)
- **Duración por video**: 30-90 segundos
- **Calidad de renderizado**: 1080p a 30 fps
- **Idioma**: Español en todos los textos y anotaciones
- **Paleta de colores**:
  - Azules (#1E88E5, #42A5F5) para matemáticas
  - Verdes (#43A047, #66BB6A) para biología y estructuras
  - Naranjas (#FB8C00, #FFA726) para redes neuronales
  - Rojos (#E53935, #EF5350) para errores y discrepancias
  - Grises (#757575) para fondos y referencias
- **Enfoque pedagógico**: Cada video es autónomo pero conectado narrativamente con su blog
- **Animaciones**: Transiciones suaves, timing claro, énfasis visual en conceptos clave

---

## Documento 1: Teoría de Convoluciones

### Video 1.1: "Convolución como Ventana Deslizante"

**Título**: Convolución como Ventana Deslizante
**Duración estimada**: 60 segundos
**Complejidad**: Media

**Descripción narrativa**:
El video comienza mostrando dos funciones 1D sobre una recta numérica: f(x) (señal, en azul) y g(x) (kernel, en naranja). La vista muestra primero ambas funciones superpuestas. En la segunda escena, g se refleja horizontalmente para convertirse en g(-x), visualizado con una breve animación de volteo. Luego, la función reflejada comienza a deslizarse lentamente sobre f, siendo la posición actual t. En cada marco, la región de superposición se sombrea en verde suave. Un valor numérico aparece encima indicando la integral del producto en esa posición. A medida que el kernel se desliza de izquierda a derecha, se construye progresivamente la salida (la convolución) como una nueva curva que crece punto a punto en un gráfico inferior. Al final, se muestra la definición matemática completa de la convolución.

**Objetos Manim a usar**:
- `NumberLine`: dos líneas numéricas con escala
- `FunctionGraph`: gráficas de f(x) y g(x)
- `Text` / `Tex`: etiquetas ("f(x)", "g(x)", "g(-x)", "Convolución")
- `Polygon`: para sombrear la región de superposición
- `VGroup`: para agrupar funciones relacionadas
- `ValueTracker` + `Updater`: para la posición deslizante de g
- `Line` + `Arrow`: indicadores visuales de la integral
- `TexMobject` / `MathTex`: fórmula de convolución

**Prompt completo para Claude Code**:

```
Crea un video Manim que ilustre el concepto de convolución 1D como una ventana deslizante.

ESCENA 1 (0-10s): Introducción
- Coloca un objeto NumberLine horizontal, centrado, con rango [-3, 8] y escala de 1 unidad por cm.
- Define f(x) = sin(x) + 1 en color azul oscuro (#1E88E5), graficado en el rango [-3, 8].
- Define g(x) = exp(-x²/0.5) en color naranja (#FB8C00), una gaussiana estrecha.
- Añade ambas gráficas. Anima ambas apareciendo con un FadeIn de 1.5 segundos.
- Coloca etiquetas "f(x) = señal" y "g(x) = kernel" con flechas que señalen claramente cada función.

ESCENA 2 (10-20s): Reflexión del kernel
- El kernel g(x) debe transformarse visualmente en g(-x) con una animación de reflexión horizontal suave.
- La reflexión debe durar 2 segundos. Reposiciona g en x=1 después de la reflexión.
- Añade una etiqueta "g(-x): reflejo del kernel" que permanezca durante 2 segundos.

ESCENA 3 (20-50s): Deslizamiento y superposición
- Crea una ValueTracker que controle la posición t del kernel, que varía de -3 a 8 en 20 segundos.
- En cada paso de animación, dibuja un Polygon que sombree (verde claro, opacidad 0.3) la región donde f(x) y g(t-x) se superponen.
- Calcula y muestra el valor numérico de la integral del producto (aproximación mediante suma de Riemann) encima de la gráfica.
- Dibuja una línea vertical que indique la posición actual de t en el eje x.
- El gráfico debe actualizar continuamente para mostrar esta integral a cada paso.

ESCENA 4 (50-60s): Resultado final
- Detén el movimiento del kernel.
- Dibuja una curva nueva (en verde oscuro, #43A047) que represente la convolución (f * g)(t), calculada con los valores de integral acumulados.
- Muestra la fórmula matemática: (f * g)(t) = ∫ f(τ) g(t-τ) dτ con animación Tex que aparece en el lado derecho.
- Fade Out de todas las etiquetas excepto la fórmula final.

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Color de fondo: blanco (#FFFFFF)
- Tipografía: TeX por defecto para todas las matemáticas
- Audio: ninguno (se añadirá después)
```

---

### Video 1.2: "Del Dominio Espacial al Frecuencial"

**Título**: Del Dominio Espacial al Frecuencial
**Duración estimada**: 60 segundos
**Complejidad**: Media-Alta

**Descripción narrativa**:
El video muestra la equivalencia fundamental: convolución en el dominio espacial = multiplicación en el dominio frecuencial. Comienza con una señal temporal compuesta por múltiples sinusoides en el lado izquierdo. El lado derecho vacío espera el espectro de frecuencias. Una transformada de Fourier anima la transición: la señal espacial se emborracha gradualmente mientras aparecen picos discretos en el dominio de frecuencias (su transformada). Luego, se introduce un segundo gráfico inferior que muestra la convolución en el dominio espacial (dos señales superpuestas y entrelazadas). A continuación, una animación transforma este resultado en multiplicación puntual en el dominio frecuencial. Al final, ambos resultados se superponen para demostrar la igualdad.

**Objetos Manim a usar**:
- `Axes`: dos pares de ejes (tiempo/frecuencia, amplitud/magnitud)
- `FunctionGraph`: señal temporal, espectro de frecuencias
- `BarChart`: para representar el espectro discreto
- `VGroup`: para agrupar dominio espacial vs frecuencial
- `Arrow`: flechas de conexión "FFT", "IFFT", "="
- `TexMobject` / `MathTex`: ecuaciones de equivalencia
- `Dot`: marcadores de picos de frecuencia

**Prompt completo para Claude Code**:

```
Crea un video Manim que demuestre la equivalencia: convolución espacial = multiplicación frecuencial.

ESCENA 1 (0-15s): Señal espacial y su transformada
- Divide la pantalla en dos mitades: izquierda (Dominio Espacial), derecha (Dominio Frecuencial).
- Lado izquierdo: crea un gráfico Axes de 8 segundos vs amplitud [-2, 2].
- Dibuja una señal f(t) = sin(2π·0.5·t) + 0.5·sin(2π·1.5·t) + 0.3·sin(2π·3·t), en azul oscuro.
- Esta señal aparece con Write animation, dibujando punto a punto, durante 3 segundos.
- Añade etiqueta "Dominio Espacial: f(t)" sobre el gráfico izquierdo.

ESCENA 2 (15-35s): Transformada de Fourier
- Lado derecho: crea un gráfico Axes con frecuencia [0, 4] Hz vs magnitud [0, 1].
- Dibuja una transformada de Fourier aproximada: barras (BarChart) en frecuencias 0.5, 1.5, 3 Hz con alturas 1.0, 0.7, 0.4.
- La transformada debe "emerger" desde cero con AnimationGroup: cada barra crece de 0 a su altura en 0.8 segundos.
- Añade etiqueta "Dominio Frecuencial: F(ω)" sobre el gráfico derecho.
- Dibuja una flecha grande "FFT ↔" en el centro que aparece con FadeIn.

ESCENA 3 (35-50s): Equivalencia de operaciones
- Reducir ligeramente ambos gráficos para hacer espacio para texto informativo.
- Sobre el gráfico izquierdo, mostrar la fórmula de convolución: (f * g)(t) = ∫ f(τ) g(t-τ) dτ
- Sobre el gráfico derecho, mostrar la fórmula de multiplicación: F(ω) · G(ω)
- Animar aparición de ambas fórmulas simultáneamente con FadeIn, lado por lado.
- Añade una flecha bidireccional que conecte ambas fórmulas, etiquetada "son equivalentes".

ESCENA 4 (50-60s): Resumen visual
- Desvanecer los gráficos, mantener solo las fórmulas.
- Mostrar un resumen de texto: "Convolución en el tiempo = Multiplicación en frecuencia"
- Último fotograma: mantener solo esta conclusión con ambas fórmulas lado a lado.

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Color de fondo: blanco
- Usar colores: azul (#1E88E5) para dominio espacial, verde (#43A047) para dominio frecuencial
- Tipografía: TeX
```

---

### Video 1.3: "Convolución 2D: El Filtro Deslizante"

**Título**: Convolución 2D: El Filtro Deslizante
**Duración estimada**: 45 segundos
**Complejidad**: Media

**Descripción narrativa**:
El video ilustra cómo funcionan los filtros 2D en procesamiento de imágenes. Comienza mostrando una rejilla 4×4 con valores numéricos (representando una imagen pequeña) en color azul. A continuación, aparece un kernel 3×3 (en naranja) en la esquina superior izquierda. El kernel se desliza lentamente, posición a posición, sobre la imagen. En cada posición, se sombrea la región de superposición en verde, y se muestra el cálculo elemento a elemento (multiplicación y suma). El resultado se escribe en la posición correspondiente de un mapa de características 2×2. El video prosigue mostrando tres kernels diferentes (blur, edge detection, sharpen) y sus efectos visuales sobre una imagen pequeña.

**Objetos Manim a usar**:
- `Matrix` / `TexMatrix`: para representar matrices 4×4 de píxeles y kernel 3×3
- `Rectangle` / `Polygon`: para resaltar la ventana de superposición
- `Tex` / `Text`: para valores numéricos y etiquetas
- `VGroup`: para agrupar imagen, kernel, salida
- `ValueTracker` + `Updater`: para controlar la posición deslizante
- `Arrow`: para indicar el kernel moviéndose
- `FadeOut` / `FadeIn`: para transiciones entre diferentes kernels

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre la convolución 2D con un filtro deslizante.

ESCENA 1 (0-8s): Presentación de imagen y kernel
- Coloca una matriz 4×4 en el lado izquierdo, con valores enteros entre 0 y 9.
  Valores sugeridos: [[1,2,3,4], [5,6,7,8], [9,8,7,6], [5,4,3,2]]
- Colorea esta matriz con gradient azul (#1E88E5).
- Añade etiqueta "Imagen: 4×4" debajo.
- Coloca un kernel 3×3 en naranja (#FB8C00) al lado derecho con valores [[1,0,-1], [2,0,-2], [1,0,-1]] (detector de bordes).
- Añade etiqueta "Kernel (Sobel): 3×3" debajo.
- Ambos deben aparecer con FadeIn durante 1 segundo.

ESCENA 2 (8-30s): Deslizamiento del kernel
- Mueve el kernel a la posición superior izquierda (0,0) de la imagen con duración 0.5 segundos.
- Crea un bucle que itera sobre posiciones (i,j) donde i,j ∈ {0,1} (2×2 salida).
- Para cada posición:
  a) Destaca el kernel y la región superpuesta con un borde grueso en color rojo (#E53935).
  b) Calcula: suma = Σ(imagen[i:i+3, j:j+3] * kernel).
  c) Muestra la operación algebraica (elemento a elemento) en una caja de texto temporal.
  d) Escribe el resultado en la matriz de salida (2×2) en posición (i,j), en color verde (#43A047).
  e) Anima el kernel moviéndose a la siguiente posición (duración 0.6 segundos).
- La matriz de salida debe aparecer a la derecha, construyéndose gradualmente.

ESCENA 3 (30-45s): Comparación de kernels
- Mantén la imagen original en la izquierda.
- Reemplaza el kernel actual con tres versiones sucesivas:
  1) Kernel Blur (Gaussiana 3×3 normalizada) - duración 3 segundos
  2) Kernel Edge (Sobel) - duración 3 segundos
  3) Kernel Sharpen [[0,-1,0], [-1,5,-1], [0,-1,0]] - duración 3 segundos
- Para cada kernel, muestra su nombre y el resultado final (solo la salida 2×2) en color apropiado.
- Al final, dibuja tres outputs pequeños lado a lado: Blur, Edge, Sharpen.

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Color de fondo: blanco
- Matrices: usar monospace font para valores numéricos
- Transiciones: 0.3s entre escenas
```

---

## Documento 2: Topología y Persistencia

### Video 2.1: "De la Taza de Café al Donut"

**Título**: De la Taza de Café al Donut
**Duración estimada**: 45 segundos
**Complejidad**: Alta

**Descripción narrativa**:
Un video visualmente impactante que muestra una taza de café (modelada como una forma 3D simple) que se transforma gradualmente en un donut (toro 3D). El énfasis está en que el "agujero" fundamental —representado por el asa de la taza, que se convierte en el agujero del donut— se preserva durante la transformación. El video incluye anotaciones que destacan la persistencia del agujero: un número 1 de Betti (β₁ = 1) permanece constante. La cámara rota alrededor de la forma para mostrar claramente el agujero desde múltiples ángulos. Al final, se muestra una comparación lado a lado: taza y donut, con el texto "Topología idéntica: ambas tienen UN agujero".

**Objetos Manim a usar**:
- `Surface` / `ParametricSurface`: para generar formas 3D (taza y toro)
- `ThreeDScene` / `ThreeDCamera`: para navegación 3D
- `TexMobject`: para anotaciones de números de Betti
- `Tex` / `Text`: para etiquetas descriptivas
- `VGroup`: para agrupar taza + etiqueta, toro + etiqueta
- `Rotate` / `Move`: para transformar la taza en toro
- `Circle` / `Arrow`: para resaltar visualmente el agujero

**Prompt completo para Claude Code**:

```
Crea un video Manim 3D que muestre la transformación topológica de una taza de café a un donut.

ESCENA 1 (0-10s): Introducción de la taza
- Inicia una escena 3D (ThreeDScene).
- Genera una taza de café como una superficie de revolución (Surface/ParametricSurface).
  Parámetros sugeridos: radio exterior 1, radio interior 0.6, altura 1.5, con asa semicircular de radio 0.3 saliendo del lateral.
- Colorea la taza en azul suave (#42A5F5).
- Posiciona la cámara a 45° elevación y 0° azimut para una vista lateral clara que muestre el asa.
- Añade etiqueta "Taza de Café" debajo con MathTex: "β₁ = 1 (un agujero)".
- La taza debe FadeIn durante 1.5 segundos.

ESCENA 2 (10-35s): Transformación a donut
- Comienza la transformación interpolando los parámetros de la superficie de la taza hacia los parámetros de un toro.
- Toro paramétrico: X(u,v) = ((R + r·cos(v))·cos(u), (R + r·cos(v))·sin(u), r·sin(v))
  donde R = 1.2 (radio mayor) y r = 0.4 (radio menor).
- La transformación debe durar 8 segundos, animada suavemente con UPDATE de la superficie.
- Durante la transformación, rota la cámara lentamente: azimut de 0° a 360°, para mostrar el agujero desde todos los ángulos.
- Mantén la anotación "β₁ = 1" visible durante toda la transformación, parpadeando ligeramente.

ESCENA 3 (35-45s): Comparación final
- Detén la transformación cuando la taza sea completamente un toro.
- Divide la pantalla: muestra la taza original a la izquierda (pequeña, semitransparente) y el donut final a la derecha.
- Añade dos etiquetas:
  - Izquierda: "Taza: β₁ = 1"
  - Derecha: "Donut: β₁ = 1"
- Coloca un signo de igualdad grande en el centro.
- Muestra el texto: "La topología es lo que importa, no la forma" en la base.

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Iluminación 3D: tres luces (frontal, lateral, trasera)
- Usar Material con specular highlight para realismo
- Color de fondo: gris muy claro (#F5F5F5)
```

---

### Video 2.2: "Filtración de Subnivel: Inundando el Paisaje"

**Título**: Filtración de Subnivel: Inundando el Paisaje
**Duración estimada**: 90 segundos
**Complejidad**: Alta

**Descripción narrativa**:
Un video fascinante que visualiza cómo se construye un diagrama de persistencia mediante filtración. Comienza mostrando un paisaje 2D con colinas y valles (una función altura). El color inicial es gris neutro. A continuación, una línea de agua (threshold) comienza a subir lentamente desde el fondo. A medida que sube:
1. Primero aparecen islas (componentes conectadas, β₀ birth)
2. Cuando dos islas se tocan, se fusionan (β₀ death)
3. Cuando se forma un lago rodeado de tierra (un agujero), β₁ birth
4. Cuando ese lago se llena, β₁ death

Cada evento genera un punto en el diagrama de persistencia que aparece dinámicamente. El video muestra tanto el paisaje como el diagrama de persistencia creciendo lado a lado, con anotaciones que explican cada evento.

**Objetos Manim a usar**:
- `Surface`: paisaje 2D (función de altura)
- `Axes`: ejes para el diagrama de persistencia
- `Dot` / `Scatter`: puntos en el diagrama que aparecen dinámicamente
- `Line`: línea de agua que sube
- `Text` / `Tex`: anotaciones de eventos (nace, muere)
- `ValueTracker`: para controlar la altura del agua
- `Updater`: para actualizar dinámicamente color de superficie y diagrama

**Prompt completo para Claude Code**:

```
Crea un video Manim que visualice la filtración de subnivel como un paisaje siendo inundado.

ESCENA 1 (0-10s): Presentación del paisaje
- Crea una superficie de altura usando Surface: h(x,y) = sin(πx)·cos(πy) + 0.3·sin(2πx).
- Dominio: x,y ∈ [-1, 1], valores altura en [0, 2].
- Colorea inicialmente en gris (#757575).
- Proyecta el paisaje en 2D (vista desde arriba), mostrando contornos de altura con etiquetas numéricas.
- Añade etiqueta "Paisaje de altura" arriba.
- Una segunda subescena (lado derecho) muestra los ejes del diagrama de persistencia:
  eje horizontal = birth (tiempo de aparición), eje vertical = death (tiempo de desaparición), rango [0, 2.5] ambos.
  Dibuja la diagonal y=x en gris claro.

ESCENA 2 (10-80s): Inundación y eventos de persistencia
- Crea un ValueTracker para la altura del agua t, variando de 0 a 2 durante 60 segundos.
- Para cada valor de t:
  a) Colorea la superficie: píxeles con altura ≤ t en azul oscuro (#1E88E5), píxeles con altura > t en gris (#757575).
  b) Detecta cambios topológicos en el árbol de fusión:
     - Cuando una nueva isla aparece: crea un Dot en (t, ?) en el diagrama, mostrado temporalmente sin color de muerte.
       Añade una etiqueta "+β₀" que desaparece después de 0.3s.
     - Cuando dos islas se fusionan: el punto β₀ más antiguo se actualiza a death=t, dibujando su Dot completamente en rojo (#E53935).
       Añade una etiqueta "-β₀" por 0.3s.
     - Cuando se forma un lago (agujero rodeado por tierra): crea un Dot en (t, ?) para β₁, mostrado en verde (#43A047).
       Etiqueta "+β₁".
     - Cuando el lago se llena (el agujero desaparece): actualiza el punto β₁ a death=t, dibujando el Dot completamente.
       Etiqueta "-β₁".
- En el diagrama de persistencia, mantén todos los puntos previos visibles, construyendo una nube de puntos.
- Dibuja una línea horizontal y=t que se actualice junto con la altura del agua (referencia visual).

ESCENA 3 (80-90s): Resumen
- Detén la inundación cuando t=2 (toda la imagen esté bajo agua).
- El diagrama de persistencia final debe mostrar:
  - Varios puntos β₀ dispersos (rojo oscuro)
  - Algunos puntos β₁ (verde oscuro)
  - Todos los puntos están sobre la diagonal, cerca de ella.
- Muestra el texto: "Cada punto = evento topológico. Distancia a la diagonal = persistencia."

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Usar colores: azul para agua, gris para tierra seca, rojo para β₀, verde para β₁
- Actualización de colores debe ser suave (no parpadeos)
```

---

### Video 2.3: "Números de Betti: Contando Agujeros"

**Título**: Números de Betti: Contando Agujeros
**Duración estimada**: 60 segundos
**Complejidad**: Media

**Descripción narrativa**:
Un video educativo que construye progresivamente ejemplos de números de Betti. Comienza con un punto (β₀=1, no hay agujeros), luego un círculo (β₀=1, β₁=1: un componente conectado, un agujero). A continuación, una esfera (β₀=1, β₁=0, β₂=1: el agujero de volumen), y un toro (β₀=1, β₁=2, β₂=1: dos agujeros 1D y uno 2D). Finalmente, muestra una aplicación médica: formas tumorales con sus números de Betti calculados, sugiriendo que la topología podría usarse para clasificación. Cada forma aparece con sus números de Betti etiquetados, y las formas se alternan con explicaciones textuales.

**Objetos Manim a usar**:
- `Circle` / `Arc` / `ParametricCurve`: para círculos y curvas
- `Surface`: para esferas y toros
- `Dot`: para el punto inicial
- `TexMobject` / `Text`: para números de Betti y etiquetas
- `VGroup`: para agrupar forma + etiqueta + Betti numbers
- `Rectangle`: para recuadros de información

**Prompt completo para Claude Code**:

```
Crea un video Manim que enseñe números de Betti mediante ejemplos progresivos.

ESCENA 1 (0-12s): Punto
- Dibuja un pequeño círculo relleno (punto) en el centro de la pantalla.
- Coloréalo en azul (#1E88E5).
- Debajo, muestra:
  β₀ = 1 (un componente conectado)
  β₁ = 0 (sin agujeros 1D)
  β₂ = 0 (sin agujeros 2D)
- Etiqueta: "Punto: El objeto más simple"
- Todo aparece con FadeIn durante 1.5 segundos.

ESCENA 2 (12-24s): Círculo
- Crea un Circle (perímetro, no relleno) con radio 0.8.
- Coloréalo en verde (#43A047).
- Debajo, muestra:
  β₀ = 1
  β₁ = 1 (ahora hay un agujero)
  β₂ = 0
- Etiqueta: "Círculo: Un agujero 1D"
- Resalta el agujero (el interior) con un sombreado rojo suave que parpadea 2 veces.
- La forma anterior (punto) desaparece con FadeOut.

ESCENA 3 (24-36s): Esfera 3D
- Crea una esfera 3D (superficie) con ParametricSurface o Sphere.
- Coloréala en naranja (#FFA726).
- Muestra:
  β₀ = 1
  β₁ = 0
  β₂ = 1 (el interior de la esfera es un agujero 2D)
- Etiqueta: "Esfera: Un agujero 2D (volumen)"
- Rota la cámara 180° para mostrar la esfera desde múltiples ángulos.
- El círculo anterior desaparece con FadeOut.

ESCENA 4 (36-48s): Toro
- Crea un toro 3D (anillo de dona).
- Coloréalo en púrpura/magenta (#AB47BC).
- Muestra:
  β₀ = 1
  β₁ = 2 (dos agujeros 1D: el del donut y el del interior)
  β₂ = 1 (el volumen interior)
- Etiqueta: "Toro: Dos agujeros 1D y uno 2D"
- Anima dos flechas que señalen los dos agujeros 1D: el grande (central) y el pequeño.
- Rota la cámara alrededor del toro.
- La esfera anterior desaparece.

ESCENA 5 (48-60s): Aplicación médica
- Divide la pantalla en dos: izquierda para forma tumoral 1, derecha para forma tumoral 2.
- Forma 1 (izquierda): un blob con un agujero (topología de donut).
  β₀ = 1, β₁ = 1, β₂ = 0. Etiqueta: "Tumor A: Topología tipo donut"
- Forma 2 (derecha): un blob sin agujero (topología de esfera).
  β₀ = 1, β₁ = 0, β₂ = 1. Etiqueta: "Tumor B: Topología esférica"
- Muestra el texto: "Diferentes topologías → Diferentes características clínicas"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Cámara 3D para escenas con objetos 3D
- Colores vibrantes con buena diferenciación
- Transiciones suaves entre escenas (0.5s)
```

---

### Video 2.4: "De Diagrama de Persistencia a Imagen de Persistencia"

**Título**: De Diagrama de Persistencia a Imagen de Persistencia
**Duración estimada**: 60 segundos
**Complejidad**: Alta

**Descripción narrativa**:
Un video que muestra la transformación matemática desde un diagrama de persistencia (gráfico birth-death) hasta una imagen de persistencia (heatmap 2D). Comienza mostrando un diagrama de persistencia estándar con puntos dispersos. Luego, transforma los ejes: el eje y cambia de "death" a "persistence = death - birth". Los puntos se reproyectan a estas nuevas coordenadas. A continuación, se coloca un kernel gaussiano en cada punto. Las gaussianas se superponen y suman, creando una superficie suave. Finalmente, esta superficie se discretiza en una cuadrícula (heatmap), mostrando colores que representan la intensidad. El video destaca que esta representación es más robusta a perturbaciones y es más fácil de usar como entrada para redes neuronales.

**Objetos Manim a usar**:
- `Axes`: para diagramas birth-death y birth-persistence
- `Scatter` / `Dot`: para puntos del diagrama
- `Surface`: para la suma de gaussianas
- `ColoredSurface` / `Heatmap`: para la imagen de persistencia final
- `Arrow`: para indicar transformaciones de ejes
- `Line`: para cambio de ejes (diagonal a vertical)
- `Text` / `TexMobject`: anotaciones de fórmulas

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre la construcción de una imagen de persistencia desde un diagrama.

ESCENA 1 (0-12s): Diagrama de persistencia inicial
- Crea un par de ejes Axes con:
  - Eje X (birth): rango [0, 2], etiquetado "Nacimiento"
  - Eje Y (death): rango [0, 2], etiquetado "Muerte"
- Dibuja la diagonal y=x en gris claro.
- Coloca 8-10 puntos Scatter en posiciones sugeridas:
  (0.2, 0.8), (0.4, 1.2), (0.6, 1.3), (0.8, 1.1), (1.0, 1.9), (1.2, 1.5), (1.4, 1.8), (1.6, 1.4)
- Colorea los puntos en azul (#1E88E5).
- Etiqueta: "Diagrama de Persistencia (birth, death)"
- Los puntos aparecen uno a uno con una AnimationGroup, duración total 2 segundos.

ESCENA 2 (12-28s): Cambio de coordenadas
- Sobre el diagrama, muestra la transformación:
  - Nueva coordenada: persistence = death - birth
  - Nuevo eje Y: "Persistencia"
- Anima el cambio de ejes: el eje Y se relabela, los puntos se reposicionan suavemente.
- Nueva posición de punto (b, d) es (b, d-b). Animación de reposicionamiento: 2 segundos.
- Muestras dos fórmulas:
  Antigua: (b, d)
  Nueva: (b, p) donde p = d - b
- Estos cambios ocurren simultáneamente en una nueva vista a la derecha del diagrama anterior.

ESCENA 3 (28-45s): Kernels gaussianos
- Mantén el diagrama (b, p) visible en la izquierda con sus 8-10 puntos.
- En una vista grande a la derecha, dibuja la suma de gaussianos:
  - Para cada punto (b_i, p_i), coloca un kernel gaussiano 2D: G(x,y) = exp(-((x-b_i)² + (y-p_i)²)/(2σ²)) con σ=0.2.
  - Suma todas las gaussianas: I(x,y) = Σ_i G(x-b_i, y-p_i)
- Visualiza esto como una Surface coloreada en blanco/azul con gradiente de altura.
- Anima la aparición gradual de cada gaussiana, una tras otra, con duración 0.3s cada una.
- Al final, muestra la superficie suma completa, suave y continua.

ESCENA 4 (45-60s): Discretización en heatmap
- Transforma la superficie continua en una imagen discreta (heatmap).
- Discretiza el rango de (b, p) en una cuadrícula 32×32 (o similar).
- Colorea cada celda según el valor I(x,y): colores van de blanco (valor bajo) a rojo oscuro (#E53935) (valor alto).
- Anima la construcción de la heatmap: dibuja celda por celda, o toda a la vez con animación de gradiente.
- Debajo, muestra el texto: "Imagen de Persistencia: Lista para entrada en redes neuronales"
- La superficie continua de la escena anterior desaparece.

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Mapa de colores: viridis o similar (blanco → azul → rojo)
- Sigma gaussiana: 0.2 (en unidades de rango normalizado)
- Transiciones: 1s entre escenas principales
```

---

## Documento 3: CNNs y U-Net

### Video 3.1: "Max-Pooling y la Pérdida de Información"

**Título**: Max-Pooling y la Pérdida de Información
**Duración estimada**: 45 segundos
**Complejidad**: Media

**Descripción narrativa**:
El video ilustra un problema crítico en redes neuronales convolucionales: max-pooling pierde información espacial. Comienza mostrando una matriz 4×4 con valores numéricos. Se aplica max-pooling 2×2, reduciendo a 2×2, destacando cuál valor fue seleccionado. Luego, intenta restaurar la información con upsampling simple (interpolación), mostrando que la información está perdida. A continuación, se introduce un truco alternativo: SegNet guarda los índices de los máximos durante pooling. Estos índices se usan durante unpooling para restaurar valores exactamente en las posiciones correctas. El video muestra cómo SegNet logra reconstrucción perfecta mientras que upsampling simple falla.

**Objetos Manim a usar**:
- `Matrix` / `TexMatrix`: para matrices de valores
- `Rectangle`: para resaltar regiones de pooling
- `Polygon`: para sombras de max-pooling
- `Arrow`: para indicar qué valor fue seleccionado
- `Text` / `Tex`: para etiquetas y valores
- `VGroup`: para agrupar matrices en comparaciones
- `Highlight`: para resaltar diferencias

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre el problema de max-pooling y la solución de SegNet.

ESCENA 1 (0-8s): Max-pooling
- Muestra una matriz 4×4 con valores numéricos a la izquierda:
  [[7,3,5,2], [8,4,6,1], [2,9,3,7], [5,1,8,4]]
- Coloréala en azul (#1E88E5).
- Dibuja recuadros verdes alrededor de cada región 2×2 (4 regiones en total).
- En cada región, resalta en naranja (#FB8C00) el valor máximo.
  Región (0,0): máximo = 8
  Región (0,1): máximo = 6
  Región (1,0): máximo = 9
  Región (1,1): máximo = 8
- A la derecha, muestra la matriz de salida 2×2: [[8,6], [9,8]]
- Añade etiquetas: "Max-Pooling 2×2" y pequeñas flechas del máximo a la salida.

ESCENA 2 (8-20s): Upsampling simple (fracasa)
- Intenta restaurar la matriz original desde la matriz 2×2 usando upsampling simple.
- El método: replicar cada valor 2×2.
  [[8,8,6,6], [8,8,6,6], [9,9,8,8], [9,9,8,8]]
- Mostrada en rojo claro (#EF5350).
- Compara lado a lado con la original (azul) en un recuadro con diferencias resaltadas en rojo oscuro.
- Mostrar el número de errores: "8 de 16 píxeles incorrectos"
- Etiqueta: "Upsampling simple: FRACASA ✗"

ESCENA 3 (20-35s): SegNet con guardado de índices
- Vuelve al paso de max-pooling, pero esta vez guarda también los ÍNDICES.
- Para cada región, muestra un número (0, 1, 2, o 3) que indica la posición del máximo dentro del bloque 2×2.
  Índices en la matriz de salida: [[3,3], [1,3]]
- Etiqueta: "Guardar índices del máximo"
- Luego, en el proceso de reconstrucción (unpooling):
  - Recibe la matriz 2×2.
  - Para cada valor, lo coloca EXACTAMENTE en la posición indicada por el índice guardado.
  - Resultado: [[0,0,0,0], [0,0,0,6], [0,9,0,0], [0,0,0,8]]
    (Nota: aquí simplifico; el proceso real es más complejo, pero esto ilustra la idea).
- Mostrada en verde (#43A047).
- Compara con la original: "14 de 16 píxeles correctos" (solo los máximos fueron recuperados).
- Etiqueta: "SegNet con índices: MEJOR ✓"

ESCENA 4 (35-45s): Conclusión
- Muestra las tres matrices lado a lado:
  Original | Upsampling Simple (rojo) | SegNet (verde)
- Resalta la diferencia fundamental: "SegNet preserva posiciones espaciales exactas"
- Conclusión de texto: "Para segmentación: preservar información espacial es crítico"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Matrices: monospace font, valores numéricos de 1 dígito
- Transiciones de color suaves
```

---

### Video 3.2: "La Arquitectura U-Net: El Viaje del Tensor"

**Título**: La Arquitectura U-Net: El Viaje del Tensor
**Duración estimada**: 90 segundos
**Complejidad**: Alta

**Descripción narrativa**:
Un video que visualiza el arquitectura U-Net mostrando cómo un tensor fluye a través de la red. Comienza con una imagen de entrada (por ejemplo, una resonancia magnética del cerebro mostrada esquemáticamente). El tensor atraviesa el codificador (encoder path) en el lado izquierdo, reduciéndose espacialmente pero aumentando en canales. En el cuello de botella (bottleneck), la dimensión espacial es mínima pero los canales alcanzan su máximo. Luego, el decodificador (decoder path) en el lado derecho aumenta la dimensión espacial nuevamente. Lo crucial es mostrar las conexiones residuales (skip connections) que vinculan la salida del codificador con la entrada del decodificador, preservando detalles espaciales. Al final, el tensor se transforma en una máscara de segmentación. Se muestran las dimensiones exactas del tensor en cada paso.

**Objetos Manim a usar**:
- `Rectangle`: para representar tensores como bloques
- `VGroup`: para apilar tensores verticalmente
- `Arrow`: para mostrar el flujo de datos
- `Text` / `Tex`: para etiquetas de dimensiones (H×W×C)
- `Line`: para skip connections (diagonales)
- `Color`: codificación de colores para canales
- `SceneCamera`: para transiciones suaves

**Prompt completo para Claude Code**:

```
Crea un video Manim que visualice la arquitectura U-Net y el flujo de tensores.

ESCENA 1 (0-10s): Entrada
- Muestra un rectángulo que representa la imagen de entrada.
- Dimensiones: 128×128×1 (imagen en escala de grises, como MRI).
- Coloréalo en azul oscuro (#1E88E5).
- Etiqueta: "Input: 128×128×1"
- Dibuja una pequeña imagen MRI esquemática dentro (círculo gris con un punto central representando un tumor).

ESCENA 2 (10-40s): Encoder path (camino hacia abajo)
- Mostrar secuencialmente cada bloque del codificador:
  - Conv Block 1: 128×128×1 → 128×128×64, mostrado como rectángulo más alto (más canales) pero mismo ancho/alto.
  - MaxPool 1: 128×128×64 → 64×64×64, rectángulo más pequeño (ancho/alto reducido).
  - Conv Block 2: 64×64×64 → 64×64×128.
  - MaxPool 2: 64×64×128 → 32×32×128.
  - Conv Block 3: 32×32×128 → 32×32×256.
  - MaxPool 3: 32×32×256 → 16×16×256.
- Cada bloque aparece con FadeIn (0.4s), seguido de una animación de flecha Down para ir al siguiente.
- Los bloques están alineados a la izquierda y se apilan hacia abajo.

ESCENA 3 (40-50s): Bottleneck
- Mostrar el cuello de botella: Conv Block 4.
- Dimensiones: 16×16×256 → 16×16×512.
- Es el punto más restrictivo espacialmente, con más canales.
- Color especial en naranja (#FFA726) para destacarlo.
- Etiqueta: "Bottleneck: 16×16×512"

ESCENA 4 (50-80s): Decoder path (camino hacia arriba) con skip connections
- Mostrar secuencialmente cada bloque del decodificador:
  - UpSample 1: 16×16×512 → 32×32×512.
  - Skip connection: dibuja una flecha diagonal roja (#E53935) desde el Conv Block 3 (32×32×256) hasta este punto, conectando las salidas.
  - Concatenate: 32×32×512 + 32×32×256 → 32×32×768.
  - Conv Block 5: 32×32×768 → 32×32×256.
  - Repetir para niveles superiores con upsample, skip, concatenate, conv.
- Estos bloques están alineados a la derecha, apilados hacia arriba, formando la forma de "U".
- Las skip connections deben ser muy visibles: líneas gruesas y brillantes en rojo/naranja.
- Cada bloque + skip connection animado en 0.6s.

ESCENA 5 (80-90s): Salida
- El decodificador final produce: 128×128×1 (máscara de segmentación).
- Coloréalo en verde (#43A047).
- Muestra una máscara esquemática (el tumor segmentado resaltado en blanco).
- Etiqueta: "Output: 128×128×1 (Máscara)"
- Dibuja una flecha desde la salida al tensor original de entrada, creando visualmente la forma de "U".
- Texto final: "U-Net: Encoder + Skip Connections + Decoder = Preserva detalles espaciales"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Proporciones de rectángulos: altura ∝ número de canales, ancho ∝ dimensión espacial
- Skip connections con color rojo brillante para claridad
- Alineación: encoder a la izquierda, decoder a la derecha, formando U clara
```

---

### Video 3.3: "Dice Alto, Topología Rota"

**Título**: Dice Alto, Topología Rota
**Duración estimada**: 60 segundos
**Complejidad**: Media

**Descripción narrativa**:
Un video que destaca el problema: un modelo puede lograr una puntuación Dice muy alta (métrica de superposición) pero aún violar la topología. El video muestra un ejemplo concreto: una verdad de referencia con forma de anillo (β₁=1), y una predicción que casi coincide excepto por un píxel faltante que rompe el anillo en una cadena (β₁=0). El Dice score calculado es 0.93 (excelente), pero la topología es incorrecta. El video destaca este error con flashes rojos. Luego muestra un segundo ejemplo con un meningioma: la predicción tiene un falso agujero (topología incorrecta) mientras que Dice sigue siendo alto. La conclusión es clara: métricas clásicas son insuficientes para aplicaciones donde la topología importa.

**Objetos Manim a usar**:
- `Grid` / `Pixels`: para mostrar píxeles de la imagen
- `Rectangle`: para destacar píxeles
- `Text` / `Tex`: para cálculos de Dice y números de Betti
- `Arrow`: para señalar diferencias
- `Highlight`: para parpadeos rojos
- `VGroup`: para comparaciones lado a lado
- `BarChart`: para mostrar métricas

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre el conflicto entre Dice alto y topología incorrecta.

ESCENA 1 (0-15s): Anillo perfecto vs anillo roto
- Divide la pantalla izquierda/derecha.
- Lado izquierdo (Ground Truth):
  - Dibuja un anillo 8×8 píxeles (cuadrado hueco de 4×4 exterior, 2×2 interior).
  - Píxeles del anillo en verde (#43A047).
  - Etiqueta: "Ground Truth: β₁ = 1"
  - Muestra una caja de información: "Número de Betti β₁ = 1 (un agujero)"
- Lado derecho (Predicción):
  - El mismo anillo PERO con un píxel faltante en la parte superior, rompiendo la continuidad.
  - Píxeles de predicción en naranja (#FFA726).
  - Etiqueta: "Predicción: β₁ = 0"
  - Caja de información: "Número de Betti β₁ = 0 (sin agujero, es una cadena)"
- Resalta el píxel faltante con un recuadro rojo parpadeante.

ESCENA 2 (15-30s): Cálculo de Dice
- Muestra la fórmula de Dice debajo: Dice = 2|GT ∩ Pred| / (|GT| + |Pred|)
- GT tiene 12 píxeles, Pred tiene 11 píxeles (falta uno).
- Intersección: 11 píxeles.
- Cálculo: Dice = 2·11 / (12+11) = 22/23 = 0.956 ≈ 0.96
- Mostrar este número en grande, en color verde (¡se ve bien!).
- Etiqueta: "Dice = 0.956 ✓ (excelente)"
- Pero debajo, mostrar: "Pero... β₁ Ground Truth: 1, β₁ Predicción: 0 ✗ (error topológico)"

ESCENA 3 (30-45s): Ejemplo médico - Meningioma
- Cambiar a un ejemplo médico más realista: contorno de meningioma.
- Lado izquierdo (GT): contorno sólido de tumor sin agujeros.
  β₀ = 1, β₁ = 0, β₂ = 0
- Lado derecho (Predicción): el mismo contorno pero con un pequeño agujero espurio (falso positivo) en el interior.
  β₀ = 1, β₁ = 1, β₂ = 0
- Calcular Dice: siguen siendo muy similares (Dice ≈ 0.92).
- Pero el falso agujero es un error topológico significativo.
- Mostrar: "Falso agujero" con flecha roja parpadeante al agujero.

ESCENA 4 (45-60s): Conclusión
- Mostrar una tabla resumen:
  | Métrica    | Valor | Conclusión       |
  | Dice       | 0.92  | Muy bueno ✓     |
  | β₁ Correcto| No    | Incorrecto ✗     |
  | Clínicamente| Inaceptable | ERROR ✗ |
- Texto grande: "Métricas clásicas son insuficientes"
- Conclusión: "Necesitamos pérdidas topológicas para garantizar corrección estructural"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Píxeles: 10×10 mm o similar para claridad
- Parpadeos rojos: 3 parpadeos de 0.2s cada uno
- Colores claros: verde (GT), naranja (Pred), rojo (error)
```

---

## Documento 4: TDA-SegUNet

### Video 4.1: "El Pipeline TDA-SegUNet Completo"

**Título**: El Pipeline TDA-SegUNet Completo
**Duración estimada**: 90 segundos
**Complejidad**: Alta

**Descripción narrativa**:
Un video integral que visualiza todo el pipeline TDA-SegUNet de principio a fin. Comienza con una imagen MRI (cerebro en sección axial). La imagen fluye a través de varios componentes secuenciales:
1. Cálculo de EDT (Euclidean Distance Transform) - visualizado como mapa de calor
2. Generación de filtración de subnivel y diagrama de persistencia
3. Construcción de imagen de persistencia (heatmap 2D suave)
4. Concatenación de canales (MRI + imagen de persistencia)
5. Procesamiento por U-Net (mostrando el flujo de tensores)
6. Salida de segmentación con topología preservada

Cada componente se visualiza como un bloque con flechas conectando, creando un flujo visual claro. Se muestran las dimensiones de los tensores en puntos clave.

**Objetos Manim a usar**:
- `Rectangle`: para bloques del pipeline
- `Arrow`: para mostrar flujo
- `Text` / `Tex`: para etiquetas de componentes y dimensiones
- `Surface`: para visualizar heatmaps (EDT, imagen de persistencia)
- `VGroup`: para agrupar bloques relacionados
- `Color`: codificación de colores para diferentes módulos
- `Highlight`: para resaltar componentes activos

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre el pipeline completo de TDA-SegUNet.

ESCENA 1 (0-10s): Entrada - Imagen MRI
- Muestra una imagen MRI esquemática (círculo gris con pequeños detalles internos que representen estructuras cerebrales).
- Dimensiones: 128×128×1.
- Coloréala en azul (#1E88E5).
- Etiqueta: "Entrada: MRI 128×128×1"
- Posiciónala en el lado izquierdo de la pantalla.

ESCENA 2 (10-28s): Cálculo de EDT
- Flecha hacia la derecha.
- Nuevo bloque: "EDT (Euclidean Distance Transform)"
- Visualiza el EDT como un mapa de calor: puntos cercanos al contorno en rojo, puntos lejanos en azul.
- La imagen MRI original se ve en el fondo, semitransparente.
- Dimensiones: salida 128×128×1.
- Colorea el mapa de calor: azul (distancia baja) → rojo (#E53935) (distancia alta).
- Duración: 5 segundos para mostrar el cálculo.

ESCENA 3 (28-45s): Filtración y Diagrama de Persistencia
- Flecha hacia la derecha.
- Nuevo bloque: "Filtración & Persistencia"
- Visualiza el proceso de filtración de subnivel (nivel de agua subiendo, como en Video 2.2, pero acelerado).
- Muestra un diagrama de persistencia construyéndose dinámicamente en la esquina inferior.
- Duración: 6 segundos.

ESCENA 4 (45-60s): Imagen de Persistencia
- Flecha hacia la derecha.
- Nuevo bloque: "Imagen de Persistencia"
- Transforma el diagrama en una heatmap 2D (como en Video 2.4).
- Muestra la suma de gaussianas transformándose en un heatmap discreto.
- Dimensiones: salida 128×128×1.
- Colorea: blanco (bajo) → rojo (#E53935) (alto).
- Duración: 5 segundos.

ESCENA 5 (60-70s): Concatenación de canales
- Muestra los dos tensores lado a lado: MRI (azul) e Imagen de Persistencia (rojo/blanco).
- Una flecha de fusión muestra la concatenación.
- Resultado: 128×128×2.
- Etiqueta: "Canales concatenados: 2"
- El nuevo tensor es más alto (representa más canales).

ESCENA 6 (70-85s): U-Net
- El tensor concatenado entra en U-Net (mostrar el bloque grande de U-Net con skip connections como en Video 3.2, pero versión comprimida).
- Mostrar brevemente el flujo encoder→bottleneck→decoder.
- Salida: 128×128×1 (máscara de segmentación).
- Coloréala en verde (#43A047).

ESCENA 7 (85-90s): Salida final
- Mostrar la máscara de segmentación final superpuesta en la MRI original.
- Etiqueta: "Salida: Segmentación con Topología Preservada"
- Mostrar el número de Betti β₁ (correcto) superpuesto.
- Texto: "TDA-SegUNet: Combina topología con aprendizaje profundo"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Flujo de izquierda a derecha, de arriba a abajo
- Coloración coherente: azul para entrada, rojo para topología, verde para salida
- Flechas grandes y claras entre bloques
- Mantener coherencia visual con videos 2.2, 2.4, 3.2
```

---

### Video 4.2: "Betti Matching vs Wasserstein: ¿Importa la Ubicación?"

**Título**: Betti Matching vs Wasserstein: ¿Importa la Ubicación?
**Duración estimada**: 60 segundos
**Complejidad**: Media-Alta

**Descripción narrativa**:
Un video que compara dos estrategias de emparejamiento en diagramas de persistencia: Wasserstein (matching óptimo clásico) y Betti matching (emparejamiento basado en número de Betti). El video muestra un escenario médico: predicción de tumor con múltiples focos de tumor predichos vs. verdad de referencia. Wasserstein intenta emparejar puntos de manera óptima globalmente, a veces emparejando incorrectamente un foco distante en lugar del foco correcto cercano. Betti matching, por el contrario, preserva la estructura local: superpone los mapas y empareja puntos por ubicación. Se muestra cómo esto genera gradientes correctos para entrenar el modelo, mientras que Wasserstein puede generar señales de gradiente contradictorias. El video destaca que para segmentación, la ubicación espacial importa.

**Objetos Manim a usar**:
- `Scatter` / `Dot`: para puntos en diagrama de persistencia
- `Arrow`: para mostrar emparejamientos (matching)
- `Line` / `Curve`: para conexiones de emparejamiento
- `VGroup`: para lado a lado comparación
- `Text` / `Tex`: para etiquetas y gradientes
- `Color`: rojo para Wasserstein (incorrecto), verde para Betti matching (correcto)
- `Highlight`: para resaltar diferencias

**Prompt completo para Claude Code**:

```
Crea un video Manim que compare Wasserstein matching y Betti matching.

ESCENA 1 (0-12s): Setup - Predicción vs Ground Truth
- Divide la pantalla: izquierda (Ground Truth), derecha (Predicción).
- Lado izquierdo: diagrama de persistencia con 4 puntos: (0.2, 1.0), (0.5, 1.5), (0.8, 1.3), (1.2, 1.8)
  Coloréalos en azul (#1E88E5).
  Etiqueta: "Ground Truth (β₁ = 4)"
- Lado derecho: diagrama de persistencia con 4 puntos: (0.3, 0.9), (0.6, 1.4), (0.9, 1.2), (1.1, 1.7)
  Coloréalos en naranja (#FFA726).
  Etiqueta: "Predicción (β₁ = 4)"
- Dibuja la diagonal y=x en gris claro en ambos.

ESCENA 2 (12-30s): Wasserstein Matching
- Título: "Wasserstein Matching: Minimiza distancia total global"
- Muestra ambos diagramas lado a lado.
- Dibuja flechas de emparejamiento de Wasserstein con color rojo (#E53935):
  - GT(0.2, 1.0) ↔ Pred(0.3, 0.9) ✓ (correcto)
  - GT(0.5, 1.5) ↔ Pred(1.1, 1.7) ✗ (incorrecto, muy lejano)
  - GT(0.8, 1.3) ↔ Pred(0.6, 1.4) ✗ (incorrecto, intercambiado)
  - GT(1.2, 1.8) ↔ Pred(0.9, 1.2) ✗ (incorrecto)
- Las flechas deben cruzarse (visualizar el desorden).
- Mostrar: "Distancia Wasserstein total: 0.87 (calculado)"

ESCENA 3 (30-45s): Betti Matching
- Título: "Betti Matching: Preserva estructura local por ubicación"
- Muestra los mismos dos diagramas.
- Dibuja flechas de emparejamiento de Betti matching con color verde (#43A047):
  - GT(0.2, 1.0) ↔ Pred(0.3, 0.9) ✓ (correcto)
  - GT(0.5, 1.5) ↔ Pred(0.6, 1.4) ✓ (correcto, cercano)
  - GT(0.8, 1.3) ↔ Pred(0.9, 1.2) ✓ (correcto, cercano)
  - GT(1.2, 1.8) ↔ Pred(1.1, 1.7) ✓ (correcto, cercano)
- Las flechas NO se cruzan (orden preservado).
- Mostrar: "Distancia Betti: 0.45 (calculado)"

ESCENA 4 (45-60s): Implicaciones para el entrenamiento
- Mostrar dos columnas: "Wasserstein" (rojo) vs "Betti" (verde).
- Bajo cada columna, mostrar un vector de gradiente:
  - Wasserstein: vector apuntando en dirección confusa (múltiples componentes, algunos contradictorios).
    Etiqueta: "Gradientes contradictorios → Convergencia lenta"
  - Betti: vector apuntando claramente hacia Ground Truth.
    Etiqueta: "Gradientes consistentes → Convergencia rápida"
- Conclusión de texto: "Para segmentación, la ubicación espacial crítica es preservada por Betti matching"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Flechas gruesas (2mm) para visibilidad
- Colores contrastantes: rojo (Wasserstein, incorrecto) vs verde (Betti, correcto)
- Diagramas idénticos, solo cambia el matching
```

---

### Video 4.3: "Curriculum Training: Despertando la Topología"

**Título**: Curriculum Training: Despertando la Topología
**Duración estimada**: 60 segundos
**Complejidad**: Media

**Descripción narrativa**:
Un video que muestra el proceso de entrenamiento con curriculum learning, donde se comienza con una pérdida clásica (Dice) y se añade gradualmente una pérdida topológica. El video visualiza cómo el diagrama de persistencia evoluciona durante el entrenamiento. Al inicio (épocas tempranas), la predicción es casi aleatoria, con un diagrama de persistencia trivial (todos los puntos cerca de la diagonal). Después del entrenamiento Dice solo, la predicción es estructurada pero la topología aún no es correcta (diagrama disperso pero no coincide con GT). Luego, se añade la pérdida topológica (Betti matching), y el diagrama comienza a converger hacia el GT, punto por punto. La visualización muestra simultáneamente: la predicción, el diagrama de persistencia, y la curva de pérdida, todos evolucionando en sincronía.

**Objetos Manim a usar**:
- `Scatter` / `Dot`: para diagramas de persistencia
- `Axes`: para gráfico de pérdida
- `FunctionGraph` / `Line`: para curva de pérdida
- `Rectangle`: para predicción (máscara)
- `Text` / `Tex`: para etiquetas de fase y épocas
- `VGroup`: para agrupar las tres visualizaciones
- `ValueTracker`: para animar sobre épocas
- `Color`: azul (Dice), verde (Topología), rojo (combinado)

**Prompt completo para Claude Code**:

```
Crea un video Manim que muestre el curriculum training de TDA-SegUNet.

ESCENA 1 (0-10s): Predicción inicial (aleatoria)
- Divide la pantalla en tres paneles:
  Panel izquierdo: Predicción (máscara)
  Panel central: Diagrama de persistencia
  Panel derecho: Gráfico de pérdida

- Predicción inicial: ruido aleatorio, sin estructura clara. Mostrada en gris semitransparente.
- Diagrama: todos los puntos esparcidos aleatoriamente cerca de la diagonal (trivial).
  Etiqueta: "Época 0: Aleatorio"
- Gráfico de pérdida: vacío (empieza en época 1).

ESCENA 2 (10-35s): Entrenamiento solo con Dice
- Título: "Fase 1: Entrenamiento Dice (épocas 1-50)"
- La predicción comienza a refinarse, mostrando estructuras reconocibles pero topología incorrecta.
  Anima la predicción evolucionando gradualmente (mostrar 3-4 fotogramas de predicción en épocas 10, 20, 35, 50).
- El diagrama de persistencia se actualiza: los puntos comienzan a formar un patrón, pero NO coinciden con GT.
  Mostrar puntos verdes para los puntos GT (fijos), puntos rojos para predicción (moviéndose).
- La curva de pérdida Dice desciende. Usa color azul para la línea.
  Mostrar valores: época 1 (Dice=0.5), época 25 (Dice=0.82), época 50 (Dice=0.91).
- Etiqueta: "Dice mejora, pero topología NO converge"

ESCENA 3 (35-55s): Entrenamiento con pérdida topológica añadida
- Título: "Fase 2: Entrenamiento Dice + Topología (épocas 51-100)"
- La predicción continúa refinándose, ahora hacia la topología correcta.
  Animar predicción en épocas 60, 75, 90, 100.
- El diagrama de persistencia: los puntos rojos comienzan a converger hacia los puntos verdes.
  Anima los puntos rojos moviéndose hacia sus vecinos verdes.
- Se añade una segunda curva de pérdida en rojo/naranja, mostrando la pérdida topológica.
  Ambas curvas (Dice azul, Topología roja) descienden.
- Al final (época 100), los puntos rojo y verde casi coinciden en el diagrama.
- Etiqueta: "Topología converge a Ground Truth"

ESCENA 4 (55-60s): Conclusión
- Mostrar lado a lado: predicción final (verde, correcta) vs Ground Truth (azul, referencia).
- Diagrama final: puntos rojos y verdes superpuestos.
- Gráfico de pérdida final: ambas curvas planas (convergencia).
- Texto: "Curriculum Learning + Pérdida Topológica = Segmentación topológicamente correcta"

REQUISITOS TÉCNICOS:
- Resolución: 1920x1080
- FPS: 30
- Tres paneles: 1/3 cada uno de ancho
- Diagrama de persistencia: diagonal gris, puntos GT verdes (fijos), puntos Pred rojos (móviles)
- Gráfico de pérdida: línea azul (Dice), línea roja (Topología), escala Y = [0, 1]
- Épocas: etiqueta X en el gráfico de pérdida: 0, 25, 50, 75, 100
```

---

## Resumen de Videos

### Tabla de Referencia

| # | Documento | Video | Duración | Objetos Manim Principales | Complejidad |
|----|-----------|-------|----------|--------------------------|-------------|
| 1.1 | Convoluciones | Ventana Deslizante | 60s | NumberLine, FunctionGraph, ValueTracker | Media |
| 1.2 | Convoluciones | Dominio Espacial-Frecuencial | 60s | Axes, BarChart, Arrow, FadeIn/Out | Media-Alta |
| 1.3 | Convoluciones | Filtro 2D Deslizante | 45s | Matrix, Rectangle, ValueTracker | Media |
| 2.1 | Topología | Taza al Donut | 45s | Surface, ThreeDScene, Rotate | Alta |
| 2.2 | Topología | Filtración de Subnivel | 90s | Surface, Axes, Dot, Updater, ValueTracker | Alta |
| 2.3 | Topología | Números de Betti | 60s | Circle, Surface, Dot, Text | Media |
| 2.4 | Topología | Persistencia → Imagen | 60s | Axes, Scatter, Surface, Heatmap | Alta |
| 3.1 | CNNs | Max-Pooling Información | 45s | Matrix, Rectangle, Arrow, Highlight | Media |
| 3.2 | CNNs | Arquitectura U-Net | 90s | Rectangle, Arrow, Text, Line (skip conn.) | Alta |
| 3.3 | CNNs | Dice Alto, Topología Rota | 60s | Pixels, Rectangle, Arrow, BarChart | Media |
| 4.1 | TDA-SegUNet | Pipeline Completo | 90s | Rectangle, Arrow, Surface, VGroup | Alta |
| 4.2 | TDA-SegUNet | Betti vs Wasserstein | 60s | Scatter, Arrow, Axes, VGroup | Media-Alta |
| 4.3 | TDA-SegUNet | Curriculum Training | 60s | Scatter, Axes, FunctionGraph, Updater | Media-Alta |

**Total de videos**: 13
**Duración total**: ~900 segundos (15 minutos)
**Complejidad promedio**: Media-Alta

---

## Notas Finales para Implementación

1. **Orden de creación recomendado**: Empezar con los videos de **complejidad media** (1.1, 1.3, 3.1, 3.3) para familiarizarse con Manim, luego avanzar a **complejidad media-alta** (1.2, 2.3, 4.2, 4.3), y finalmente los **complejos** (2.1, 2.2, 2.4, 3.2, 4.1).

2. **Reutilización de código**: Muchos videos comparten componentes. Crear funciones auxiliares en Python para:
   - Diagramas de persistencia reutilizables
   - Matrices 2D/3D genéricas
   - Animadores de threshold/filtración
   - Constructores de ejes etiquetados

3. **Paleta de colores final**:
   - Matemática: Azul #1E88E5, #42A5F5
   - Biología: Verde #43A047, #66BB6A
   - Redes Neuronales: Naranja #FB8C00, #FFA726
   - Errores: Rojo #E53935, #EF5350
   - Neutro: Gris #757575, #F5F5F5

4. **Requisitos de audio**: Considerar añadir narración en español (60 palabras por minuto ≈ 900 palabras para 15 minutos total).

5. **Validación**: Después de crear cada video, verificar que las matemáticas sean correctas y que el mensaje pedagógico sea claro.

