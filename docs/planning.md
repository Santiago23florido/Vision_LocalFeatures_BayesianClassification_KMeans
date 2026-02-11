# Reparto propuesto (carga equitativa)

## Persona 1: Características (features) y análisis de espacios de color

**Objetivo:** preparar y justificar el espacio de observación (qué features se usan) y por qué.

**Tareas:**
- Ejecutar y documentar la comparación `RGB` vs `HSV` vs `YCbCr` con `Display_Components.py`.
- Responder y redactar:
  - Límites de usar solo color como características.
  - Propuesta de otras características (p. ej., textura, gradiente o vecindarios) y por qué ayudarían en su problema.
- Implementar (o dejar listo) un módulo simple de extracción de features reutilizable por Bayes y K-Means (aunque sea una primera versión).

**Entregables:**
- 2-3 figuras comparativas de componentes.
- Párrafos de justificación de features.

---

## Persona 2: Clasificación bayesiana supervisada (ML vs MAP y modelos gaussianos)

**Objetivo:** construir y evaluar el enfoque supervisado.

**Tareas:**
- Usar `Bayes_Model_Training.py` para entrenar con RoIs (píxeles "p" y "n") y probar segmentación.
- Responder y redactar:
  - Diferencia entre ML y MAP, y cómo se refleja en el programa (qué cambia conceptualmente en la decisión y en el uso de priors).
  - Comparación entre gaussiano multidimensional y gaussiano ingenuo (naif): qué asume cada uno y qué se observa en resultados.
- Probar al menos 2 configuraciones (espacio de observación y/o número de clases si amplían a más de 2) y guardar resultados.

**Entregables:**
- Capturas/figuras de segmentación Bayes.
- Tabla breve (texto) de `configuración -> efecto`.

---

## Persona 3: K-Means no supervisado y comparación contra Bayes

**Objetivo:** construir y evaluar el enfoque no supervisado y compararlo.

**Tareas:**
- Usar `KMeans_Clustering.py` para clusterizar (por defecto 6 clases) y luego ajustar `K` según el caso.
- Responder y redactar:
  - Elección de espacio de observación y `K` mejor adaptado, con dificultades típicas.
  - Ventajas e inconvenientes de K-Means frente al bayesiano supervisado (control, estabilidad, necesidad de etiquetas, etc.).
- Probar mínimo 2 valores de `K` y guardar resultados.

**Entregables:**
- Resultados de K-Means.
- Comparación directa (mismas imágenes/features) con Bayes.

---

## Equidad de carga (muy importante)

Para que nadie quede solo con teoría o solo con código:

- Cada persona trabaja sobre 1 dataset distinto (por ejemplo `INRA`, `Essex` o `Kitti`) y produce resultados del método que le tocó, más una breve discusión.
- Luego intercambian una imagen: cada quien corre rápidamente el método del otro (ej.: Persona 2 también corre K-Means en 1 imagen; Persona 3 también corre Bayes en 1 imagen). Con eso, todos aportan a la comparación.

---

## Reparto de escritura del informe (también equitativo)

- Persona 1: "Características y espacios de observación" + respuesta de límites del color.
- Persona 2: "Clasificación bayesiana" + ML vs MAP + comparación de modelos gaussianos.
- Persona 3: "K-Means y comparación" + pros/contras y discusión final.

Todos aportan figuras de resultados, porque el TP valora más los argumentos y la crítica que solo el resultado final.
