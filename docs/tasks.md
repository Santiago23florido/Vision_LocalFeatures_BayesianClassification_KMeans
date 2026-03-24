## Plan 3.1: Clasificación bayesiana de píxeles en `Essex_Faces`

### Resumen
- Objetivo funcional: resolver `piel` vs `no-piel` sobre `img/Essex_Faces`, que operacionaliza tu pedido de `piel vs fondo` en un problema binario exhaustivo para clasificación por píxel.
- Número de clases: mantener `2` clases en 3.1 (`skin`, `non_skin`); dejar multiclase fuera de este alcance.
- Enfoque: refactor moderado con guardado de resultados, reutilizando [Display_Components.py](/home/javit/images/Vision_LocalFeatures_BayesianClassification_KMeans/src/Display_Components.py) y [Bayes_Model_Training.py](/home/javit/images/Vision_LocalFeatures_BayesianClassification_KMeans/src/Bayes_Model_Training.py).
- Comparativas obligatorias: espacios `RGB`, `HSV`, `YCrCb` y modelos `QDA` vs `GaussianNB`, además de una comparación `MAP` vs `ML`.

### Cambios de implementación
- Corregir el visor de componentes para que use `cv2.COLOR_BGR2YCrCb` en lugar de `BGR2YUV`, manteniendo el orden real `Y`, `Cr`, `Cb`, y permitir guardar figuras/canales para el informe.
- Extraer la lógica de features a un módulo reutilizable, por ejemplo `src/features.py`, con cuatro modos cerrados:
  - `rgb` -> `[R, G, B]`
  - `hsv` -> `[H, S, V]`
  - `ycrcb` -> `[Y, Cr, Cb]`
  - `cbcr_grad` -> `[Cb, Cr, |grad(Y)|]` como descriptor extendido y compacto para reducir sensibilidad a iluminación y añadir contexto local
- Refactorizar el script bayesiano para corregir el bug de RoI actual: las coordenadas, ancho/alto y el slicing deben quedar consistentes entre rectángulo dibujado, extracción del batch y creación de labels.
- Mantener entrenamiento interactivo con ratón/teclado, pero endurecer el flujo:
  - ignorar RoIs vacías
  - impedir entrenar sin muestras de ambas clases
  - etiquetar negativos como `non_skin` e incluir fondo, pelo, cejas, gafas y ropa dentro de esa clase
- Añadir guardado automático por corrida en `outputs/bayes/<imagen>/<features>_<model>_<decision>/` con:
  - máscara predicha
  - overlay máscara + imagen
  - mapa de probabilidad de `skin` si el modelo expone `predict_proba`
  - metadatos mínimos de corrida (`imagen`, `features`, `modelo`, `decision`, conteo de RoIs y píxeles por clase)

### Interfaces públicas
- [Display_Components.py](/home/javit/images/Vision_LocalFeatures_BayesianClassification_KMeans/src/Display_Components.py): aceptar `<image_path>` y flags `--space {rgb,hsv,ycrcb,all}`, `--save-dir`, `--no-show`.
- [Bayes_Model_Training.py](/home/javit/images/Vision_LocalFeatures_BayesianClassification_KMeans/src/Bayes_Model_Training.py): aceptar `<train_image>` y flags `--test-image`, `--features {rgb,hsv,ycrcb,cbcr_grad}`, `--model {qda,gnb}`, `--decision {map,ml}`, `--output-dir`.
- Regla de decisión:
  - `map`: usar priors empíricos derivados de los píxeles anotados
  - `ml`: forzar priors uniformes `[0.5, 0.5]`

### Plan experimental y validación
- Imagen de entrenamiento fija: `img/Essex_Faces/94/asamma.19.jpg`.
- Imágenes de prueba no vistas: `img/Essex_Faces/94/ajones.19.jpg` y `img/Essex_Faces/96/arwebb.19.jpg`.
- Protocolo de anotación en train: recoger `5-8` RoIs de piel y `5-8` RoIs de no-piel; los negativos deben cubrir fondo y elementos faciales/no faciales oscuros para evitar que el modelo aprenda “verde vs piel”.
- Matriz mínima de comparación:
  - features: `rgb`, `hsv`, `ycrcb`, `cbcr_grad`
  - modelos: `qda`, `gnb`
  - decisión: `map` y `ml` sobre el mismo conjunto de RoIs
- Criterios de aceptación:
  - el visor exporta correctamente canales RGB/HSV/YCrCb
  - la RoI dibujada coincide con la región usada para entrenar
  - cada corrida genera artefactos guardados sin sobrescribir otras configuraciones
  - al menos una configuración en `YCrCb` o `cbcr_grad` produce una máscara de piel visualmente más limpia que `RGB` en imágenes no vistas
  - la discusión final deja explícito que `QDA` modela covarianza completa y `GaussianNB` asume independencia por componente

### Supuestos y defaults
- No hay ground truth por píxel en el repo; la evaluación de 3.1 será cualitativa y comparativa, no basada en métricas de segmentación.
- El descriptor multiescala con Hessiana que aparece sugerido en el reporte queda fuera de esta iteración; `cbcr_grad` es la extensión elegida para mantener el alcance moderado.
- K-Means y multiclase facial quedan fuera de este plan porque pertenecen a 3.2 o a una extensión posterior.