# Clasificación de imágenes histopatológicas de tejido pulmonar

Proyecto de clasificación multi–clase usando PyTorch y una ResNet18 preentrenada. Está implementado en un único notebook.

## Cómo ejecutar

1. Instala dependencias (CPU):

   ```bash
   pip install numpy scikit-learn
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. Coloca los datos en la carpeta `data/` (no se suben al repo por tamaño). Descarga y descomprime desde:
   [https://drive.google.com/file/d/1vHpktP4M3uQOoh\_QlBAqcvC111o8e5ef/view](https://drive.google.com/file/d/1vHpktP4M3uQOoh_QlBAqcvC111o8e5ef/view)

   Estructura esperada:

   ```
<img width="332" height="371" alt="image" src="https://github.com/user-attachments/assets/5dd3d2a6-3c39-4d8d-b8bb-65e7b67e481b" />

3. Abre `reto.ipynb` y ejecuta las celdas en orden. No necesitas modificar parámetros para correrlo.

## Modelo seleccionado

* ResNet18 con pesos preentrenados en ImageNet.
* Se reemplaza la capa final por una capa lineal con 7 salidas.
* Entrenamiento en CPU, con early stopping y reducción automática de la tasa de aprendizaje.

## Pre-procesamiento

* Redimensionado a 224×224.
* Conversión a tensor.
* Normalización por canal con medias y desviaciones de ImageNet:
  `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.


## Hiperparámetros

* Imagen: 224×224
* Batch size: 32
* Épocas máximas: 10

* Early stopping: patience = 3
* División de datos (fold1): 70% train, 15% val, 15% test interno (estratificado)
* Conjunto de prueba final: `fold2` completo

## Métricas de evaluación 

* Accuracy global: 0.8269

* Resumen por clase (precision/recall/f1/support):

  ```
  aca_bd: p=1.00 r=0.75 f1=0.86 (8)
  aca_md: p=1.00 r=0.88 f1=0.93 (8)
  aca_pd: p=1.00 r=0.50 f1=0.67 (6)
  nor   : p=0.80 r=1.00 f1=0.89 (12)
  scc_bd: p=0.86 r=1.00 f1=0.92 (6)
  scc_md: p=0.40 r=0.50 f1=0.44 (4)
  scc_pd: p=0.78 r=0.88 f1=0.82 (8)
  accuracy: 0.83
  macro avg f1: 0.79
  weighted avg f1: 0.82
  ```

* Matriz de confusión:

  ```
  [[6 0 2 0 0 0 0]
   [0 7 0 1 0 0 0]
   [0 3 3 0 0 3 0]
   [0 0 12 0 0 0 0]
   [0 0 0 6 0 0 0]
   [0 0 0 0 0 2 2]
   [0 0 0 1 0 0 7]]
  ```
El modelo alcanzó una exactitud general del 83 por ciento, mostrando un buen desempeño en clases como nor, scc_bd, aca_bd y aca_md, donde logró altos valores de precisión y recall. Sin embargo, se observaron dificultades en aca_pd y especialmente en scc_md, con f1-scores bajos debido a confusiones frecuentes: aca_pd fue confundido con scc_md, y scc_md con scc_pd. Esto indica que el modelo distingue bien la mayoría de los tejidos, pero aún tiene problemas para separar subtipos con características muy similares, lo que sugiere la necesidad de aplicar más técnicas de data augmentation, ajuste fino del modelo y balanceo de clases para mejorar su rendimiento en estas categorías.

