# Proyecto: Comparación de Optimizadores en Redes Neuronales

Este proyecto implementa una red neuronal para resolver un problema de clasificación multiclase utilizando la base de datos MNIST. El objetivo es entrenar el modelo con varios optimizadores, ajustar hiperparámetros, evaluar su desempeño y visualizar los resultados.

---

## Contenido

1. [Requisitos](#requisitos)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Descripción del Código](#descripción-del-código)
4. [Instrucciones de Ejecución](#instrucciones-de-ejecución)
5. [Resultados](#resultados)
6. [Conclusiones](#conclusiones)
7. [Notas Adicionales](#notas-adicionales)

---

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes dependencias:

- **Python 3.8 o superior**
- Librerías necesarias:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Puedes instalar las dependencias ejecutando:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## Estructura del Proyecto

- **prueba1.py**: Script principal que implementa la red neuronal y realiza la comparación de optimizadores.
- **train.csv**: Conjunto de datos de entrenamiento (42000 imágenes de 28x28 píxeles).
- **test.csv**: Conjunto de datos de validación.
- **README.md**: Archivo de documentación del proyecto.

---

## Descripción del Código

El script `prueba1.py` realiza las siguientes tareas principales:

### 1. Preprocesamiento de Datos
- Descarga y carga de la base de datos MNIST desde los archivos `train.csv` y `test.csv`.
- Normalización de los datos para mejorar la convergencia de la red.

### 2. Definición del Modelo
- Creación de una red MLP (Perceptrón Multicapa) utilizando Keras.
- Arquitectura del modelo:
  - Múltiples capas ocultas con funciones de activación ReLU.
  - Capa de salida con 10 neuronas (correspondientes a los dígitos del 0 al 9) y función de activación softmax.

### 3. Prueba de Optimizadores
- Entrenamiento del modelo utilizando diferentes optimizadores:
  - SGD
  - Adam
  - Nadam
  - RMSprop
  - Adadelta
- Análisis del impacto de cada optimizador en el rendimiento del modelo.

### 4. Ajuste de Hiperparámetros
- Variación de:
  - Número de capas ocultas.
  - Neuronas por capa.
  - Tasa de aprendizaje.
  - Épocas de entrenamiento.

### 5. Evaluación y Métricas
- Evaluación del modelo utilizando métricas como:
  - Precisión
  - Recall
  - F1-score
  - Pérdida
- Presentación de la matriz de confusión.
- Gráficos de evolución de precisión y pérdida durante el entrenamiento y validación.

### 6. Comparación y Conclusiones
- Comparación de los resultados obtenidos con los distintos optimizadores y configuraciones de hiperparámetros.
- Conclusiones sobre la configuración de red y optimizador que ofrece el mejor rendimiento para la clasificación de dígitos en la base de datos MNIST.

---

## Instrucciones de Ejecución

1. Asegúrate de que los archivos `train.csv` y `test.csv` estén en el directorio raíz del proyecto.
2. Ejecuta el script principal:

```bash
python prueba1.py
```

3. Sigue las instrucciones que aparecen en la consola para seleccionar el optimizador y otros parámetros.
4. Los resultados se guardarán en el directorio de salida especificado o se mostrarán en la consola.

---

## Resultados

Los resultados incluyen:
- Gráficos de pérdida y precisión para cada optimizador.
- Comparación de métricas clave como precisión, recall y F1-score.
- Tablas resumen con el rendimiento de cada optimizador.
- Matriz de confusión para evaluar el desempeño del modelo.

---

## Conclusiones

Este proyecto demuestra cómo diferentes optimizadores y configuraciones de hiperparámetros afectan el rendimiento de una red neuronal en un problema de clasificación. Los resultados obtenidos pueden ayudar a seleccionar el optimizador y la configuración más adecuada para problemas similares.

---

## Notas Adicionales

- Asegúrate de que los datos en `train.csv` y `test.csv` estén correctamente formateados y preprocesados antes de ejecutar el script.
- Si deseas agregar nuevos optimizadores o modificar la arquitectura de la red, edita el archivo `prueba1.py` según sea necesario.

---


