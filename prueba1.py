# =========================
# IMPORTAR LIBRERÍAS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. CARGAR DATOS
# =========================
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# =========================
# 2. SEPARAR X Y Y
# =========================
X = df_train.drop('label', axis=1)
y = df_train['label']

# =========================
# 3. NORMALIZAR
# =========================
X = X / 255.0
X_test = df_test / 255.0

# =========================
# 4. ONE HOT ENCODING
# =========================
y = to_categorical(y, num_classes=10)

# =========================
# 5. CREAR MODELO (FUNCION)
# =========================
def crear_modelo(optimizador):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=optimizador,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# 6. ENTRENAR CON OTROS OPTIMIZADORES
# =========================
optimizadores = ['adam', 'sgd', 'nadam', 'rmsprop', 'adadelta']
resultados = {}
model_adam = None  # Variable para guardar el modelo Adam

for opt in optimizadores:
    print(f"Entrenando con {opt}...")
    modelo = crear_modelo(opt)
    history = modelo.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )
    loss, acc = modelo.evaluate(X, y)
    resultados[opt] = {
        'accuracy': acc,
        'loss': loss,
        'history': history
    }
    if opt == 'adam':  # Guardar el modelo Adam
        model_adam = modelo

# =========================
# 7. AJUSTE DE HIPERPARÁMETROS
# =========================
hiperparametros = [
    {'capas': [128, 64], 'lr': 0.001},
    {'capas': [256, 128, 64], 'lr': 0.0005},
    {'capas': [64], 'lr': 0.01}
]

for config in hiperparametros:
    print(f"Entrenando con configuración: {config}")
    model = Sequential()
    for neurons in config['capas']:
        model.add(Dense(neurons, activation='relu', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=Adam(learning_rate=config['lr']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

# =========================
# 8. MÉTRICAS ADICIONALES
# =========================
y_true = np.argmax(y, axis=1)
predicciones = model_adam.predict(X)  # Usar el modelo Adam
y_pred = np.argmax(predicciones, axis=1)

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_true, y_pred))

# =========================
# 9. GRÁFICAS ADICIONALES
# =========================
plt.figure(figsize=(12, 6))
for opt in optimizadores:
    history = resultados[opt]['history']
    plt.plot(history.history['loss'], label=f'{opt} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{opt} Val Loss')

plt.title("Comparación de Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

# =========================
# 10. CONCLUSIONES
# =========================
print("\nComparación de Resultados:")
mejor_optimizador = None
mejor_accuracy = 0

for opt, res in resultados.items():
    print(f"{opt}: Accuracy={res['accuracy']:.4f}, Loss={res['loss']:.4f}")
    if res['accuracy'] > mejor_accuracy:
        mejor_accuracy = res['accuracy']
        mejor_optimizador = opt

print(f"\nConclusión: El optimizador con mejor rendimiento fue '{mejor_optimizador}' con una precisión de {mejor_accuracy:.4f}.")

# =========================
# 11. MOSTRAR EJEMPLOS DE PREDICCIONES
# =========================
import random

# Seleccionar 10 ejemplos aleatorios
indices = random.sample(range(len(X)), 10)
ejemplos = X.iloc[indices]
predicciones = model_adam.predict(ejemplos)
predicciones_clases = np.argmax(predicciones, axis=1)

# Mostrar las imágenes y sus predicciones
plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(ejemplos.iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicciones_clases[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
