import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from utils import loadData, augment_image
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
import os

# === 1. Chargement des données ===
data_path = "./data"
X, y = loadData(data_path, targetSize=(50, 50))

# === 2. Séparer en train / val / test ===
# 10% test, puis 20% val du reste
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# Appliquer l'augmentation sur chaque image du jeu d'entraînement
X_train_aug = tf.stack([augment_image(tf.convert_to_tensor(img)) for img in X_train])


X_train_aug = tf.concat([augment_image(tf.convert_to_tensor(img)) for img in X_train], axis=0)

# === 3. Définir le modèle CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# === 4. Compilation du modèle ===
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.summary()

# === 5. Entraînement ===
model.fit(
    X_train_aug, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[TqdmCallback(verbose=1)]
)

# === 6. Évaluation sur le test set ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# === 7. Sauvegarde du modèle ===
os.makedirs("model", exist_ok=True)
model.save("model/modeleVerifierSignature0.h5")
