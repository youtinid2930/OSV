import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from utils import loadData, augment_image
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === 1. Chargement des données ===
data_path = "./data"
X, y = loadData(data_path, targetSize=(50, 50))

# === 2. Séparer en train / val / test ===
# 10% test, puis 20% val du reste
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

# Appliquer l'augmentation sur chaque image du jeu d'entraînement
X_train_aug = tf.stack([augment_image(img) for img in X_train])
X_train_combined = tf.concat([X_train, X_train_aug], axis=0)
y_train_combined = tf.concat([y_train, y_train], axis=0)


print("X_train shape:", X_train.shape)
print("X_train_aug shape:", X_train_aug.shape)
print("X_train_combined shape:", X_train_combined.shape)

# === 3. Définir le modèle CNN ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),

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
    X_train_combined, y_train_combined,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[TqdmCallback(verbose=1)]
)

# === 6. Évaluation sur le test set ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n[INFO] Test Loss: {test_loss:.4f}")
print(f"[INFO] Test Accuracy (from model.evaluate): {test_acc:.4f}")

# === 6.1. Prédictions ===
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()  # Convert probabilities to 0/1

# === 6.2. Métriques ===
print("\n[INFO] Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"MSE:       {mse:.4f}")

# === 6.3. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Original", "Forged"], yticklabels=["Original", "Forged"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === 7. Sauvegarde du modèle ===
os.makedirs("model", exist_ok=True)
model.save("model/modeleVerifierSignature0.h5")
