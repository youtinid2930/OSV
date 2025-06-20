
from tensorflow import keras

from keras.models import load_model

from utils import loadData


testDr = "./data/Test"


X_test, y_test = loadData(testDr, (50, 50))


trainDr = "./data/Train"


X_train, y_train = loadData(trainDr, (50, 50))



model = load_model("model/modeleVerifierSignature0.h5")

loss, accuracy = model.evaluate(X_test, y_test)

train_loss, train_accuracy = model.evaluate(X_train, y_train)

print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")