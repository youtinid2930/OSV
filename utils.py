import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def imagePreprocess (imagePath, targetSize=(64, 64)):
    """
    Charge une image, la convertit en niveaux de gris, redimention, et normalization.

    Prametres:
        imagePath type string : chemin vers l'image
        target_size type tuple : taille finale (width, height)

    retour:
        np.array : image pretraiter (float32, taille fixe)
    """


    # 1- Chargement l'image (en couleur par defaut)
    image = cv2.imread(imagePath)

    # 2 - Convertion en niveau de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3 - Redimention
    resized = cv2.resize(gray, targetSize)

    # 4- Normalisation (valeur entre 0 et 1)
    normalized = resized.astype("float32") / 255.0

    processed_image = normalized.reshape(1, targetSize[1], targetSize[0], 1)

    return processed_image



def loadData(directoryPath, targetSize=(64, 64)):
    """
    Parcourir chaque dossier de personne (1, 2, 3 ...), puis 'Forged' et 'Original',
    prétraiter les images et leur attribuer un label.

    Parameters:
        directoryPath (str) : chemin vers les dossiers des personnes
        targetSize (tuple)  : taille finale de l'image (width, height)

    Retour:
        X (np.array) : images prétraitées
        y (np.array) : labels (0=original, 1=forgée)
    """
    X = []
    y = []

    personFolders = [f for f in os.listdir(directoryPath) if os.path.isdir(os.path.join(directoryPath, f))]

    for personFolder in tqdm(personFolders, desc="Chargement des données"):
        personPath = os.path.join(directoryPath, personFolder)

        for labelName in ['Forged', 'Original']:
            classPath = os.path.join(personPath, labelName)

            if not os.path.isdir(classPath):
                continue

            for fileName in os.listdir(classPath):
                if fileName.endswith(('.jpg', '.png', '.jpeg')):
                    filePath = os.path.join(classPath, fileName)

                    try:
                        img = imagePreprocess(filePath, targetSize)

                        # Attribuer un label
                        label = 1 if labelName.lower() == 'forged' else 0

                        X.append(img)
                        y.append(label)

                    except Exception as e:
                        print(f"Erreur en traitant {filePath} : {e}")
                        continue

    X = np.array(X)
    y = np.array(y)
    return X, y

# def loadData (directoryPath, targetSize=(64, 64)):
#     """
#     Parcourir chaque dossier (1, 2, 3 ...) , Parcourir les fichiers d'image, pretraiter, labeling 

#     Parameters:
#         directoryPath type string : chemin vers les dossier des personnes
#         targetSize type tuple : taille final (width, height)

#     retour:
#         np.arrays : X (images), y (labels)
#     """
#     X = []
#     y = []
#     personeFolders = [f for f in os.listdir(directoryPath) if os.path.isdir(os.path.join(directoryPath, f))]
#     # Le Parcours de chaque dossier (1, 2, 3 ...)
#     for personeFolder in tqdm(personeFolders, desc="loading dataset"):
#         personePath = os.path.join(directoryPath, personeFolder)
        
#         if not os.path.isdir(personePath):
#             continue
        
#         # Le Parcours des images de chaque dossier personne
#         for fileName in os.listdir(personePath):
#             if fileName.endswith(".jpg") or fileName.endswith(".png"):
#                 filePath = os.path.join(personePath, fileName)

#                 # Pretraitement l'image
#                 img = imagePreprocess(filePath, targetSize)

#                 # Definition de labels selon le nom de fichier
#                 if "forge" or "-g-" in fileName.lower():
#                     label = 1
#                 elif "original" or "-g-" in fileName.lower():
#                     label = 0
#                 else:
#                     continue

#                 X.append(img)
#                 y.append(label)


#     # Convertion en numpy array 
#     X = np.array(X)
#     y = np.array(y)


#     return X, y        


def augment_image(image):
    """
    Augmentation de jeu d'entrainement pour un image

    Parameters:
        image type any (should be a np.array): image pour augmenter
    retour:
         image : iamge augmenter
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image