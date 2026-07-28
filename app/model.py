"""
Chargement du ResNet50V2 entraine et prediction sur une image.

Deux points meritent une explication, parce qu'ils ne sont pas evidents et que
s'en ecarter fait chuter le modele au niveau du hasard.

1. LE PRETRAITEMENT
   Le modele a ete entraine avec ImageDataGenerator(rescale=1./255), donc il
   attend des pixels dans [0, 1]. Mesure sur 600 images du jeu de test :
       x / 255           -> 89,2 % d'accuracy, AUC 97,1 %   <- correct
       pixels bruts      -> 50,0 %  (le modele sature a 1,0 sur toute image)
   Ne pas confondre avec resnet_v2.preprocess_input, qui ramene dans [-1, 1] :
   c'est la convention habituelle de ResNet50V2, mais ce n'est pas celle
   utilisee ici.

2. LA CONVENTION DE SORTIE
   flow_from_directory() trie les dossiers par ordre alphabetique, donc
   fake -> 0 et real -> 1. Une sortie sigmoid proche de 1 signifie
   IMAGE REELLE, proche de 0 signifie IMAGE GENEREE.
"""

import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import h5py
import numpy as np
from PIL import Image
import keras
from keras import layers

IMG_SIZE = 224


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_model():
    """Reconstruit l'architecture exacte du modele entraine.

    Base ResNet50V2 (sans sa couche de classification ImageNet, avec pooling
    moyen global), puis la tete ajoutee pendant l'entrainement. Les noms de
    couches doivent correspondre a ceux du .h5, sinon l'injection des poids
    ci-dessous ne retrouve rien.
    """
    base = keras.applications.ResNet50V2(
        include_top=False, weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')

    return keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        base,
        layers.Dense(256, activation='relu', name='dense_6'),
        layers.BatchNormalization(name='batch_normalization_2'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='dense_7'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', name='dense_8'),
    ])


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def _poids_depuis_h5(chemin):
    """Aplatit un .h5 Keras en dictionnaire {couche/variable: tableau}."""
    poids = {}

    def parcourir(groupe, prefixe=''):
        for cle in groupe:
            element = groupe[cle]
            if isinstance(element, h5py.Dataset):
                parties = (prefixe + cle).split('/')
                poids['/'.join(parties[-2:])] = element[()]
            else:
                parcourir(element, prefixe + cle + '/')

    with h5py.File(chemin, 'r') as f:
        parcourir(f['model_weights'])
    return poids


def load_model(chemin):
    """Charge le modele depuis un .keras ou un .h5. Retourne (modele, methode).

    Le chargement direct echoue sur les .h5 produits par Keras 3.8 quand on
    utilise une version plus recente ("ValueError: Invalid dtype: tuple").
    Dans ce cas on reconstruit l'architecture et on y injecte les poids lus
    dans le fichier, ce qui donne un modele strictement identique.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(chemin)

    try:
        return keras.saving.load_model(chemin, compile=False), 'direct'
    except Exception:
        if not chemin.lower().endswith('.h5'):
            raise

    modele = build_model()
    poids = _poids_depuis_h5(chemin)

    charges = manquants = 0
    couches = []
    for couche in modele.layers:
        couches.extend(getattr(couche, 'layers', [couche]))
    for couche in couches:
        for variable in couche.weights:
            cle = f"{couche.name}/{variable.name}"
            if cle in poids and tuple(poids[cle].shape) == tuple(variable.shape):
                variable.assign(poids[cle])
                charges += 1
            else:
                manquants += 1

    if manquants:
        raise RuntimeError(
            f"{manquants} variables n'ont pas pu etre retrouvees dans {chemin}. "
            f"Le fichier ne correspond pas a l'architecture attendue.")

    return modele, f'reconstruction ({charges} variables)'


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def prepare(image):
    """PIL.Image -> tableau (1, 224, 224, 3) avec des pixels dans [0, 1]."""
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(image, dtype='float32')[None, ...] / 255.0


def predict(modele, image):
    """Retourne (label, score, confiance).

    label    : 'real' ou 'fake'
    score    : sortie sigmoid brute, probabilite que l'image soit reelle
    confiance: certitude du modele sur le label retenu, en pourcentage
    """
    score = float(modele.predict(prepare(image), verbose=0).ravel()[0])
    label = 'real' if score > 0.5 else 'fake'
    confiance = score if label == 'real' else 1.0 - score
    return label, score, confiance * 100.0
