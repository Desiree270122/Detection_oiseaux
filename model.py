
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

def create_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Création du modèle de classification d'oiseaux basé sur MobileNetV2
    
    Args:
        input_shape: Dimensions des images d'entrée (hauteur, largeur, canaux)
        num_classes: Nombre de classes à prédire
        
    Returns:
        model: Modèle compilé prêt pour l'entraînement
    """
    # Charger le modèle pré-entraîné sans couche de classification
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Geler les couches du modèle de base
    base_model.trainable = False
    
    # Créer le modèle complet
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compiler le modèle
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def unfreeze_model(model, num_layers=20):
    """
    Dégèle les dernières couches du modèle de base pour le fine-tuning
    
    Args:
        model: Modèle à modifier
        num_layers: Nombre de couches à dégeler depuis la fin
        
    Returns:
        model: Modèle modifié
    """
    # Accéder au modèle de base (MobileNetV2)
    base_model = model.layers[0]
    
    # Dégeler les dernières couches
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    
    # Recompiler avec un taux d'apprentissage plus faible
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model