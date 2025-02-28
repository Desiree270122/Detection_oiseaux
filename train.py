import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from model import create_model, unfreeze_model
import numpy as np

# Paramètres d'entraînement
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10
CLASSES = 6

def train_model():
    """
    Fonction principale pour entraîner le modèle
    """
    # Création des générateurs d'images avec augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    # Pour validation et test, uniquement la normalisation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Charger les données
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        'dataset/val',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Vérifier si le dossier pour sauvegarder les modèles existe
    os.makedirs('models', exist_ok=True)
    
    # Callbacks pour l'entraînement
    checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Créer le modèle
    model = create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=CLASSES)
    
    # Afficher le résumé du modèle
    model.summary()
    
    # Première phase d'entraînement
    print("Phase 1: Entraînement avec base model gelé")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Sauvegarder le modèle après la première phase
    model.save('models/phase1_model.h5')
    
    # Dégeler certaines couches pour le fine-tuning
    model = unfreeze_model(model)
    
    print("Phase 2: Fine-tuning")
    history_fine = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )
    
    # Sauvegarder le modèle final
    model.save('models/final_model.h5')
    
    # Évaluer sur l'ensemble de test
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Précision sur l'ensemble de test: {test_acc:.4f}")
    
    # Tracer les courbes d'apprentissage
    plot_history(history, history_fine)
    
    return model

def plot_history(history, history_fine=None):
    """
    Trace les courbes d'apprentissage
    """
    plt.figure(figsize=(12, 6))
    
    # Plot de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train (Phase 1)')
    plt.plot(history.history['val_accuracy'], label='Validation (Phase 1)')
    
    if history_fine is not None:
        # Ajout du fine-tuning au plot
        offset = len(history.history['accuracy'])
        plt.plot(
            np.arange(offset, offset + len(history_fine.history['accuracy'])),
            history_fine.history['accuracy'], 
            label='Train (Phase 2)'
        )
        plt.plot(
            np.arange(offset, offset + len(history_fine.history['val_accuracy'])),
            history_fine.history['val_accuracy'], 
            label='Validation (Phase 2)'
        )
    
    plt.title('Précision du modèle')
    plt.ylabel('Précision')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train (Phase 1)')
    plt.plot(history.history['val_loss'], label='Validation (Phase 1)')
    
    if history_fine is not None:
        # Ajout du fine-tuning au plot
        offset = len(history.history['loss'])
        plt.plot(
            np.arange(offset, offset + len(history_fine.history['loss'])),
            history_fine.history['loss'], 
            label='Train (Phase 2)'
        )
        plt.plot(
            np.arange(offset, offset + len(history_fine.history['val_loss'])),
            history_fine.history['val_loss'], 
            label='Validation (Phase 2)'
        )
    
    plt.title('Perte du modèle')
    plt.ylabel('Perte')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

if __name__ == "__main__":
    train_model()