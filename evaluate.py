import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

# Paramètres
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def evaluate_model(model_path='models/final_model.h5'):
    """
    Évalue le modèle sur l'ensemble de test
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
    """
    # Charger le modèle
    model = load_model(model_path)
    
    # Générateur de données pour l'ensemble de test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important pour conserver l'ordre des prédictions
    )
    
    # Évaluer le modèle
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Précision sur l'ensemble de test: {test_acc:.4f}")
    
    # Faire des prédictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Vraies classes
    y_true = test_generator.classes
    
    # Noms des classes
    class_names = list(test_generator.class_indices.keys())
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Afficher la matrice de confusion
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Créer le dossier pour les résultats s'il n'existe pas
    os.makedirs('results', exist_ok=True)
    
    # Sauvegarder la matrice de confusion
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    
    # Exemples d'images mal classées
    plot_misclassified_examples(model, test_generator, y_pred_classes, y_true, class_names)

def plot_misclassified_examples(model, test_generator, y_pred_classes, y_true, class_names, num_examples=5):
    """
    Affiche des exemples d'images mal classées
    
    Args:
        model: Modèle à évaluer
        test_generator: Générateur de l'ensemble de test
        y_pred_classes: Classes prédites
        y_true: Vraies classes
        class_names: Noms des classes
        num_examples: Nombre d'exemples à afficher
    """
    # Trouver les indices des exemples mal classés
    misclassified_indices = np.where(y_pred_classes != y_true)[0]
    
    if len(misclassified_indices) == 0:
        print("Aucune image mal classée trouvée.")
        return
    
    # Limiter le nombre d'exemples
    num_examples = min(num_examples, len(misclassified_indices))
    
    # Sélectionner aléatoirement des exemples
    selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    
    # Récupérer les images
    test_generator.reset()
    all_images = []
    all_labels = []
    
    for i in range(len(test_generator)):
        images, labels = test_generator.next()
        all_images.append(images)
        all_labels.append(labels)
        if len(all_images) * BATCH_SIZE >= len(y_true):
            break
    
    all_images = np.vstack(all_images)
    
    # Afficher les exemples mal classés
    plt.figure(figsize=(15, 3 * num_examples))
    
    for i, idx in enumerate(selected_indices):
        plt.subplot(num_examples, 1, i + 1)
        
        # L'index dans le batch
        batch_idx = idx % len(all_images)
        plt.imshow(all_images[batch_idx])
        
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred_classes[idx]]
        
        plt.title(f"Vraie classe: {true_class}, Prédiction: {pred_class}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/misclassified_examples.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()