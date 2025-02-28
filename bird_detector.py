import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def load_and_prep_image(img_path, img_size=224):
    """
    Charge et prépare une image pour la prédiction
    
    Args:
        img_path: Chemin vers l'image à préparer
        img_size: Taille de redimensionnement de l'image
        
    Returns:
        img: Image originale pour affichage
        img_array: Image préparée pour prédiction
    """
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation
    return img, img_array

def detect_bird(model_path, img_path):
    """
    Détecte et identifie l'espèce d'oiseau sur une image
    
    Args:
        model_path: Chemin vers le modèle entraîné
        img_path: Chemin vers l'image à analyser
        
    Returns:
        class_name: Classe d'oiseau prédite
        confidence: Score de confiance pour la prédiction
    """
    # Charger le modèle
    model = load_model(model_path)
    
    # Classes d'oiseaux
    class_names = [
        "labels_label_perroquet",
        "labels_label_aigle",
        "labels_label-colombe",
        "labels_label-du-corbo",
        "labels_label-du-hibou",
        "labels_label_calao"
    ]
    
    # Charger et préparer l'image
    img, img_array = load_and_prep_image(img_path)
    
    # Faire la prédiction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Affichage du résultat
    class_name = class_names[predicted_class]
    print(f"Classe prédite : {class_name}")
    print(f"Confiance : {confidence:.2f}%")
    
    # Afficher l'image avec la prédiction
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prédiction: {class_name}\nConfiance: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    # Afficher toutes les probabilités pour chaque classe
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, predictions[0] * 100)
    plt.xlabel('Classes')
    plt.ylabel('Probabilité (%)')
    plt.title('Probabilités pour chaque classe')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return class_name, confidence

def detect_birds_in_folder(model_path, folder_path):
    """
    Détecte et identifie les oiseaux sur toutes les images d'un dossier
    
    Args:
        model_path: Chemin vers le modèle entraîné
        folder_path: Chemin vers le dossier d'images
    """
    # Charger le modèle
    model = load_model(model_path)
    
    # Classes d'oiseaux
    class_names = [
        "labels_label_perroquet",
        "labels_label_aigle",
        "labels_label-colombe",
        "labels_label-du-corbo",
        "labels_label-du-hibou",
        "labels_label_calao"
    ]
    
    # Lister les fichiers d'images
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                  if os.path.isfile(os.path.join(folder_path, f)) and
                  any(f.lower().endswith(ext) for ext in valid_extensions)]
    
    if not image_files:
        print(f"Aucune image trouvée dans {folder_path}")
        return
    
    # Créer une figure avec plusieurs sous-plots
    n_images = len(image_files)
    fig_rows = int(np.ceil(n_images / 3))
    
    plt.figure(figsize=(15, 5 * fig_rows))
    
    results = []
    
    for i, img_path in enumerate(image_files):
        # Charger et préparer l'image
        img, img_array = load_and_prep_image(img_path)
        
        # Faire la prédiction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Stocker les résultats
        results.append({
            'filename': os.path.basename(img_path),
            'class': class_names[predicted_class],
            'confidence': confidence
        })
        
        # Afficher l'image avec la prédiction
        plt.subplot(fig_rows, 3, i + 1)
        plt.imshow(img)
        plt.title(f"{os.path.basename(img_path)}\nPrédiction: {class_names[predicted_class]}\nConfiance: {confidence:.1f}%")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Résumé des prédictions
    print("\nRésumé des prédictions:")
    for result in results:
        print(f"{result['filename']}: {result['class']} ({result['confidence']:.1f}%)")
    
    return results

def main():
    """
    Fonction principale permettant l'utilisation en ligne de commande
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Détection et classification d\'oiseaux')
    parser.add_argument('--model', default='models/final_model.h5', help='Chemin vers le modèle')
    parser.add_argument('--image', help='Chemin vers l\'image à analyser')
    parser.add_argument('--folder', help='Chemin vers le dossier d\'images à analyser')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Le modèle {args.model} n'existe pas.")
        return
    
    if args.image and os.path.exists(args.image):
        detect_bird(args.model, args.image)
    elif args.folder and os.path.exists(args.folder):
        detect_birds_in_folder(args.model, args.folder)
    else:
        print("Veuillez spécifier une image ou un dossier valide.")
        print("Exemple: python bird_detector.py --image exemple.jpg")
        print("ou: python bird_detector.py --folder dossier_images/")

if __name__ == "__main__":
    main()