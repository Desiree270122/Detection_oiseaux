
import os
import shutil
import random
from PIL import Image
import numpy as np

def create_directory_structure():
    """
    Crée la structure de dossiers nécessaire pour le projet
    """
    dirs = [
        'dataset/train/labels_label_perroquet',
        'dataset/train/labels_label_aigle',
        'dataset/train/labels_label-colombe',
        'dataset/train/labels_label-du-corbo',
        'dataset/train/labels_label-du-hibou',
        'dataset/train/labels_label_calao',
        'dataset/val/labels_label_perroquet',
        'dataset/val/labels_label_aigle',
        'dataset/val/labels_label-colombe',
        'dataset/val/labels_label-du-corbo',
        'dataset/val/labels_label-du-hibou',
        'dataset/val/labels_label_calao',
        'dataset/test/labels_label_perroquet',
        'dataset/test/labels_label_aigle',
        'dataset/test/labels_label-colombe',
        'dataset/test/labels_label-du-corbo',
        'dataset/test/labels_label-du-hibou',
        'dataset/test/labels_label_calao'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Dossier créé : {dir_path}")

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Divise les images d'un dossier source en ensembles d'entraînement, validation et test
    """
    if not os.path.exists(source_dir):
        print(f"Le dossier source {source_dir} n'existe pas.")
        return
        
    # Obtenir tous les fichiers d'images
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Mélanger aléatoirement les fichiers
    random.shuffle(files)
    
    # Calculer les indices de division
    train_end = int(len(files) * split_ratio[0])
    val_end = train_end + int(len(files) * split_ratio[1])
    
    # Diviser les fichiers
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    # Copier les fichiers vers les répertoires correspondants
    for file_list, dest_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for file in file_list:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(dest_dir, file)
            shutil.copy2(src_path, dst_path)
    
    print(f"Division terminée pour {source_dir}:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")

def main():
    """
    Fonction principale pour préparer les données
    """
    # Créer la structure de dossiers
    create_directory_structure()
    
    # Définir les chemins des dossiers sources et destinations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "LABEL_FINAL")
    
    # Liste des classes d'oiseaux
    bird_classes = {
        "labels_label_perroquet": "labels_label_perroquet",
        "labels_calao_bird": "labels_label_calao",
        "labels_label_aigle": "labels_label_aigle",
        "labels_label-colombe": "labels_label-colombe",
        "labels_label-corbo": "labels_label-du-corbo",
        "labels_label-du-hibou": "labels_label-du-hibou"
    }
    
    # Diviser les données pour chaque classe
    for src_folder, dst_folder in bird_classes.items():
        source_dir = os.path.join(data_dir, src_folder)
        train_dir = os.path.join(base_dir, "dataset", "train", dst_folder)
        val_dir = os.path.join(base_dir, "dataset", "val", dst_folder)
        test_dir = os.path.join(base_dir, "dataset", "test", dst_folder)
        
        split_data(source_dir, train_dir, val_dir, test_dir)
    
    print("Préparation des données terminée!")

if __name__ == "__main__":
    main()