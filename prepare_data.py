import os
import shutil
import random

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

def list_available_directories():
    """
    Liste tous les répertoires disponibles dans le dossier courant
    """
    current_dir = os.getcwd()
    print(f"Dossier courant : {current_dir}")
    
    all_items = os.listdir(current_dir)
    directories = [d for d in all_items if os.path.isdir(os.path.join(current_dir, d))]
    
    print("Répertoires disponibles :")
    for d in directories:
        print(f"  - {d}")
    
    return directories

def split_data_from_directories(directories, split_ratio=(0.7, 0.15, 0.15)):
    """
    Répartit les images depuis les répertoires spécifiés
    """
    current_dir = os.getcwd()
    
    # Filtrer pour ne conserver que les dossiers qui contiennent potentiellement des images d'oiseaux
    bird_dirs = [d for d in directories if 'label' in d.lower() or 'bird' in d.lower() or 'oiseau' in d.lower()]
    
    if not bird_dirs:
        print("Aucun dossier d'oiseaux trouvé. Utilisation de tous les dossiers disponibles.")
        bird_dirs = directories
    
    print(f"\nTraitement des dossiers suivants :")
    for d in bird_dirs:
        print(f"  - {d}")
    
    for subdir in bird_dirs:
        source_dir = os.path.join(current_dir, subdir)
        
        # Vérifier si c'est un fichier zip
        if subdir.lower().endswith('.zip'):
            print(f"{subdir} est un fichier zip, ignoré.")
            continue
            
        # Obtenir tous les fichiers d'images
        try:
            files = [f for f in os.listdir(source_dir) 
                    if os.path.isfile(os.path.join(source_dir, f)) and 
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except:
            print(f"Impossible de lire le contenu de {subdir}, ignoré.")
            continue
        
        if not files:
            print(f"Aucune image trouvée dans {subdir}, ignoré.")
            continue
        
        print(f"\nTraitement du dossier {subdir} - {len(files)} images trouvées")
        
        # Mélanger aléatoirement les fichiers
        random.shuffle(files)
        
        # Calculer les indices de division
        train_end = int(len(files) * split_ratio[0])
        val_end = train_end + int(len(files) * split_ratio[1])
        
        # Diviser les fichiers
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # Déterminer le nom du dossier de destination en fonction du nom du dossier source
        dest_folder_name = subdir
        
        # Créer les dossiers de destination
        train_dir = os.path.join(current_dir, 'dataset', 'train', dest_folder_name)
        val_dir = os.path.join(current_dir, 'dataset', 'val', dest_folder_name)
        test_dir = os.path.join(current_dir, 'dataset', 'test', dest_folder_name)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Copier les fichiers vers les répertoires correspondants
        for file_list, dest_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for file in file_list:
                src_path = os.path.join(source_dir, file)
                dst_path = os.path.join(dest_dir, file)
                try:
                    shutil.copy2(src_path, dst_path)
                except:
                    print(f"Impossible de copier {file} vers {dest_dir}")
        
        print(f"Division terminée pour {subdir}:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Validation: {len(val_files)} images")
        print(f"  Test: {len(test_files)} images")

def main():
    """
    Fonction principale pour préparer les données
    """
    # Créer la structure de dossiers de base
    create_directory_structure()
    
    # Lister les répertoires disponibles
    directories = list_available_directories()
    
    if not directories:
        print("Aucun répertoire trouvé dans le dossier courant.")
        return
    
    # Répartir les données
    split_data_from_directories(directories)
    
    print("\nPréparation des données terminée!")

if __name__ == "__main__":
    main()