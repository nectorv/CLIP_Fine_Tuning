import os
import webdataset as wds
import torch
from torchvision import transforms
from PIL import Image

def identity(x):
    return x

def get_wds_pipeline(
    data_dir, 
    processor, 
    transform=None, 
    is_train=True, 
    batch_size=64,
    epoch_len=10000  # Nombre d'échantillons estimé pour définir la taille d'une époque
):
    """
    Crée un DataLoader basé sur WebDataset.
    """
    # 1. Trouver les shards (fichiers .tar)
    # data_dir ressemble à /tmp/data/train
    shards = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tar')]
    
    if not shards:
        raise FileNotFoundError(f"❌ Aucun fichier .tar trouvé dans {data_dir}")
        
    print(f"📦 Trouvé {len(shards)} shards dans {data_dir}")

    # 2. Définir le pipeline de transformation
    def preprocess_sample(sample):
        # sample est un tuple (image_pil, text_str) venant de .to_tuple("jpg", "txt")
        image, text = sample
        
        # Application de l'augmentation (si définie)
        if transform:
            image = transform(image)
        
        # Le processeur CLIP gère la normalisation et la tokenization
        # On retourne directement les tenseurs
        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0)
        }

    # 3. Construire le dataset WebDataset
    # nodesplitter=wds.split_by_node permet le multi-GPU plus tard si besoin
    dataset = wds.WebDataset(shards, nodesplitter=wds.split_by_node)
    
    if is_train:
        # Mélange important pour l'entraînement
        dataset = dataset.shuffle(1000)
    
    dataset = (
        dataset
        .decode("pil")          # Convertit automatiquement les bytes "jpg" en PIL Image
        .to_tuple("jpg", "txt") # Extrait seulement l'image et le prompt (ignore le json pour l'instant)
        .map(preprocess_sample) # Applique CLIP Processor
    )

    # 4. Batching
    # WebDataset gère mieux le batching s'il est fait dans le pipeline ou via le DataLoader standard.
    # Ici, nous allons utiliser le DataLoader standard de PyTorch pour le batching final,
    # car cela facilite l'intégration avec `accelerate` ou `bitsandbytes`.
    # Cependant, nous devons ajouter une longueur artificielle pour que tqdm fonctionne.
    
    dataset = dataset.with_length(epoch_len)
    
    return dataset

def get_transforms(aug_type="simple"):
    # (Identique à avant, mais assurez-vous de retourner une PIL Image à la fin pour le processor)
    if aug_type == "simple":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
        ])
    elif aug_type == "advanced":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(), 
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            transforms.ToPILImage() 
        ])
    return None