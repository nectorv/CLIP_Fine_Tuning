import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import ModelConfig

class FurnitureDataset(Dataset):
    def __init__(self, data_dir, processor, transform=None, use_txt_captions=True):
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform
        self.use_txt_captions = use_txt_captions

        # Load all image paths
        self.image_paths = self._collect_image_paths(data_dir)

        # Extract text labels from sidecar .txt files when present
        self.captions = [self._get_label_from_path(p) for p in self.image_paths]

    def _collect_image_paths(self, data_dir):
        image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(("jpg", "jpeg", "png")):
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)

    def _get_label_from_path(self, path):
        # Prefer per-image captions in sidecar .txt files
        if self.use_txt_captions:
            txt_path = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as handle:
                        caption = handle.read().strip()
                    if caption:
                        return caption
                except Exception:
                    pass

        # Fallback: use folder name
        folder_name = os.path.basename(os.path.dirname(path))
        return folder_name.replace("_", " ")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback for corrupted images
            image = Image.new("RGB", (ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
        
        # Apply Data Augmentation
        if self.transform:
            image = self.transform(image)
            
        # Processor handles normalization and tokenization
        # Note: CLIP expects specific inputs. 
        # We assume transform output is a PIL Image, processor does ToTensor+Norm
        inputs = self.processor(
            text=[caption], 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

def get_transforms(aug_type="simple"):
    if aug_type == "simple":
        return transforms.Compose([
            transforms.Resize((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE)), # Resize first
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(ModelConfig.IMAGE_SIZE, padding=4),
            # Note: ToTensor/Normalize is done by CLIPProcessor usually, 
            # but if using Augmentation, we often pass PIL to Processor
        ])
    elif aug_type == "advanced":
        return transforms.Compose([
            transforms.Resize((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(), # Required for RandomErasing
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            transforms.ToPILImage() # Convert back so Processor can handle it
        ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((ModelConfig.IMAGE_SIZE, ModelConfig.IMAGE_SIZE))
    ])