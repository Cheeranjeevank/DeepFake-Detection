import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.utils import logger

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.class_to_idx = {"real": 0, "fake": 1}
        
        for class_name in self.class_to_idx.keys():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.glob("*.[jp][pn]*g"): # match .jpg, .jpeg, .png
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
                
        logger.info(f"Loaded {len(self.image_paths)} images from {data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Create a blank image as fallback
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
