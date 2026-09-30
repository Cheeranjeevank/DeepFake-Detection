import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from src.config import config
from src.utils import logger
from src.preprocessing import get_train_transforms, get_val_transforms
from src.dataset import DeepfakeDataset
from src.model import DeepfakeClassifier

def plot_history(history, save_path):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_device():
    pref = config["training"]["device_preference"]
    if pref == "auto" or pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
    if pref == "auto" or pref == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")

def train_model():
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Paths
    train_dir = config["dataset"]["train_path"]
    val_dir = config["dataset"]["val_path"]
    
    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    
    # Transforms
    train_transforms = get_train_transforms(image_size)
    val_transforms = get_val_transforms(image_size)
    
    # Datasets
    train_dataset = DeepfakeDataset(train_dir, transform=train_transforms)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transforms)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Dataset is empty. Please check the data directories.")
        return
        
    # Handle Class Imbalance using WeightedRandomSampler
    class_counts = Counter(train_dataset.labels)
    total_samples = len(train_dataset)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=total_samples, replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )
    
    # Model
    model = DeepfakeClassifier(num_classes=config["model"]["num_classes"])
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    num_epochs = config["training"]["epochs"]
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = config["training"]["best_model_path"]
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    early_stopping_patience = 5
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = 100. * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100. * val_correct / val_total
        
        # Update Scheduler
        scheduler.step(epoch_val_loss)
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
                    f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
                    
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        
        # Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    # Save training history graph
    graphs_dir = Path("outputs/graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, graphs_dir / "training_history.png")
    logger.info(f"Saved training history plot to {graphs_dir / 'training_history.png'}")

if __name__ == "__main__":
    train_model()
