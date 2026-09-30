import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from pathlib import Path
from tqdm import tqdm

from src.config import config
from src.utils import logger
from src.preprocessing import get_val_transforms
from src.dataset import DeepfakeDataset
from src.model import DeepfakeClassifier
from src.train import get_device

def evaluate_model():
    device = get_device()
    logger.info(f"Evaluating on device: {device}")
    
    test_dir = config["dataset"]["test_path"]
    image_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    
    test_transforms = get_val_transforms(image_size)
    test_dataset = DeepfakeDataset(test_dir, transform=test_transforms)
    
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty. Please check the data directories.")
        return
        
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )
    
    model = DeepfakeClassifier(num_classes=config["model"]["num_classes"])
    best_model_path = config["training"]["best_model_path"]
    
    if not Path(best_model_path).exists():
        logger.error(f"Model checkpoint not found at {best_model_path}")
        return
        
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of FAKE class
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0 # Handle case where only one class is present in test set
        
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["REAL", "DEEPFAKE"], zero_division=0))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["REAL", "DEEPFAKE"], yticklabels=["REAL", "DEEPFAKE"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    cm_dir = Path("outputs/confusion_matrix")
    cm_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_dir / "confusion_matrix.png")
    plt.close()
    
    # ROC Curve Plot
    if roc_auc > 0:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        graphs_dir = Path("outputs/graphs")
        graphs_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(graphs_dir / "roc_curve.png")
        plt.close()

if __name__ == "__main__":
    evaluate_model()
