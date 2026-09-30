import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:
    pass

class DeepfakeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(DeepfakeClassifier, self).__init__()
        
        # Load EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
            
        # Freeze backbone parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, num_layers=None):
        """
        Unfreeze the backbone for fine-tuning.
        If num_layers is None, unfreeze all.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
