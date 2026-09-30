import torch
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None

from src.utils import logger
from src.preprocessing import get_val_transforms
from src.config import config


def _get_cmap(name):
    """Compat wrapper — matplotlib 3.9+ removed cm.get_cmap."""
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return plt.get_cmap(name)


class GradCAMExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transforms = get_val_transforms(config["training"]["image_size"])

        if GradCAM is not None:
            try:
                target_layers = [model.backbone.features[-1]]
                self.cam = GradCAM(model=model, target_layers=target_layers)
            except AttributeError:
                logger.error("Could not find target layers for Grad-CAM.")
                self.cam = None
        else:
            self.cam = None
            logger.error("grad-cam library not installed.")

    def generate_heatmap(self, face_image_pil, target_class=1):
        """
        Generate a Grad-CAM heatmap for the given face image.
        target_class: 1 for DEEPFAKE, 0 for REAL.
        Returns a PIL Image with the heatmap overlaid.
        """
        if self.cam is None:
            return face_image_pil

        img_size = config["training"]["image_size"]
        rgb_img = face_image_pil.resize((img_size, img_size)).convert("RGB")
        rgb_arr = np.clip(np.float32(rgb_img) / 255.0, 0.0, 1.0)

        input_tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)
        targets = [ClassifierOutputTarget(target_class)]

        try:
            # GradCAM needs gradients through the conv layers.
            # The backbone was frozen for training; temporarily enable grads here.
            for param in self.model.backbone.features.parameters():
                param.requires_grad_(True)

            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

            if grayscale_cam is None:
                raise ValueError("GradCAM returned None.")

            grayscale_cam = grayscale_cam[0, :]  # (H, W)
            visualization = show_cam_on_image(rgb_arr, grayscale_cam, use_rgb=True)
            return Image.fromarray(visualization)

        except Exception as e:
            logger.warning(f"Grad-CAM failed ({e}), using colormap fallback.")

        finally:
            # Re-freeze backbone to restore original training state
            for param in self.model.backbone.features.parameters():
                param.requires_grad_(False)

        # ── Fallback: jet colormap blended over the face ──
        try:
            gray = np.array(rgb_img.convert("L")).astype(np.float32) / 255.0
            cmap    = _get_cmap("jet")
            heatmap = np.array(cmap(gray))[:, :, :3]   # (H, W, 3) float
            blend   = np.clip(rgb_arr * 0.55 + heatmap * 0.45, 0.0, 1.0)
            return Image.fromarray((blend * 255).astype(np.uint8))
        except Exception as e2:
            logger.error(f"Grad-CAM fallback also failed: {e2}")
            return face_image_pil
