import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    try:
        from src.config import config
        from src.utils import logger as src_logger
        from src.preprocessing import get_train_transforms, get_val_transforms
        from src.dataset import DeepfakeDataset
        from src.face_detection import FaceDetector
        from src.model import DeepfakeClassifier
        
        src_logger.info("Local modules imported successfully.")
        src_logger.info(f"Model config: {config['model']['name']}")
        return True
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        sys.exit(0)
    else:
        sys.exit(1)
