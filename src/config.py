import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback if running from a different directory (e.g., inside src or app)
        fallback_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(fallback_path, "r") as f:
            return yaml.safe_load(f)

config = load_config()
