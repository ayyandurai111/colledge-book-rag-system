import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# Resolve config path relative to project root (parent of this file's package)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "default.yaml")


def load_config(path: str = None) -> dict:
    path = path or _DEFAULT_CONFIG
    with open(path, "r") as f:
        return yaml.safe_load(f)


_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_nvidia_api_key() -> str:
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY not set in environment")
    return key
