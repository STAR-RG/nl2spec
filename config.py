from pathlib import Path
import yaml


def load_config(config_path: str):
    """
    Load YAML config from an explicit path.
    The path is interpreted relative to the project root.
    """
    p = Path(config_path)

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
