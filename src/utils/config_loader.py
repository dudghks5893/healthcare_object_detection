from pathlib import Path
import yaml


def load_config(config_path: str | Path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"config 파일이 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config