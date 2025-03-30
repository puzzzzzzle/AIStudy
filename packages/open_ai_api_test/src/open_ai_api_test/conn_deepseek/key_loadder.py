from pathlib import Path
import yaml


def load_key() -> (str, str):
    key_path = Path(__file__).parent / "../../../api_key.yaml"
    key_path = key_path.resolve().absolute()
    key_config = yaml.safe_load(open(key_path))
    api_key = key_config["default"]["api_key"]
    base_url = key_config["default"]["url"]
    model = key_config["default"]["model"]
    return api_key, base_url, model
