import yaml
from yaml.loader import SafeLoader

# Load configuration from file
def load_config(file_path: str = "auth.config.yaml") -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)