import yaml
from yaml.loader import SafeLoader
import pandas as pd
import time

# Load configuration from file
def load_config(file_path: str = "auth.config.yaml") -> dict:
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)

def stream_text(text: str, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)


def stream_df(df, delay: float = 0.02):
    for i in range(len(df)):
        yield df.iloc[[i]]
        time.sleep(delay)