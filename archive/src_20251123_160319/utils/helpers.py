"""Helper utilities"""

import os
import json
from datetime import datetime


def create_experiment_folder(experiment_name):
    """Create folder for experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{experiment_name}_{timestamp}"
    folder_path = os.path.join('experiments', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def save_config(config, folder_path):
    """Save experiment configuration"""
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


def load_config(folder_path):
    """Load experiment configuration"""
    config_path = os.path.join(folder_path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
