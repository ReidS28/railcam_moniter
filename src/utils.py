import os
import json
from datetime import datetime
from termcolor import colored

def load_config(file_path):
    """Loads a JSON configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def print_time(message, color='white', highlight_color=None, attrs=[]):
    """
    Print a message with timestamp in color
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strings = message.split("\n")
    for string in strings:
        print(colored(f"[{timestamp}]", 'blue') + colored(f" {string}", color, highlight_color, attrs))