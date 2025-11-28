import os
import json
from datetime import datetime
from termcolor import colored
from enum import Enum

class BoundingBoxColor(Enum):
    RED = (255, 0, 0, 255)      # (R, G, B, A)
    BLUE = (0, 0, 255, 255)
    GREEN = (0, 255, 0, 255)
    YELLOW = (255, 255, 0, 255)

def print_time(message, color='white', highlight_color=None, attrs=[]):
    """
    Print a message with timestamp in color
    """
    message_str = str(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strings = message_str.split("\n")
    for string in strings:
        print(colored(f"[{timestamp}]", 'blue') + colored(f" {string}", color, highlight_color, attrs))