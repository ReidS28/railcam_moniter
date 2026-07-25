import time

from cv2.typing import MatLike
from web_player import webPlayer

from utils import print_time

def color_to_hex(color) -> str:
    b, g, r = color
    return f"{r:02x}{g:02x}{b:02x}"

def on_frame(frame: MatLike):
    print_time(f"{frame.mean():.2} : #{color_to_hex(frame[0][0])}")

player = webPlayer(frame_callback=on_frame)

try:
    while True:
        time.sleep(1) 
except KeyboardInterrupt:
    player.end()

