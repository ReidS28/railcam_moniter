import time

from web_player import webPlayer

player = webPlayer()

try:
    while True:
        time.sleep(1) 
        # player.skip_ad()
except KeyboardInterrupt:
    player.end()


