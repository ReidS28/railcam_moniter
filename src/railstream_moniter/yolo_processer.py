import threading
import cv2
from cv2.typing import MatLike
from utils import print_time
import queue

MAX_QUEUE_SIZE = 10

def _worker(frame_queue: queue.Queue[MatLike | None]):
    frame_queue.get()

class YOLOProcessor:
    def __init__(self):
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        thread = threading.Thread()

    def add_frame(self, frame: MatLike):
        # DEBUG
        if self._frame_queue.qsize() > 2:
            print_time(f"Frame added: {self._frame_queue.qsize()}", (255, 0, 255)) # purple
        self._frame_queue.put(frame)

