import queue
import threading

import multiprocessing

import cv2
from cv2.typing import MatLike
from utils import print_time

MAX_QUEUE_SIZE = 10


def _worker(
    frame_queue: queue.Queue[MatLike | None],
    event_queue: multiprocessing.Queue,
    run_nth_frame: int,
):

    frame_count = -1

    cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

    while True:
        frame = frame_queue.get()

        if frame is None:
            frame_queue.task_done()
            break

        frame_queue.task_done()

        frame_count += 1
        if frame_count >= run_nth_frame:
            frame_count = 0

        if frame_count != 0:
            continue


class YOLOProcessor:
    def __init__(self, event_queue: multiprocessing.Queue,  run_nth_frame=1):
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

        thread = threading.Thread(target=_worker, args=(self._frame_queue, event_queue, run_nth_frame), name="YOLO", daemon=True)
        thread.start()

    def add_frame(self, frame: MatLike):
        # DEBUG
        if self._frame_queue.qsize() > 2:
            print_time(
                f"Frame added: {self._frame_queue.qsize()}", (109, 0, 128)
            )  # purple
        self._frame_queue.put(frame)
