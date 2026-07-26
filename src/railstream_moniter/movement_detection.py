import multiprocessing
from collections import deque

import cv2
import numpy as np
from cv2.typing import MatLike
from utils import print_time

# Following this guide
# https://hackmd.io/@lKuOpplzSUWLhLim2Z7ZJw/BkGj66aW3

MIN_CONTOUR_AREA = 500
PIXEL_DIFF_THRESHOLD = 40


def _worker(frame_queue: multiprocessing.JoinableQueue[MatLike | None]):

    n = 60
    rolling_storage: deque[MatLike | None] = deque([None] * n, maxlen=n)

    cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        frame = frame_queue.get()

        if frame is None:
            frame_queue.task_done()
            break

        fgmask = fgbg.apply(frame)
        th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(
            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Movement", frame)

        cv2.waitKey(1)

        frame_queue.task_done()


class MovementDetection:
    def __init__(self, run_nth_frame=1) -> None:

        self.run_nth_frame = run_nth_frame
        self._frame_count = 0

        self._background = None

        self._frame_queue: multiprocessing.JoinableQueue[MatLike | None] = (
            multiprocessing.JoinableQueue(maxsize=10)
        )

        self.worker = multiprocessing.Process(
            target=_worker, args=(self._frame_queue,), daemon=True
        )
        self.worker.start()

    def add_frame(self, frame: MatLike):
        if self._frame_count == 0:
            if self._frame_queue.qsize() > 2:
                print_time(f"Frame added: {self._frame_queue.qsize()}")
            self._frame_queue.put(frame)

        self._frame_count += 1
        if self._frame_count >= self.run_nth_frame:
            self._frame_count = 0

    def shutdown(self):
        self._frame_queue.join()
        self._frame_queue.put(None)
