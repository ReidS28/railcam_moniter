import multiprocessing
from collections import deque

import cv2
from cv2.typing import MatLike
from utils import print_time

# Following this guide
# https://hackmd.io/@lKuOpplzSUWLhLim2Z7ZJw/BkGj66aW3

MOG2_HISTORY = 300

MOG2_VAR_THRESHOLD = 32
BLUR_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 250
MIN_TOTAL_MOVING_AREA = 1500

MOVEMENT_START_FRAMES = 6
MOVEMENT_END_FRAMES = 20
MAX_EVENT_SECONDS = 40

PRE_ROLL_FRAMES = 60


def _worker(frame_queue: multiprocessing.JoinableQueue[MatLike | None]):

    rolling_storage: deque[MatLike | None] = deque(
        [None] * PRE_ROLL_FRAMES, maxlen=PRE_ROLL_FRAMES
    )

    cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=True,
    )
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    while True:
        frame = frame_queue.get()

        if frame is None:
            frame_queue.task_done()
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

        fgmask = fgbg.apply(gray)

        fgmask = fgbg.apply(frame)
        _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph_kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph_kernel)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
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
