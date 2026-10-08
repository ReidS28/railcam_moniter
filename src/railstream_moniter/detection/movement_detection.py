# -----------------------------
#   THIS CLASS IS NOT USED!!!
# -----------------------------

import multiprocessing

import cv2
from cv2.typing import MatLike
from utils import print_time

# Following this guide
# https://hackmd.io/@lKuOpplzSUWLhLim2Z7ZJw/BkGj66aW3

MOG2_HISTORY = 300

MOG2_VAR_THRESHOLD = 10
DOWNSCALE_FACTOR = 0.5
BLUR_KERNEL_SIZE = 5
MORPH_KERNEL_SIZE = 5
MIN_CONTOUR_AREA = 164

PRE_ROLL_FRAMES = 60


def _worker(
    frame_queue: multiprocessing.JoinableQueue[MatLike | None],
    event_queue: multiprocessing.Queue,
    run_nth_frame: int,
):

    frame_count = -1

    # cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

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

        frame_count += 1
        if frame_count >= run_nth_frame:
            frame_count = 0

        if frame_count != 0:
            continue

        small = cv2.resize(
            frame,
            None,
            fx=DOWNSCALE_FACTOR,
            fy=DOWNSCALE_FACTOR,
            interpolation=cv2.INTER_LINEAR,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

        fgmask = fgbg.apply(gray)

        _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph_kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph_kernel)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Boxes
        movement = False
        scale = 1 / DOWNSCALE_FACTOR
        scaled_area = MIN_CONTOUR_AREA / (scale**2)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > scaled_area:
                movement = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(
                    frame,
                    (int(x * scale), int(y * scale)),
                    (int((x + w) * scale), int((y + h) * scale)),
                    (0, 255, 0),
                    2,
                )

        if movement:
            event_queue.put({"type": "movement"})

        # cv2.imshow("Movement", frame)
        # cv2.waitKey(1)

        frame_queue.task_done()


class MovementDetection:
    def __init__(self, event_queue: multiprocessing.Queue, run_nth_frame=1) -> None:

        self._frame_count = 0

        self._background = None

        self._frame_queue: multiprocessing.JoinableQueue[MatLike | None] = (
            multiprocessing.JoinableQueue(maxsize=10)
        )

        self.worker = multiprocessing.Process(
            target=_worker,
            args=(self._frame_queue, event_queue, run_nth_frame),
            daemon=True,
        )
        self.worker.start()

    def add_frame(self, frame: MatLike):
        if self._frame_queue.full():
            print_time(
                f"Frame added: {self._frame_queue.qsize()}", (255, 128, 0)
            )  # orange
        self._frame_queue.put(frame)

    def shutdown(self):
        self._frame_queue.join()
        self._frame_queue.put(None)
