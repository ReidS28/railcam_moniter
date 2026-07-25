import multiprocessing

import cv2
from cv2.typing import MatLike
from utils import print_time


def _worker(frame_queue: multiprocessing.JoinableQueue[MatLike | None]):
    previous_frame: MatLike | None = None

    cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

    while True:
        frame = frame_queue.get()

        if frame is None:
            frame_queue.task_done()
            break

        if previous_frame is not None:
            diff = cv2.absdiff(previous_frame, frame)
            diff_sum = diff.sum()
            cv2.imshow("Movement", diff)
            if diff_sum > 2e7:
                print_time(f"[Worker] Diff sum: {diff_sum}")
        else:
            print_time("[Worker] First frame, nothing to compare yet.")

        cv2.waitKey(1)
        previous_frame = frame

        frame_queue.task_done()


class MovementDetection:
    def __init__(self) -> None:
        self._frame_count = 0

        self._frame_queue: multiprocessing.JoinableQueue[MatLike | None] = (
            multiprocessing.JoinableQueue()
        )

        self.worker = multiprocessing.Process(
            target=_worker, args=(self._frame_queue,), daemon=True
        )
        self.worker.start()

    def add_frame(self, frame: MatLike):
        if self._frame_count == 0:
            print_time(f"Frame added: {self._frame_queue.qsize()}")
            self._frame_queue.put(frame)

        self._frame_count += 1
        if self._frame_count >= 1:
            self._frame_count = 0

    def shutdown(self):
        self._frame_queue.join()
        self._frame_queue.put(None)
