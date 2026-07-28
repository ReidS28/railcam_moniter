import multiprocessing
import queue
import threading

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO
from utils import print_time

MAX_QUEUE_SIZE = 40

DETECTION_TARGET = "TRAIN"


def _worker(
    frame_queue: queue.Queue[MatLike | None],
    model: YOLO,
    event_queue: multiprocessing.Queue,
):

    cv2.namedWindow("Movement", cv2.WINDOW_NORMAL)

    while True:
        frame = frame_queue.get()

        if frame is None:
            frame_queue.task_done()
            break

        results = model.predict(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                if class_name.upper() != DETECTION_TARGET:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Movement", frame)
        cv2.waitKey(1)

        frame_queue.task_done()


class YOLOProcessor:
    def __init__(self, event_queue: multiprocessing.Queue, run_nth_frame=1):
        self._frame_count = 0
        self.run_nth_frame = run_nth_frame
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        model = YOLO("yolo26n.pt")  # Consider changing from pytorch to TensorRT

        thread = threading.Thread(
            target=_worker,
            args=(self._frame_queue, model, event_queue),
            name="YOLO",
            daemon=True,
        )
        thread.start()

    def add_frame(self, frame: MatLike):
        if self._frame_count == 0:
        # DEBUG
            if self._frame_queue.qsize() > 2:
                print_time(
                    f"Frame added: {self._frame_queue.qsize()}", (109, 0, 128)
                )  # purple
            self._frame_queue.put(frame)

        self._frame_count += 1
        if self._frame_count >= self.run_nth_frame:
            self._frame_count = 0
