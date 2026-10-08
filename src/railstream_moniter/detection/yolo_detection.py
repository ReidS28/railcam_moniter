import multiprocessing
import queue
import threading
from collections import deque
from collections.abc import Callable, Iterator
from queue import Queue
from typing import cast
from uuid import UUID

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils import DetectionResults, IndexedFrame, get_frame_from_id, print_time

MAX_QUEUE_SIZE = 40

DETECTION_TARGET = "TRAIN"
DETECTION_CONFIDENCE_THRESHOLD = 0.3

START_DETECTION_FRAMES = 4
END_DETECTION_FRAMES = 10

BoundingBoxCallback = Callable[
    [bool, float, float, float, float, str],
    None,
]


def _worker(
    rolling_storage: deque[IndexedFrame],
    id_queue: queue.Queue[UUID],
    model: YOLO,
    event_queue: multiprocessing.Queue,
    bounding_box_callback: BoundingBoxCallback,
):

    target_detected = False
    detection_frames = 0

    while True:
        try:
            target_id = id_queue.get(timeout=0.03)
            indexed_frame: IndexedFrame | None = get_frame_from_id(
                rolling_storage, target_id
            )
            if indexed_frame is None:
                continue

        except queue.Empty:
            cv2.waitKey(1)
            continue

        if indexed_frame is None:
            id_queue.task_done()
            break

        results = cast(
            Iterator[Results],
            model.predict(
                indexed_frame.frame,
                stream=True,
                verbose=False,
                conf=DETECTION_CONFIDENCE_THRESHOLD,
                imgsz=640,  # this number seems to work best (It's what it was trained at)
                device=0,
            ),
        )

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            currently_detected = False

            target_boxes = [
                box
                for box in boxes
                if model.names[int(box.cls[0])].upper() == DETECTION_TARGET
            ]

            if target_boxes:
                best_box = max(target_boxes, key=lambda b: float(b.conf[0]))
                conf = float(best_box.conf[0])

                currently_detected = True

                # Draw Box
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                frame_height, frame_width = indexed_frame.frame.shape[:2]
                label = f"{DETECTION_TARGET} {conf:.2f}"

                indexed_frame.detection_results = DetectionResults(
                    True, x1, y1, x2, y2, conf
                )

                bounding_box_callback(
                    True,
                    x1 / frame_width * 100,
                    y1 / frame_height * 100,
                    (x2 - x1) / frame_width * 100,
                    (y2 - y1) / frame_height * 100,
                    label,
                )
            else:
                indexed_frame.detection_results = DetectionResults(False, 0, 0, 0, 0, 0)
                bounding_box_callback(False, 0, 0, 0, 0, "")

            if currently_detected:
                detection_frames += 1
            elif target_detected:
                detection_frames -= 1
            else:
                detection_frames = 0

            detection_frames = max(
                -END_DETECTION_FRAMES, min(START_DETECTION_FRAMES, detection_frames)
            )

            if detection_frames >= START_DETECTION_FRAMES:
                if not target_detected:
                    target_detected = True
                    event_queue.put({"type": "target_detected_started"})

            elif detection_frames <= -END_DETECTION_FRAMES:
                if target_detected:
                    target_detected = False
                    event_queue.put({"type": "target_detected_ended"})

            elif id_queue.empty() and not target_detected and not currently_detected:
                event_queue.put({"type": "done_detecting_queue"})

        id_queue.task_done()


class YOLODectection:
    def __init__(
        self,
        event_queue: multiprocessing.Queue,
        rolling_storage: deque[IndexedFrame],
        run_nth_frame: int = 1,
        bounding_box_callback: BoundingBoxCallback = (lambda *_: None),
    ):
        self.event_queue = event_queue
        self.rolling_storage = rolling_storage
        self.run_nth_frame = run_nth_frame
        self.bounding_box_callback = bounding_box_callback

        self._frame_count = 0
        self._id_queue: Queue[UUID] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._model = YOLO("yolo26x.pt")  # Consider changing from pytorch to TensorRT

        thread = threading.Thread(
            target=_worker,
            args=(
                self.rolling_storage,
                self._id_queue,
                self._model,
                self.event_queue,
                self.bounding_box_callback,
            ),
            name="YOLO",
            daemon=True,
        )
        thread.start()

    def add_frame_id(self, id: UUID):
        if self._frame_count == 0:
            # DEBUG
            if self._id_queue.full():
                print_time(
                    f"ID added: {self._id_queue.qsize()}", (109, 0, 128)
                )  # purple
            self._id_queue.put(id)

        self._frame_count += 1
        if self._frame_count >= self.run_nth_frame:
            self._frame_count = 0
