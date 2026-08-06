import multiprocessing
import queue
import threading
from collections.abc import Iterator
from typing import cast

import cv2
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Results
from utils import print_time

MAX_QUEUE_SIZE = 40

DETECTION_TARGET = "TRAIN"
START_DETECTION_CONFIDENCE_THRESHOLD = 0.1

START_DETECTION_FRAMES = 2
END_DETECTION_FRAMES = 10


def _worker(
    frame_queue: queue.Queue[MatLike | None],
    model: YOLO,
    event_queue: multiprocessing.Queue,
):

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    target_detected = False
    detection_frames = 0

    while True:
        try:
            frame = frame_queue.get(timeout=0.03)
        except queue.Empty:
            cv2.waitKey(1)
            continue

        if frame is None:
            frame_queue.task_done()
            break

        results = cast(
            Iterator[Results],
            model.predict(
                frame,
                stream=True,
                verbose=False,
                conf=START_DETECTION_CONFIDENCE_THRESHOLD,
                imgsz=1280,
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

                if conf > START_DETECTION_CONFIDENCE_THRESHOLD:
                    currently_detected = True

                    # Draw Box
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{DETECTION_TARGET} {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

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
                    print_time("Detection Started", "light_green")
            elif detection_frames <= -END_DETECTION_FRAMES:
                if target_detected:
                    target_detected = False
                    event_queue.put({"type": "target_detected_ended"})
                    print_time("Detection Ended", "light_red")
            elif frame_queue.empty() and not target_detected and not currently_detected:
                event_queue.put({"type": "done_detecting_queue"})

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)

        frame_queue.task_done()


class YOLOProcessor:
    def __init__(self, event_queue: multiprocessing.Queue, run_nth_frame=1):
        self._frame_count = 0
        self.run_nth_frame = run_nth_frame
        self._frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        model = YOLO("yolo26x.pt")  # Consider changing from pytorch to TensorRT

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
            if self._frame_queue.full():
                print_time(
                    f"Frame added: {self._frame_queue.qsize()}", (109, 0, 128)
                )  # purple
            self._frame_queue.put(frame)

        self._frame_count += 1
        if self._frame_count >= self.run_nth_frame:
            self._frame_count = 0
