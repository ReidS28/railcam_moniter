import multiprocessing
import queue
import threading
from collections import deque
from collections.abc import Callable
from queue import Queue
from uuid import UUID

import constants
import cv2
from ultralytics import YOLO
from utils import IndexedFrame, get_frame_from_id, print_time

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
    update_bounding_boxes_callback: Callable,
):

    target_detected = False
    detection_frames = 0

    target_ids = [
        i
        for i, name in model.names.items()
        if name.upper() == constants.DETECTION_TARGET
    ]

    while True:
        try:
            target_frame_id = id_queue.get(timeout=0.03)
            indexed_frame: IndexedFrame | None = get_frame_from_id(
                rolling_storage, target_frame_id
            )
            if indexed_frame is None:
                continue

        except queue.Empty:
            cv2.waitKey(1)
            continue

        if (
            indexed_frame is None
        ):  # Quit if None frame is given TODO: Make this id based
            id_queue.task_done()
            break

        result = model.track(
            indexed_frame.frame,
            verbose=False,
            conf=constants.YOLO_DETECTION_CONFIDENCE_THRESHOLD,
            imgsz=640,  # this number seems to work best (It's what it was trained at)
            device=0,
            # classes=target_ids,
            persist=True,
            tracker="botsort.yaml",  # tracker="bytetrack.yaml",
        )[0]

        # indexed_frame.detection_result = result

        annotated_frame = result.plot()
        cv2.imshow("YOLO26 Tracking", annotated_frame)
        cv2.waitKey(1)

        boxes = result.boxes

        currently_detected = False

        payload = []
        if boxes is not None:
            for box in boxes:

                def extract_val(val):
                    if val is None:
                        return None
                    if hasattr(val, "cpu"):
                        val = val.cpu()
                    if hasattr(val, "tolist"):
                        arr = val.tolist()
                    else:
                        try:
                            arr = list(val)
                        except Exception:  # noqa BLE001
                            arr = [val]
                    if isinstance(arr, list) and len(arr) > 0:
                        return arr[0]
                    return arr

                detected_item = model.names[int(box.cls[0])].upper()

                payload.append(
                    {
                        "conf": extract_val(box.conf),
                        "class_name": detected_item,
                        "color": None,
                        "id": extract_val(box.id),
                        # "xywh": extract_val(box.xywh),
                        "xywhn": extract_val(box.xywhn),
                        # "xyxy": extract_val(box.xyxy),
                        # "xyxyn": extract_val(box.xyxyn),
                    }
                )

        update_bounding_boxes_callback(payload)

        if boxes is not None:  # At least one detection
            currently_detected = True

        if currently_detected:
            detection_frames += 1
        elif target_detected:
            detection_frames -= 1
        else:
            detection_frames = 0

        detection_frames = max(
            -constants.YOLO_END_DETECTION_FRAMES,
            min(constants.YOLO_START_DETECTION_FRAMES, detection_frames),
        )

        if detection_frames >= constants.YOLO_START_DETECTION_FRAMES:
            if not target_detected:
                target_detected = True
                event_queue.put({"type": "target_detected_started"})

        elif detection_frames <= -constants.YOLO_END_DETECTION_FRAMES:
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
        update_bounding_box_callback: Callable = (lambda *_: None),
    ):
        self.event_queue = event_queue
        self.rolling_storage = rolling_storage
        self.run_nth_frame = run_nth_frame
        self.update_bounding_box_callback = update_bounding_box_callback
        # self.edit_bounding_box_callback = edit_bounding_box_callback
        # self.reset_bounding_box_callback = reset_bounding_box_callback

        self._frame_count = 0
        self._id_queue: Queue[UUID] = queue.Queue(maxsize=constants.YOLO_MAX_QUEUE_SIZE)
        self._model = YOLO("yolo26x.pt")  # Consider changing from pytorch to TensorRT

        thread = threading.Thread(
            target=_worker,
            args=(
                self.rolling_storage,
                self._id_queue,
                self._model,
                self.event_queue,
                self.update_bounding_box_callback,
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
