import cv2
import numpy as np
from cv2.typing import MatLike
from utils import IndexedFrame

SCALE = 0.5
MIN_RESPONSE = 0.1
MIN_MOVEMENT = 1


class MovementDetection:
    def __init__(
        self,
    ) -> None:
        self._latest_box = None
        self.old_frame: MatLike | None = None

    def _preprocess(self, frame, box):
        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        roi = frame[y1:y2, x1:x2]

        if SCALE != 1.0:
            roi = cv2.resize(
                roi, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA
            )

        return roi.astype(np.float32)

    def get_movement(self, item: IndexedFrame):
        frame = cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY)

        if (
            item.detection_results is not None
            and item.detection_results.boxes is not None
            and len(item.detection_results.boxes) > 0
        ):
            box = item.detection_results.boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            self._latest_box = (int(x1), int(y1), int(x2), int(y2))

        if self.old_frame is None:
            self.old_frame = frame
            return None

        if self._latest_box is None:
            self.old_frame = frame
            return None

        box = self._latest_box
        roi = self._preprocess(frame, box)
        old_roi = self._preprocess(self.old_frame, box)
        self.old_frame = frame

        if old_roi.shape != roi.shape:
            return None

        window = cv2.createHanningWindow((roi.shape[1], roi.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(old_roi, roi, window)

        dx /= SCALE
        dy /= SCALE

        if response < MIN_RESPONSE or abs(dx) < MIN_MOVEMENT:
            dx = 0
            dy = 0

        # print_time(f"roi.shape: {roi.shape}, old_roi.shape: {old_roi.shape}")
        cv2.imshow("roi", roi.astype(np.uint8))
        cv2.imshow("old_roi", old_roi.astype(np.uint8))
        cv2.waitKey(1)

        return dx
