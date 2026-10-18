import constants
import cv2
import numpy as np
from config import config
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

    def get_movement(self, item: IndexedFrame):
        frame = cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape

        blank = np.zeros((h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        if (
            item.detection_results is not None
            and item.detection_results.boxes is not None
        ):
            for box in item.detection_results.boxes:
                coords = (
                    box.xyxy[0].cpu().numpy()
                    if hasattr(box.xyxy[0], "cpu")
                    else box.xyxy[0]
                )
                x1, y1, x2, y2 = map(int, coords[:4])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        corners = cv2.goodFeaturesToTrack(
            frame,
            maxCorners=constants.DIRECTION_TRACKING_POINTS,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask,
        )

        # print_time(f"Corners: {corners}")

        if corners is not None and config["dev"].getboolean("plot_corner_points"):
            corners = corners.astype(int)
            for i in corners:
                x, y = i.ravel()
                cv2.circle(item.frame, (x, y), 3, (0, 255, 0), -1)
                cv2.circle(blank, (x, y), 3, 255, -1)

        cv2.imshow("frame", item.frame)
        # cv2.waitKey(1)

        return 0
