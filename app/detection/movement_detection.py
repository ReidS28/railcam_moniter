import cv2
import numpy as np
from config import config
from cv2.typing import MatLike
from utils import IndexedFrame, print_time

SCALE = 0.5
MIN_MOVEMENT = 1  # px/frame 
STATIC_THRESH = 0.3 # px/frame 
MIN_MOVING_PX = 200 # px
MIN_GRADIENT = 20 

FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


class MovementDetection:
    def __init__(self) -> None:
        self._latest_box = None
        self.old_frame: MatLike | None = None
        self._debug_mask: np.ndarray | None = None

    def get_movement(self, item: IndexedFrame) -> float:
        frame = cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

        box_mask = self._build_mask(item, frame.shape)
        movement = self._track(frame, box_mask)

        self.old_frame = frame

        if config["dev"].getboolean("show_movement_details"):
            self._draw_debug(item.frame)
        cv2.imshow("frame", item.frame)

        if movement is None:
            return 0

        dx, _dy = movement
        if abs(dx) < MIN_MOVEMENT:
            return 0
        return dx

    def _build_mask(self, item: IndexedFrame, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        results = item.detection_results
        if results is None or results.boxes is None:
            return mask

        for box in results.boxes:
            coords = (
                box.xyxy[0].cpu().numpy()
                if hasattr(box.xyxy[0], "cpu")
                else box.xyxy[0]
            )
            x1, y1, x2, y2 = (int(v * SCALE) for v in coords[:4])
            cv2.rectangle(
                mask, (x1, y1), (x2, y2), 255, -1
            )
        return mask

    def _track(
        self, frame: np.ndarray, box_mask: np.ndarray
    ) -> tuple[float, float] | None:
        self._debug_mask = None
        if self.old_frame is None:
            return None

        if self.old_frame.shape != frame.shape:
            print_time(f"self.old_frame.shape: {self.old_frame.shape}, frame.shape: {frame.shape}")
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.old_frame,
            frame,
            None,  # type: ignore
            **FARNEBACK,
        )

        # Only trust textured pixels inside the detection boxes
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        textured = np.hypot(gx, gy) > MIN_GRADIENT
        valid = (box_mask > 0) & textured

        if np.count_nonzero(valid) < MIN_MOVING_PX:
            return None

        # Drop static background (empty centerbeams, scenery behind the car)
        mag = np.linalg.norm(flow, axis=2)
        moving = valid & (mag > STATIC_THRESH)

        if np.count_nonzero(moving) < MIN_MOVING_PX:
            return 0.0, 0.0  # stopped, or only background visible

        # Keep only the dominant horizontal direction
        pos = moving & (flow[..., 0] > 0)
        neg = moving & (flow[..., 0] < 0)
        moving = pos if np.count_nonzero(pos) >= np.count_nonzero(neg) else neg

        if np.count_nonzero(moving) < MIN_MOVING_PX:
            return 0.0, 0.0

        self._debug_mask = moving

        vecs = flow[moving]  # (N, 2)
        dx, dy = np.median(vecs, axis=0)
        return float(dx / SCALE), float(dy / SCALE)

    def _draw_debug(self, frame: np.ndarray) -> None:
        """Tint the pixels that contributed to the movement estimate."""
        if self._debug_mask is None:
            return
        full = cv2.resize(
            self._debug_mask.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        frame[full > 0] = (0.6 * frame[full > 0] + 0.4 * np.array([0, 255, 0])).astype(
            np.uint8
        )
