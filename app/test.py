from collections import deque

import cv2
import numpy as np

VIDEO_PATH = "detections/detection.mp4"

X1, Y1, X2, Y2 = 0, 260, 1280, 420

# Downscale factor applied to the ROI before correlating (1.0 = none).
# Try 0.5 or 0.25 if dx is erratic on fast/repetitive sections.
SCALE = 0.5

# Ignore measurements with a weaker correlation peak than this (tune on your footage)
MIN_RESPONSE = 0.1

# Number of recent measurements used for the median smoothing
SMOOTH_N = 5


def preprocess(gray):
    """Crop to the ROI, optionally downscale, convert to float32."""
    roi = gray[Y1:Y2, X1:X2]
    if SCALE != 1.0:
        roi = cv2.resize(roi, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return roi.astype(np.float32)


cap = cv2.VideoCapture(VIDEO_PATH)

ret, old_frame = cap.read()
if not ret:
    raise RuntimeError(f"Could not read from {VIDEO_PATH}")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_roi = preprocess(old_gray)

# Hanning window reduces edge effects; must match the ROI size (width, height)
window = cv2.createHanningWindow((old_roi.shape[1], old_roi.shape[0]), cv2.CV_32F)

recent_dx = deque(maxlen=SMOOTH_N)
total_dx = 0.0  # accumulated displacement in full-resolution pixels
speed_px = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = preprocess(frame_gray)

    # Shift of the current ROI relative to the previous one
    (dx, dy), response = cv2.phaseCorrelate(old_roi, roi, window)

    # Convert back to full-resolution pixels
    dx /= SCALE
    dy /= SCALE

    if response > MIN_RESPONSE:
        recent_dx.append(dx)
        speed_px = float(np.median(recent_dx))
        total_dx += dx
        status = "OK"
    else:
        status = "LOW CONFIDENCE (skipped)"

    direction = "right" if speed_px > 0 else "left" if speed_px < 0 else "-"

    # ---- Drawing ----
    cv2.rectangle(frame, (X1, Y1), (X2 - 1, Y2 - 1), (0, 255, 0), 2)

    # Arrow from the ROI centre, length proportional to the smoothed speed
    cx, cy = (X1 + X2) // 2, (Y1 + Y2) // 2
    cv2.arrowedLine(
        frame,
        (cx, cy),
        (int(cx + speed_px * 5), cy),
        (0, 0, 255),
        4,
        tipLength=0.3,
    )

    lines = [
        f"dx: {dx:7.2f} px/frame   dy: {dy:6.2f}",
        f"smoothed: {speed_px:7.2f} px/frame ({direction})",
        f"total: {total_dx:9.1f} px",
        f"response: {response:.3f}  {status}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("frame", frame)

    k = cv2.waitKey(25)
    if k == 27:  # Esc
        break

    # Update previous ROI
    old_roi = roi

cv2.destroyAllWindows()
cap.release()
