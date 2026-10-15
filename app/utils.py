from collections import deque
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import numpy as np
import torch
from cv2.typing import MatLike
from termcolor import colored
from ultralytics.engine.results import Boxes


@dataclass
class IndexedFrame:
    frame: MatLike
    id: UUID
    detection_result: Boxes | None


def print_time(
    message="",
    color: str | tuple[int, int, int] = "white",
    highlight_color=None,
    attrs=None,
):
    """
    Print a message with timestamp in color
    """
    message_str = str(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # noqa
    strings = message_str.split("\n")
    for string in strings:
        print(
            colored(f"[{timestamp}]", "blue")
            + colored(f" {string}", color, highlight_color, attrs)
        )


def get_frame_index_from_id(
    rolling_storage: deque[IndexedFrame], target_id: UUID
) -> int | None:
    for i, frame in enumerate(list(rolling_storage)):
        if frame.id == target_id:
            return i
    return None


def get_frame_from_id(
    rolling_storage: deque[IndexedFrame], target_id: UUID
) -> IndexedFrame | None:
    for frame in rolling_storage:
        if frame.id == target_id:
            return frame
    return None


def safe_convert(val):
    if val is None:
        return []
    if isinstance(val, torch.Tensor):
        return val.cpu().tolist()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    # Fallback if it has a tolist method
    if hasattr(val, "tolist"):
        return val.tolist()
    return list(val)
