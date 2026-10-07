from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from cv2.typing import MatLike
from termcolor import colored


@dataclass
class IndexedFrame:
    frame: MatLike
    id: UUID
    detection_results: DetectionResults | None


@dataclass
class DetectionResults:
    pass


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
