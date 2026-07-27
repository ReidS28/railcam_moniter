from datetime import datetime

from termcolor import colored


def print_time(
    message="",
    color: str | tuple[int, int, int] = "white",
    highlight_color=None,
    attrs=[],
):
    """
    Print a message with timestamp in color
    """
    message_str = str(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    strings = message_str.split("\n")
    for string in strings:
        print(
            colored(f"[{timestamp}]", "blue")
            + colored(f" {string}", color, highlight_color, attrs)
        )
