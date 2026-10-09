import multiprocessing
import time
import uuid
from collections import deque

import constants
from config import config
from cv2.typing import MatLike
from detection.detection_handler import DetectionHandler
from detection.yolo_detection import YOLODectection
from slack_bot import SlackBot
from utils import IndexedFrame, print_time
from web_player import webPlayer


def main():

    event_queue = multiprocessing.Queue(maxsize=10)
    rolling_storage: deque[IndexedFrame] = deque(maxlen=constants.PRE_ROLL_FRAMES)

    slack_bot = SlackBot("#railstream-alerts")
    detection_handler = DetectionHandler(rolling_storage, slack_bot)

    yolo_active = False

    def on_frame(frame: MatLike):

        id = uuid.uuid4()
        indexed_frame = IndexedFrame(frame, id, None)

        rolling_storage.append(indexed_frame)
        yolo_detection.add_frame_id(id)
        # detection_handler.add_id(id)

    player = webPlayer(
        config["general"]["url"], frame_callback=on_frame, print_console_log=False
    )

    yolo_detection = YOLODectection(
        event_queue,
        rolling_storage,
        run_nth_frame=4,
        bounding_box_callback=lambda *args, **kwargs: player.edit_bounding_box(
            *args, **kwargs
        ),
    )

    def handle_event(event: dict) -> None:
        nonlocal yolo_active
        event_type = event.get("type")

        match event_type:
            case "movement":
                yolo_active = True

            case "target_detected_started":
                print_time(
                    "",
                )
                print_time(
                    "",
                )
                print_time(
                    "------------------------------------------------------", "blue"
                )
                print_time("Detection Started", "light_green")
                detection_handler.new_detection()

            case "target_detected_ended":
                print_time("Detection Ended", "light_red")
                print_time(
                    "------------------------------------------------------", "blue"
                )

            case "done_detecting_queue":
                yolo_active = False

            case _:
                print_time(f"Unknown event type: {event_type!r}", "yellow")

    def drain_event_queue(event_queue: multiprocessing.Queue):
        while not event_queue.empty():
            event = event_queue.get()
            handle_event(event)

    try:
        while True:
            drain_event_queue(event_queue)

            time.sleep(0.01)

    except KeyboardInterrupt:
        player.end()


if __name__ == "__main__":
    main()
