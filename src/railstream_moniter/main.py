import time

import multiprocessing
from collections import deque

from cv2.typing import MatLike
from yolo_processer import YOLOProcessor
from movement_detection import MovementDetection
from web_player import webPlayer

from utils import print_time

PRE_ROLL_FRAMES = 30

def main():

    event_queue = multiprocessing.Queue(maxsize=10)

    rolling_storage: deque[MatLike] = deque(maxlen=PRE_ROLL_FRAMES)

    yolo_active = False

    def on_frame(frame: MatLike):
        rolling_storage.append(frame)
        movement_detector.add_frame(frame)
    
    movement_detector = MovementDetection(event_queue, run_nth_frame=2)
    yolo_processer = YOLOProcessor(event_queue, run_nth_frame=2)

    # url = "https://railstream.net/live-cameras/item/cptweed-guest"
    url = "https://railstream.net/live-cameras/item/fostoria-oh-east-guest"
    # url = "https://railstream.net/live-cameras/item/fullerton-guest"
    # url = "https://railstream.net/live-cameras/item/oregon-guest"
    # url = "https://railstream.net/live-cameras/item/selma-guest"
    player = webPlayer(
        url,
        frame_callback=on_frame,
    )

    def handle_event(event: dict) -> None:
        nonlocal yolo_active
        event_type = event.get("type")

        match event_type:
            case "movement":
                yolo_active = True
            case _:
                print_time(f"Unknown event type: {event_type!r}", 'yellow')

    def drain_event_queue(event_queue: multiprocessing.Queue):
        while not event_queue.empty():
            event = event_queue.get()
            handle_event(event)

    try:
        while True:

            drain_event_queue(event_queue)

            if yolo_active:
                if rolling_storage:
                    yolo_processer.add_frame(rolling_storage.pop())
                else:
                    yolo_active = False

            time.sleep(0.001)

    except KeyboardInterrupt:
        player.end()


if __name__ == "__main__":
    main()
