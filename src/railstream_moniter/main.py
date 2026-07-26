import time

from cv2.typing import MatLike
from movement_detection import MovementDetection
from web_player import webPlayer


def main():
    movement_detector = MovementDetection(run_nth_frame=2)

    def on_frame(frame: MatLike):
        movement_detector.add_frame(frame)

    # url = "https://railstream.net/live-cameras/item/cptweed-guest"
    url = "https://railstream.net/live-cameras/item/fostoria-oh-east-guest"
    # url = "https://railstream.net/live-cameras/item/fullerton-guest"
    # url = "https://railstream.net/live-cameras/item/oregon-guest"
    # url = "https://railstream.net/live-cameras/item/selma-guest"
    player = webPlayer(
        url,
        frame_callback=on_frame,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        player.end()


if __name__ == "__main__":
    main()
