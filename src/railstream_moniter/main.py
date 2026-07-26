import time

from cv2.typing import MatLike
from movement_detection import MovementDetection
from web_player import webPlayer


def main():
    movement_detector = MovementDetection(run_nth_frame=2)

    frame_count = 0

    def color_to_hex(color) -> str:
        b, g, r = color
        return f"{r:02x}{g:02x}{b:02x}"

    def on_frame(frame: MatLike):
        # print_time(f"{frame.mean():.2} : #{color_to_hex(frame[0][0])}")
        movement_detector.add_frame(frame)

    player = webPlayer(
        "https://railstream.net/live-cameras/item/fostoria-oh-east-guest",
        frame_callback=on_frame,
    )
    # player = webPlayer(
    #     frame_callback=on_frame,
    # )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        player.end()


if __name__ == "__main__":
    main()
