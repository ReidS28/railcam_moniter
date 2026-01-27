import configparser
import threading
import time
import dearpygui.dearpygui as dpg
import numpy as np
from utils import print_time, BoundingBoxColor

from stream_handler import StreamHandler
from yolo_processor import YOLOProcessor
from ui import StreamUI
from detection_handler import DecectionHandler

# --- Global Variables ---
config = None
data_lock = None
shared_data = {}
streamer = None
yolo_processor = None
ui = None
detection_handler = None

target_detected = False
detection_end_countdown = 0
last_render_time = 0.0

last_box = None
current_frame = None
detection_result = None

TARGET_FPS = 30
FRAME_DURATION_SEC = 1 / TARGET_FPS

DETECTON_COUNTDOWN_FRAMES = 30


def main():

    setup()

    while ui.is_running():
        main_loop()
        time.sleep(0.001)

    cleanup()


def setup():
    global config, data_lock, shared_data, streamer, yolo_processor, ui, detection_handler, target_detected, detection_end_countdown, last_render_time

    config = configparser.ConfigParser()
    config.read("config.ini")

    target_detected = False
    detection_end_countdown = 0

    data_lock = threading.Lock()
    shared_data = {
        "latest_frame": None,
        "detection_result": {
            "confidence": 0.0,
            "box": None,
            "color": BoundingBoxColor.RED,
        },
    }

    streamer = StreamHandler(config, shared_data, data_lock)
    streamer.daemon = True
    streamer.start()

    #yolo_processor = YOLOProcessor(config, shared_data, data_lock)
    #yolo_processor.daemon = True
    #yolo_processor.start()

    ui = StreamUI(config)
    ui.create_ui()

    detection_handler = DecectionHandler(config, shared_data, data_lock)

    print_time("UI loop starting. Close the window to exit.")

    last_render_time = time.time()


def main_loop():
    global last_render_time, target_detected, detection_end_countdown, last_box, current_frame, detection_result

    current_time = time.time()

    if (current_time - last_render_time) >= FRAME_DURATION_SEC:

        current_frame = None
        with data_lock:
            if shared_data["latest_frame"] is not None:
                current_frame = shared_data["latest_frame"].copy()
            if shared_data["detection_result"] is not None:
                detection_result = shared_data["detection_result"].copy()

        update_detection()

        if current_frame is not None:
            ui.update_frame(current_frame, detection_result)

        last_render_time = current_time

    ui.render()


def update_detection():
    # TODO Use confidence to improve results
    global target_detected, detection_end_countdown, current_frame, detection_result, last_box, detection_handler, DETECTON_COUNTDOWN_FRAMES

    current_frame_detection = detection_result["box"] != None

    orange = (255, 192, 128)

    if current_frame_detection:
        if not target_detected:
            target_detected = True

            detection_thread = threading.Thread(
                target=detection_handler.new_detection,
                args=(current_frame, detection_result),
            )
            detection_thread.start()

        detection_end_countdown = DETECTON_COUNTDOWN_FRAMES
    else:
        if detection_end_countdown > 0:
            detection_end_countdown -= 1
        else:
            if target_detected:
                target_detected = False

                detection_handler.end_detection()

    if target_detected and current_frame_detection:
        last_box = detection_result["box"]
        detection_result["color"] = BoundingBoxColor.GREEN
    elif target_detected:
        detection_result["confidence"] = -1
        detection_result["box"] = last_box
        detection_result["color"] = BoundingBoxColor.YELLOW
    else:
        detection_result["color"] = None


def cleanup():
    # Access global variables (streamer, yolo_processor, ui)
    print_time("Window closed, stopping stream threads...")

    # Stop and join the threads
    if yolo_processor:
        yolo_processor.stop()
        yolo_processor.join()

    if streamer:
        streamer.stop()
        streamer.join()

    # Clean up UI resources
    if ui:
        ui.cleanup()

    print_time("Program exited gracefully.")


if __name__ == "__main__":
    main()
