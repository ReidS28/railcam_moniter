import configparser
import threading
import time
import dearpygui.dearpygui as dpg
import numpy as np
from utils import print_time, BoundingBoxColor

from stream_handler import StreamHandler
from yolo_processor import YOLOProcessor
from ui import StreamUI

# --- Global Variables ---
config = None
data_lock = None
shared_data = {}
streamer = None
yolo_processor = None
ui = None

target_detected = False
detection_end_countdown = 0
last_render_time = 0.0

last_box = None

TARGET_FPS = 30
FRAME_DURATION_SEC = 1 / TARGET_FPS 

DETECTON_COUNTDOWN_FRAMES = 30

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def main():
    
    setup()

    while ui.is_running():
        main_loop()
        time.sleep(0.001)
    
    cleanup()

def setup():
    global config, data_lock, shared_data, streamer, yolo_processor, ui, target_detected, detection_end_countdown, last_render_time

    config = load_config()
    
    target_detected = False
    detection_end_countdown = 0

    data_lock = threading.Lock()
    shared_data = {
        'latest_frame': None,
        'detection_result': {'confidence': 0.0, 'box': None, 'color': BoundingBoxColor.RED} 
    }

    streamer = StreamHandler(config, shared_data, data_lock)
    streamer.daemon = True 
    streamer.start()

    yolo_processor = YOLOProcessor(config, shared_data, data_lock)
    yolo_processor.daemon = True
    yolo_processor.start()

    ui = StreamUI(config)
    ui.create_ui()

    print_time("UI loop starting. Close the window to exit.")

    last_render_time = time.time() 

def main_loop():
    global last_render_time, target_detected, detection_end_countdown, last_box
    
    current_time = time.time()

    if (current_time - last_render_time) >= FRAME_DURATION_SEC:
        
        current_frame = None
        detection_result = None
        with data_lock:
            if shared_data['latest_frame'] is not None:
                current_frame = shared_data['latest_frame'].copy()
            if shared_data['detection_result'] is not None:
                detection_result = shared_data['detection_result'].copy()

        detection_result['color'] = update_detection(detection_result['confidence'], detection_result['box'])

        if detection_result['box']:
            last_box = detection_result['box']

        elif detection_result['color'] == BoundingBoxColor.YELLOW:
            print_time(last_box, "red")
            detection_result['box'] = last_box
        #else:
        #    detection_result['box'] = None

        if current_frame is not None:
            ui.update_frame(current_frame, detection_result)

        last_render_time = current_time

    ui.render() 

def update_detection(confidence, box):
    #TODO Use confidence to improve results
    global target_detected, detection_end_countdown, DETECTON_COUNTDOWN_FRAMES

    current_frame_detection = box != None

    orange = (255, 192, 128)

    if current_frame_detection:
        if not target_detected:
            target_detected = True

            print_time()
            print_time("-----------------------", 'green')
            print_time("-- Detection Started --", 'green')
            print_time("-----------------------", 'green')
        
        detection_end_countdown = DETECTON_COUNTDOWN_FRAMES
    else:
        if detection_end_countdown > 0:
            detection_end_countdown -= 1
        else:
            if target_detected:
                target_detected = False

                print_time("--- Detection Ended ---", orange)
                print_time("-----------------------", orange)

    if target_detected and current_frame_detection:
        print_time("GREEN", 'green')
        return BoundingBoxColor.GREEN
    elif target_detected:
        print_time("YELLOW", 'yellow')
        return BoundingBoxColor.YELLOW
    else:
        return None

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