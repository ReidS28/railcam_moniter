import configparser
import threading
import time
import dearpygui.dearpygui as dpg
import numpy as np
from utils import print_time, BoundingBoxColor

from stream_handler import StreamHandler
from yolo_processor import YOLOProcessor
from ui import StreamUI

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def main():
    config = load_config()
    
    target_detected = False
    detection_end_countdown = 0

    data_lock = threading.Lock()
    shared_data = {
        'latest_frame': None,
        'detection_result': {'confidence': 0.0, 'box': None, 'color': BoundingBoxColor.RED} # Box: [cx, cy, w, h] list 
    }

    streamer = StreamHandler(config, shared_data, data_lock)
    streamer.daemon = True 
    streamer.start()

    yolo_processor = YOLOProcessor(config, shared_data, data_lock)
    yolo_processor.daemon = True
    yolo_processor.start()

    # UI Setup
    ui = StreamUI(config)
    ui.create_ui()

    print_time("UI loop starting. Close the window to exit.")

    # Playback Setup
    TARGET_FPS = 30
    FRAME_DURATION_SEC = 1 / TARGET_FPS 
    
    last_render_time = time.time()

    # Main Loop
    while ui.is_running():
        
        current_time = time.time()

        if (current_time - last_render_time) >= FRAME_DURATION_SEC:
            
            current_frame = None
            detection_result = None
            with data_lock:
                if shared_data['latest_frame'] is not None:
                    current_frame = shared_data['latest_frame'].copy()
                if shared_data['detection_result'] is not None:
                    detection_result = shared_data['detection_result'].copy()

            confidence = detection_result['confidence']
            box = detection_result['box']

            if box:
                if (not target_detected):
                    print_time("-------------------------", 'blue')
                    print_time("-- Train Detected -------", 'blue')
                    print_time("-------------------------", 'blue')
                    target_detected = True
                detection_end_countdown = 60 # TODO Make real number
            else:
                if detection_end_countdown >= 0:
                    detection_end_countdown -= 1
                else:
                    target_detected = False

            detection_result['color'] = BoundingBoxColor.GREEN

            if current_frame is not None:
                ui.update_frame(current_frame, detection_result)

            last_render_time = current_time

        ui.render()

        time.sleep(0.001) 

    # Cleanup
    print_time("Window closed, stopping stream threads...")

    yolo_processor.stop() 
    yolo_processor.join()

    streamer.stop()
    streamer.join()

    ui.cleanup()
    
    print_time("Program exited gracefully.")


if __name__ == "__main__":
    main()