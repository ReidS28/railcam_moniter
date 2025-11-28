import threading
import time
import numpy as np
from utils import print_time
from ultralytics import YOLO
import torch

class YOLOProcessor(threading.Thread):
    def __init__(self, config, shared_data, data_lock):
        super().__init__()
        self.config = config
        self.shared_data = shared_data
        self.data_lock = data_lock
        self._stop_event = threading.Event()

        self.detection_target = "train"

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print_time(f"YOLO detection running on device: {self.device}")

        self.model = YOLO("yolo11n.pt")

        self.target_class_id = -1
        
        for cls_id, name in self.model.names.items():
            if name == self.detection_target:
                self.target_class_id = cls_id
                break
        
        if self.target_class_id == -1:
            print_time(f"Warning: Target '{self.detection_target}' not found in model names. Processor will not detect this target.")
        else:
            print_time(f"YOLO Processor configured to detect: '{self.detection_target}' (Class ID: {self.target_class_id})")

        print_time("YOLO Processor initialized (placeholder model loaded).")

    def run(self):
        TARGET_FPS = 5
        FRAME_DURATION_SEC = 1 / TARGET_FPS
        
        last_process_time = time.time()
        
        print_time("YOLO Processor thread started.")
        
        while not self._stop_event.is_set():
            current_time = time.time()
            time_to_wait = FRAME_DURATION_SEC - (current_time - last_process_time)
            
            if time_to_wait > 0:
                self._stop_event.wait(time_to_wait)
                if self._stop_event.is_set():
                    break
            
            last_process_time = time.time()

            current_frame = None

            with self.data_lock:
                if self.shared_data['latest_frame'] is not None:
                    current_frame = self.shared_data['latest_frame'].copy() 

            if current_frame is not None:

                confidence, box = self._run_detection(current_frame)

                if box:
                    print_time(f"Detected a {self.detection_target} with confidence: {confidence:.2f} at box: {box}")

    def _run_detection(self, frame):

        if self.target_class_id == -1:
            print_time("No class id... Returning")
            return 0.0, None
        
        results = self.model(frame, classes=[self.target_class_id], verbose=False)

        max_confidence = 0.0
        best_detection_box = None

        for result in results:
            if result.boxes: 
                for box_normalized, conf in zip(result.boxes.xywhn.cpu().numpy(), 
                                                result.boxes.conf.cpu().numpy()):
                    
                    current_conf = float(conf)

                    if current_conf > max_confidence:
                        max_confidence = current_conf
                        best_detection_box = [float(coord) for coord in box_normalized]

        return max_confidence, best_detection_box

    def stop(self):
        self._stop_event.set()
        print_time("YOLO Processor stop signal sent.")