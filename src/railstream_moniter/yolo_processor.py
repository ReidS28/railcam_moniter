import threading
import time
import numpy as np
from utils import print_time
from ultralytics import YOLO
import torch
import sys


class YOLOProcessor(threading.Thread):
    def __init__(self, config, shared_data, data_lock, new_frame_event):
        super().__init__()
        self.config = config
        self.shared_data = shared_data
        self.data_lock = data_lock
        self.new_frame_event = new_frame_event
        self.running = True

        self.detection_target = "train"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolo11m.pt")

        self.model.to(self.device)

        print_time("Warming up model...")
        dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
        
        # Run a silent inference
        self.model(dummy_input, verbose=False) 
        print_time("Warmup complete. Model is ready.")

        self.target_class_id = -1
        for cls_id, name in self.model.names.items():
            if name == self.detection_target:
                self.target_class_id = cls_id
                break

        if self.target_class_id == -1:
            print_time(
                f"Warning: Target '{self.detection_target}' not found in model names. Processor will not run detection."
            )
            sys.exit()

    def run(self):

        frame_count = 0
        PROCESS_EVERY_N_FRAMES = 1

        print_time("YOLO Processor thread started")

        while self.running:

            frame_ready = self.new_frame_event.is_set()
            if frame_ready:
                self.new_frame_event.clear()


                frame_count += 1
                if frame_count < PROCESS_EVERY_N_FRAMES:
                    continue
                else:
                    frame_count = 0

                current_frame = None
                with self.data_lock:
                    if self.shared_data["latest_frame"] is not None:
                        current_frame = self.shared_data["latest_frame"].copy()

                if current_frame is not None:
                    confidence, box = self._run_detection(current_frame)

                    with self.data_lock:
                        self.shared_data["detection_result"] = {
                            "confidence": confidence,
                            "box": box,
                            "timestamp": time.time(),
                        }

    def _run_detection(self, frame):

        if self.target_class_id == -1:
            print_time("No class id... Returning")
            return 0.0, None

        results = self.model(frame, classes=[self.target_class_id], verbose=False)

        max_confidence = 0.0
        best_detection_box = None

        for result in results:
            if result.boxes:
                for box_normalized, conf in zip(
                    result.boxes.xywhn.cpu().numpy(), result.boxes.conf.cpu().numpy()
                ):

                    current_conf = float(conf)

                    if current_conf > max_confidence:
                        max_confidence = current_conf
                        best_detection_box = [float(coord) for coord in box_normalized]

        return max_confidence, best_detection_box

    def stop(self):
        self.running = False
        print_time("YOLO Processor stop signal sent.")
