import cv2
import numpy as np
from PIL import Image
import datetime
import time

from utils import print_time
from slack import Slack

class DecectionHandler():

    def __init__(self, config, shared_data, data_lock):
        self.slack = Slack()

        self.shared_data = shared_data
        self.data_lock = data_lock

        self.frame_size = None

    def new_detection(self, current_frame, detection_result):
        global latest_start_frame, latest_start_detection_result

        h, w, _ = current_frame.shape
        self.frame_size = (w, h)

        print_time()
        print_time("-----------------------", 'green')
        print_time("-- Detection Started --", 'green')
        print_time("-----------------------", 'green')

        now = datetime.datetime.now()
        file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        file_path = f"detections/{file_name}"

        self.save_video(30, 30 * 3, file_path)

        message = f"Train Detected @ {now.strftime('%Y/%m/%d %H:%M:%S')}"

        self.slack.upload_file(file_path, message)

    def end_detection(self):
        orange = (255, 192, 128)
        print_time("-----------------------", orange)
        print_time("--- Detection Ended ---", orange)
        print_time("-----------------------", orange)

    def save_frame(self, frame, file_path):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        pil_image.save(file_path)
        print_time(f"Frame saved as {file_path}")

        return file_path

    def save_video(self, target_fps, total_frames, file_path):

        if self.frame_size == None:
            print_time("Error no frame size", "red")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        
        FRAME_DURATION_SEC = 1.0 / target_fps
        
        try:
            video_writer = cv2.VideoWriter(
                file_path,
                fourcc,
                target_fps,
                self.frame_size
            )
            if not video_writer.isOpened():
                raise IOError("VideoWriter failed to open. Check codec support or file path permissions.")
                
        except Exception as e:
            print_time(f"FATAL VIDEO WRITER SETUP ERROR: {e}", "red")
            return

        current_frame = 0
        last_run_time = time.monotonic() 
        
        while current_frame < total_frames:
            
            current_time = time.monotonic()
            
            while (current_time - last_run_time) >= FRAME_DURATION_SEC:
                
                last_run_time += FRAME_DURATION_SEC
                
                frame = None
                with self.data_lock:
                    if self.shared_data.get('latest_frame') is not None:
                        frame = self.shared_data['latest_frame'].copy()
                
                if frame is not None:
                    video_writer.write(frame)
                    current_frame += 1
                else:
                    print_time("Warning: Shared frame was None, skipped write for this step.", "yellow")

                if current_frame >= total_frames:
                    break
                
                current_time = time.monotonic()

            time.sleep(0.001)

        if video_writer is not None:
            video_writer.release()
            print_time(f"Successfully saved {current_frame} frames to {file_path}.")
