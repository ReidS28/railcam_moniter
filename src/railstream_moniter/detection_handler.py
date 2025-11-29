import cv2
import numpy as np
from PIL import Image
import datetime

from utils import print_time

class DecectionHandler():

    latest_start_frame = None
    latest_start_frame_detection_result = None

    def new_detection(self, current_frame, detection_result):
        global latest_start_frame, latest_start_detection_result

        print_time()
        print_time("-----------------------", 'green')
        print_time("-- Detection Started --", 'green')
        print_time("-----------------------", 'green')

        latest_start_frame = current_frame
        latest_start_detection_result = detection_result

        self.save_frame(current_frame)

    def end_detection(self):
        orange = (255, 192, 128)
        print_time("-----------------------", orange)
        print_time("--- Detection Ended ---", orange)
        print_time("-----------------------", orange)

    def save_frame(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        now = datetime.datetime.now()

        file_path = "detections/"
        file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"

        pil_image.save(file_path + file_name)
        print_time(f"Frame saved as {file_path + file_name}")

        return file_name
