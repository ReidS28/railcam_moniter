import os
import cv2
import random
import numpy as np
from utils import print_time
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
from slack_message import send_slack_message, send_slack_photo, send_slack_video

def get_bounding_boxes(cap, model, start_frame, end_frame, detection_class_id=6):

    detections_by_frame = []
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_index = start_frame

        while current_frame_index <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_detections = {
                'frame_number': current_frame_index,
                'bounding_boxes': [],
                'confidences': []
            }
            
            results = model.predict(frame, verbose=False)
            for r in results:
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        if int(box.cls[0]) == detection_class_id:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            frame_detections['bounding_boxes'].append([x1, y1, x2, y2])
                            frame_detections['confidences'].append(confidence)
            
            detections_by_frame.append(frame_detections)
            current_frame_index += 1
            
    finally:
        cap.release()
        
    return detections_by_frame

def get_best_frame_index(detection_data):
    """
    Analyzes detection data and returns the index of the frame with the
    highest confidence score for a single detection.

    Args:
        detection_data (list): A list of dictionaries containing detection info.

    Returns:
        int: The index of the best frame, or -1 if no detections are found.
    """
    best_frame_index = -1
    max_confidence = 0
    
    for frame_data in detection_data:
        if frame_data['confidences']:
            # Find the highest confidence in the current frame
            current_max_confidence = max(frame_data['confidences'])
            
            if current_max_confidence > max_confidence:
                max_confidence = current_max_confidence
                best_frame_index = frame_data['frame_number']
                
    return best_frame_index

def get_frame(video_path, frame_index):
    """
    Retrieves a single, specific frame from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_index (int): The index of the frame to retrieve.

    Returns:
        numpy.ndarray: The frame as a NumPy array, or None if retrieval fails.
    """
    if not os.path.exists(video_path):
        print_time(f"Video file does not exist: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_time(f"Failed to open video file: {video_path}")
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            return frame
        else:
            print_time(f"Failed to read frame {frame_index}")
            return None
    finally:
        cap.release()

async def handle_train_detected(video_path):

    model = YOLO('yolov8m.pt')
    
    start_frame = 0
    end_frame = 150

    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    message = f"🚂 Train Detected @ {timestamp}"

    if not os.path.exists(video_path):
        print_time(f"Video file does not exist: {video_path}", 'red')
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print_time(f"Failed to open video file: {video_path}", 'red')
        return []

    detections_data = get_bounding_boxes(cap, model, start_frame, end_frame)

    best_frame_index = get_best_frame_index(detections_data)

    if best_frame_index == -1:
        print_time(f"No train detected in video: {video_path}", 'red')
        fallback_message = f"{message}\nFailed to find best frame."
        await send_slack_message(fallback_message)
        return

    frame = get_frame(video_path, best_frame_index)

    if frame is None:
        print_time(f"Failed to get frame {best_frame_index} from video: {video_path}", 'red')
        fallback_message = f"{message}\nFailed to get frame {best_frame_index}."
        await send_slack_message(fallback_message)
        return
    
    # Resize the frame if it's too large
    height, width = frame.shape[:2]
    if width > 1280:
        new_width = 1280
        new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))
    
    # Send the extracted frame
    print_time("Sending Slack Message", 'yellow')
    success = await send_slack_photo(frame, message)
    #success = await send_slack_video(video_path, message)
    color = (255, 192, 128) # Orange
    print_time("-----------------------------------", color)
    print_time(f"Message: {message}", color)
    print_time(f"Media Success: {success}", color)
    print_time("-----------------------------------", color)
    
    if not success:
        fallback_message = f"{message}\n Failed to send image."
        await send_slack_message(fallback_message)

if __name__ == "__main__":
    import asyncio
    asyncio.run(handle_train_detected("detections/train_detection_20250808_153732.mp4"))