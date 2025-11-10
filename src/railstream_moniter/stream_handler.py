import cv2
import queue
import time
from config import STREAM_URL, MAX_QUEUE_SIZE

raw_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def stream_reader():
    """Reads frames from the m3u8 stream and puts them into a queue."""
    print(f"Stream Reader started. Connecting to: {STREAM_URL}")
    
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("ERROR: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Stream ended or error reading frame. Restarting stream...")
            time.sleep(5)
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue
        
        # Drop oldest frame if queue is full to prevent lag
        if raw_frame_queue.full():
            try:
                raw_frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        raw_frame_queue.put(frame)
