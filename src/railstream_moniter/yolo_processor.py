from ultralytics import YOLO
from config import MODEL_PATH, DETECTION_SETTINGS, MAX_QUEUE_SIZE
from stream_handler import raw_frame_queue
import queue

processed_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def yolo_processor():
    """Runs YOLO detection on frames and puts results into a new queue."""
    print(f"YOLO Processor started. Loading model: {MODEL_PATH}")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model. {e}")
        return

    while True:
        frame = raw_frame_queue.get()
        if frame is None:
            break

        conf_threshold = DETECTION_SETTINGS.get("start_detection_confidence_threshold", 0.4)
        results = model.track(frame, persist=True, conf=conf_threshold)
        annotated_frame = results[0].plot()
        detection_count = len(results[0].boxes)

        if not processed_frame_queue.full():
            processed_frame_queue.put((annotated_frame, detection_count))
        
        raw_frame_queue.task_done()
