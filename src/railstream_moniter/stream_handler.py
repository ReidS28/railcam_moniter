import cv2
import threading
import time
import collections

class StreamHandler(threading.Thread):

    INIT_TARGET_FPS = 30.0
    RECONNECT_DELAY = 5.0
    MAX_CONSECUTIVE_FAILURES = 5
    
    ADJUSTMENT_GAIN = 0.5 

    def __init__(self, config, shared_data, data_lock):
        super().__init__()
        self.stream_url = config.get('Stream', 'url')
        self.shared_data = shared_data
        self.data_lock = data_lock

        self.running = True
        
        self.target_fps = self.INIT_TARGET_FPS
        self.last_read_time = time.monotonic()

        print(f"StreamHandler initialized for URL: {self.stream_url}")

    def _attempt_connection(self):
        print("Connecting to stream...")
        cap = cv2.VideoCapture(self.stream_url)
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30) 

        if not cap.isOpened():
            print("Error: Could not open stream.")
            return None
        
        print("Stream opened successfully.")
        return cap

    def run(self):
        print("StreamHandler thread started.")
        cap = None
        failure_count = 0

        while self.running:
            if cap is None:
                cap = self._attempt_connection()
                if cap is None:
                    time.sleep(self.RECONNECT_DELAY)
                    continue
                self.last_read_time = time.monotonic()
            
            ret, frame = cap.read()
            
            read_time_end = time.monotonic()

            if not ret:
                failure_count += 1
                print(f"Warning: Failed to grab frame. (Failure {failure_count}/{self.MAX_CONSECUTIVE_FAILURES})")
                
                if failure_count >= self.MAX_CONSECUTIVE_FAILURES:
                    print("Critical: Max consecutive failures reached. Releasing and attempting full reconnection...")
                    cap.release()
                    cap = None
                    time.sleep(self.RECONNECT_DELAY)
                    failure_count = 0
                else:
                    time.sleep(0.1) 
                continue
            
            failure_count = 0

            with self.data_lock:
                self.shared_data['latest_frame'] = frame

            time.sleep(0.0042) #This number works


        if cap:
            cap.release()
        print("StreamHandler thread stopped.")

    def stop(self):
        print("Stopping StreamHandler...")
        self.running = False