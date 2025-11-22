import cv2
import threading
import time

class StreamHandler(threading.Thread):

    def __init__(self, config, shared_data, data_lock):

        super().__init__()
        self.stream_url = config.get('Stream', 'url')
        self.shared_data = shared_data
        self.data_lock = data_lock

        self.running = True
        
        print(f"StreamHandler initialized for URL: {self.stream_url}")

    def run(self):

        print("StreamHandler thread started.")
        
        cap = None
        while self.running:
            try:
                if cap is None:
                    print("Connecting to stream...")
                    cap = cv2.VideoCapture(self.stream_url)
                    if not cap.isOpened():
                        print("Error: Could not open stream. Retrying in 5s...")
                        cap = None
                        time.sleep(5)
                        continue
                    else:
                        print("Stream opened successfully.")

                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to grab frame or stream ended. Reconnecting...")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue

                with self.data_lock:
                    self.shared_data['latest_frame'] = frame

                time.sleep(0.005) 

            except Exception as e:
                print(f"An error occurred in StreamHandler: {e}")
                if cap:
                    cap.release()
                cap = None
                time.sleep(5)

        if cap:
            cap.release()
        print("StreamHandler thread stopped.")

    def stop(self):
        """
_        Signals the thread to stop.
        """
        print("Stopping StreamHandler...")
        self.running = False