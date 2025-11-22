import cv2
import threading
import time

class StreamHandler(threading.Thread):
    """
    A class to read frames from a video stream in a separate thread.
    
    This "producer" thread continuously reads frames from the specified
    stream URL and updates a shared dictionary with the latest frame,
    protected by a lock.
    """
    def __init__(self, config, shared_data, data_lock):
        """
        Initializes the stream handler.

        Args:
            config (configparser.ConfigParser): Configuration object.
            shared_data (dict): A dictionary shared between threads.
            data_lock (threading.Lock): A lock to ensure thread-safe
                                        access to shared_data.
        """
        super().__init__()
        self.stream_url = config.get('Stream', 'url')
        self.shared_data = shared_data
        self.data_lock = data_lock
        
        # A flag to signal the thread to stop
        self.running = True
        
        print(f"StreamHandler initialized for URL: {self.stream_url}")

    def run(self):
        """
        The main loop of the thread.
        Connects to the stream and continuously reads frames.
        """
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
                    time.sleep(1) # Wait a bit before reconnecting
                    continue

                # We have a valid frame, update the shared data
                with self.data_lock:
                    self.shared_data['latest_frame'] = frame

            except Exception as e:
                print(f"An error occurred in StreamHandler: {e}")
                if cap:
                    cap.release()
                cap = None
                time.sleep(5) # Wait before retrying

        # Cleanup when the loop is stopped
        if cap:
            cap.release()
        print("StreamHandler thread stopped.")

    def stop(self):
        """
_        Signals the thread to stop.
        """
        print("Stopping StreamHandler...")
        self.running = False