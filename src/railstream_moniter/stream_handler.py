import cv2
import threading
import time
import queue
from utils import print_time

class StreamHandler(threading.Thread):

    BASE_FPS = 30.0
    BASE_INTERVAL = 1.0 / BASE_FPS

    MAX_INTERVAL_DEVIATION = BASE_INTERVAL - 1e-6
    MIN_INTERVAL = BASE_INTERVAL - MAX_INTERVAL_DEVIATION
    MAX_INTERVAL = BASE_INTERVAL + MAX_INTERVAL_DEVIATION

    RECONNECT_DELAY = 5.0
    QUEUE_SIZE = 128
    TARGET_QUEUE_SIZE = 30

    def __init__(self, config, shared_data, data_lock):
        super().__init__()
        self.stream_url = config.get("Stream", "url")
        self.shared_data = shared_data
        self.data_lock = data_lock
        self.running = True

        # PID Controller State
        self.Kp = 0.1
        self.Ki = 0
        self.Kd = 0
        self.last_pid_time = time.monotonic()

        # The glue between the threads
        self.frame_queue = queue.Queue(maxsize=self.QUEUE_SIZE)

        print_time(f"StreamHandler: Dynamic Pacing enabled. Target Queue: {self.TARGET_QUEUE_SIZE}")

    def _attempt_connection(self):
        print_time("Connecting...")
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def _capture_worker(self):
        """
        PRODUCER THREAD
        """
        print_time("Capture Worker thread started.")
        cap = None

        while self.running:
            # 1. Handle Connection
            if cap is None or not cap.isOpened():
                print_time("Capture: Attempting connect...", "yellow")
                cap = self._attempt_connection()
                
                if cap is None:
                    print_time("Capture: Connection failed. Retrying...", "red")
                    time.sleep(self.RECONNECT_DELAY)
                    continue
            
            # 2. Read Frame
            ret, frame = cap.read()

            if ret:
                # 3. Put in Queue (Non-blocking drop-oldest strategy)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait() # Remove oldest
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
            else:
                print_time("Capture: Frame read failed. Resetting connection.", "red")
                cap.release()
                cap = None
                time.sleep(1.0) # Prevent tight loop on failure

        # Cleanup when thread finishes
        if cap:
            cap.release()
        print_time("Capture Worker thread stopped.")

    def run(self):
        """
        CONSUMER THREAD (Main Class Thread)
        """
        print_time("StreamHandler Consumer thread started.")

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        
        last_update_time = time.monotonic()
        next_update_time = time.monotonic()

        while self.running:
            now = time.monotonic()

            if now >= next_update_time:
                #print_time(now - next_update_time)
                try:
                    pushed_frame = self.frame_queue.get(timeout=0.1)

                    with self.data_lock:
                        self.shared_data["latest_frame"] = pushed_frame
                        self.shared_data["frame_id"] = (
                            self.shared_data.get("frame_id", 0) + 1
                        )
                    
                    last_update_time = now

                    adjustment = self.calculate_PID()
                    current_interval = self.BASE_INTERVAL + adjustment
                    
                    current_interval = max(self.MIN_INTERVAL, min(current_interval, self.MAX_INTERVAL))

                    next_update_time = last_update_time + current_interval

                    # Debug output
                    current_fps = 1.0 / current_interval
                    color = "white"
                    if adjustment > 0:
                        color = "blue"
                    elif adjustment < 0:
                        color = "red"
                    print_time(
                        f"result: {adjustment:.4f} | "
                        f"fps: {current_fps:.2f} | "
                        f"frame time: {current_interval:.4f} | "
                        f"queue: {self.frame_queue.qsize()}",
                        color
                    )

                except queue.Empty:
                    time.sleep(0.01) 
            else:
                sleep_time = (next_update_time - now) * 0.2
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
                #print_time("Loop Skipped", "red")

    def stop(self):
        print_time("Stopping StreamHandler...")
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
    
    def calculate_PID(self):
        q_size = self.frame_queue.qsize()
        
        error = (self.TARGET_QUEUE_SIZE - q_size) * self.BASE_INTERVAL

        p_out = error * self.Kp

        output = p_out

        return output