import cv2
import threading
import time
import queue
from utils import print_time

class StreamHandler(threading.Thread):

    BASE_FPS = 30.0
    BASE_INTERVAL = 1.0 / BASE_FPS

    MAX_INTERVAL_DEVIATION = BASE_INTERVAL - 1e-4
    MIN_INTERVAL = BASE_INTERVAL - MAX_INTERVAL_DEVIATION
    MAX_INTERVAL = BASE_INTERVAL + MAX_INTERVAL_DEVIATION

    RECONNECT_DELAY = 5.0
    QUEUE_SIZE = 128
    TARGET_QUEUE_SIZE = 30

    def __init__(self, config, shared_data, data_lock, new_frame_event):
        super().__init__()
        self.stream_url = config.get("Stream", "url")
        self.shared_data = shared_data
        self.data_lock = data_lock
        self.new_frame_event = new_frame_event
        self.running = True

        # PID Controller State
        self.Kp =0.04
        self.Ki = 0.001
        self.Kd = 0.001

        self.queue_error_intigral = 0
        self.last_pid_error = 0

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
                cap = self._attempt_connection()
                
                if cap is None:
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
                try:
                    pushed_frame = self.frame_queue.get(timeout=0.1)

                    with self.data_lock:
                        self.shared_data["latest_frame"] = pushed_frame
                        self.shared_data["frame_id"] = (
                            self.shared_data.get("frame_id", 0) + 1
                        )
                    self.new_frame_event.set()
                    
                    last_update_time = now

                    adjustment = self.calculate_PID()
                    current_interval = self.BASE_INTERVAL + adjustment
                    
                    current_interval = max(self.MIN_INTERVAL, min(current_interval, self.MAX_INTERVAL))

                    next_update_time = last_update_time + current_interval

                    current_fps = 1.0 / current_interval
                    color_sensitivity = 30
                    color = (255, 255, 255)
                    error = self.TARGET_QUEUE_SIZE - self.frame_queue.qsize()
                    saturation_offset = min(abs(error) * color_sensitivity, 255)
                    if error > 0:
                        color = (255 - saturation_offset, 255 - saturation_offset, 255)
                    elif error < 0:
                        color = (255, 255 - saturation_offset, 255 - saturation_offset)
                    print_time(
                        #f"result: {adjustment:.4f} | "
                        #f"fps: {current_fps:.2f} | "
                        #f"frame time: {current_interval:.4f} | "
                        f"queue: {self.frame_queue.qsize()} | "
                        f"intigral: {(self.queue_error_intigral / self.BASE_INTERVAL):.0f}",
                        color
                    )

                except queue.Empty:
                    time.sleep(0.01) 
            else:
                sleep_time = (next_update_time - now) * 0.2
                if sleep_time > 0.001:
                    time.sleep(sleep_time)
    
    def stop(self):
        print_time("Stopping StreamHandler...")
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
    
    def calculate_PID(self):
        q_size = self.frame_queue.qsize()
        
        error = (self.TARGET_QUEUE_SIZE - q_size) * self.BASE_INTERVAL
        self.queue_error_intigral += error

        p_out = error * self.Kp
        i_out = self.queue_error_intigral * self.Ki
        d_out = (self.last_pid_error - error) * self.Kd

        output = p_out + i_out + d_out

        self.last_pid_error = error

        return output