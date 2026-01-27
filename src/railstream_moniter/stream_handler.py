import cv2
import threading
import time
import queue
from utils import print_time


class StreamHandler(threading.Thread):

    INIT_TARGET_FPS = 3.0
    MIN_FPS = 5
    MAX_FPS = 60
    RECONNECT_DELAY = 5.0
    MAX_CONSECUTIVE_FAILURES = 5
    QUEUE_SIZE = 128

    TARGET_QUEUE_SIZE = 30

    def __init__(self, config, shared_data, data_lock):
        super().__init__()
        self.stream_url = config.get("Stream", "url")
        self.shared_data = shared_data
        self.data_lock = data_lock
        self.running = True

        self.Kp = 1e-8
        self.Ki = 0
        self.Kd = 0

        self.error_integral = 0
        self.last_error = 0
        self.last_pid_time = time.monotonic()

        self.frame_queue = queue.Queue(maxsize=self.QUEUE_SIZE)

        # Timing state
        self.base_frame_time = 1.0 / self.INIT_TARGET_FPS

        print_time(f"StreamHandler: Dynamic Pacing enabled. Target Queue: {self.TARGET_QUEUE_SIZE}")

    def _attempt_connection(self):
        print_time("Connecting...")
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def run(self):
        print_time("StreamHandler thread started.")
        cap = None
        
        next_update_time = time.monotonic()

        while self.running:
            if cap is None:
                cap = self._attempt_connection()
                if cap is None:
                    time.sleep(self.RECONNECT_DELAY)
                    continue

            ret, frame = cap.read()

            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                cap.release()
                cap = None
                time.sleep(self.RECONNECT_DELAY)
                pass

            now = time.monotonic()
            #next_update_time = now + self.base_frame_time + self.calculate_PID()
            next_update_time = now + self.calculate_PID()

            if now >= next_update_time:
                try:
                    pushed_frame = self.frame_queue.get_nowait()

                    with self.data_lock:
                        self.shared_data["latest_frame"] = pushed_frame
                        self.shared_data["frame_id"] = (
                            self.shared_data.get("frame_id", 0) + 1
                        )

                
                except queue.Empty:
                    next_update_time = now
                    print_time("queue empty", "red")
            else:
                print_time("Loop Skipped", "red")

    def stop(self):
        print_time("Stopping StreamHandler...")
        self.running = False
    
    def calculate_PID(self):
        q_size = self.frame_queue.qsize()
        error = (self.TARGET_QUEUE_SIZE - q_size) * self.base_frame_time


        p_out = error * self.Kp

        output = p_out

        print_time(f"Queue count: {q_size} | Err: {error} | Output: {output}")

        return output