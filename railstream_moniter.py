import numpy as np
import cv2 as cv
import time
import threading
import asyncio
from ultralytics import YOLO
from collections import deque

# region --- CONSTANTS ---
# Stream configuration
STREAM_URL = "https://edge01.railstream.net/Live_Web_Free/CPT_CAM01/playlist_dvr_timeshift-0-1800.m3u8"
STREAM_URL = "2025-07-31 14-40-10_loop.mp4"

# Display and performance settings
SCALE_FACTOR = 0.5
TARGET_WIDTH = None
TARGET_HEIGHT = None
SHOW_STATS = True

# Buffering and refresh
REFRESH_INTERVAL_SECONDS = 300
MIN_BUFFER_SIZE = 10
TARGET_BUFFER_SIZE = 20
BUFFER_MAXLEN = 60

# Detection settings
DETECTION_INTERVAL = 0.5
DETECTION_ENABLED = True
DETECTION_MODEL_PATH = 'yolov8n.pt'
DETECTION_CLASS_ID = 6
DETECTION_CONFIDENCE_THRESHOLD = 0.4
DETECTION_TIMEOUT = 5

# UI and control
WAITKEY_DELAY_MS = 1
FONT_HERSHEY = cv.FONT_HERSHEY_SIMPLEX
STATS_FONT_SCALE = 1.0
STATS_FONT_THICKNESS = 2
STATS_COLOR_FPS = (0, 255, 0)
STATS_COLOR_BUFFER = (0, 255, 0)
STATS_COLOR_SIZE = (255, 255, 255)
STATS_COLOR_DETECTION = (255, 255, 0)
STATS_COLOR_BUFFERING = (0, 0, 255)
# endregion

class HLSStreamer:
    def __init__(self, url, scale_factor=SCALE_FACTOR, width=TARGET_WIDTH, height=TARGET_HEIGHT, 
                refresh_interval=REFRESH_INTERVAL_SECONDS, detection_interval=DETECTION_INTERVAL):
        self.url = url
        self.frame_buffer = deque(maxlen=BUFFER_MAXLEN)
        self.running = False
        self.cap = None
        self.show_stats = SHOW_STATS
        
        # Scaling options
        self.scale_factor = scale_factor
        self.target_width = width
        self.target_height = height
        self.original_size = None
        
        # Auto-refresh to stay current
        self.refresh_interval = refresh_interval
        self.last_refresh_time = time.time()
        
        # Buffer management
        self.min_buffer_size = MIN_BUFFER_SIZE
        self.target_buffer_size = TARGET_BUFFER_SIZE
        self.buffering = True
        
        # Stats
        self.frames_read = 0
        self.frames_displayed = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # Async detection settings
        self.detection_interval = detection_interval
        self.last_detection_time = 0
        self.detection_enabled = DETECTION_ENABLED
        self.detection_queue = asyncio.Queue(maxsize=10)
        self.detection_loop = None
        self.detection_task = None
        self.model = YOLO(DETECTION_MODEL_PATH)
        
        # detection states
        self.train_detected = False
        self.last_train_detected_time = time.time()
        
        # Threading
        self.read_thread = None
        self.buffer_lock = threading.Lock()
    
    async def run_detection(self, frame):
        """Async frame detection - detects trains using YOLOv8n"""
        if frame is None:
            return
            
        loop = asyncio.get_event_loop()
        
        def analyze_frame():
            try:
                results = self.model(frame, verbose=False)
                
                train_in_frame = False
                train_confidences = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls.item())
                            confidence = float(box.conf.item())
                            
                            if class_id == DETECTION_CLASS_ID and confidence > DETECTION_CONFIDENCE_THRESHOLD:
                                train_in_frame = True
                                train_confidences.append(confidence)
                
                if train_in_frame:
                    if not self.train_detected:
                        print(f"train first detected | max confidence: {max(train_confidences)}")
                        self.train_detected = True
                    self.last_train_detected_time = time.time()
                elif (time.time() - self.last_train_detected_time) > DETECTION_TIMEOUT:
                    if self.train_detected:
                        self.train_detected = False
                        print("train not detected")
                        
            except Exception as e:
                return f"Detection error: {e}"
        
        result = await loop.run_in_executor(None, analyze_frame)
    
    async def detection_worker(self):
        """Async worker that processes detection frames independently"""
        print("Starting async detection worker...")
        
        while self.running:
            try:
                frame = await asyncio.wait_for(self.detection_queue.get(), timeout=1.0)
                await self.run_detection(frame)
                self.detection_queue.task_done()
                await asyncio.sleep(0.01)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Detection error: {e}")
                await asyncio.sleep(0.1)
        
        print("Async detection worker stopped")

    def start_async_detection(self):
        """Start the async detection loop in a separate thread"""
        def run_async_loop():
            self.detection_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.detection_loop)
            
            self.detection_task = self.detection_loop.create_task(self.detection_worker())
            
            try:
                self.detection_loop.run_until_complete(self.detection_task)
            except Exception as e:
                print(f"Async detection loop error: {e}")
            finally:
                self.detection_loop.close()
        
        self.detection_thread = threading.Thread(target=run_async_loop, daemon=True)
        self.detection_thread.start()
    
    def add_frame_for_detection(self, frame):
        """Add frame to detection queue (thread-safe)"""
        if self.detection_loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.detection_queue.put(frame.copy()), 
                    self.detection_loop
                )
                future.result(timeout=0.001)
            except (asyncio.TimeoutError, RuntimeError):
                pass
    
    def connect(self):
        """Connect to the stream"""
        print("Connecting to stream...")
        self.cap = cv.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print("Cannot open stream")
            return False
            
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        
        print("Stream opened successfully!")
        return True
    
    def read_frames(self):
        """Background thread to continuously read frames"""
        consecutive_failures = 0
        last_read_time = time.time()
        
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue
                
            try:
                ret, frame = self.cap.read()
            except cv.error as e:
                print(f"Caught OpenCV exception during read: {e}")
                ret = False
                frame = None
            
            current_time = time.time()
            
            if ret:
                with self.buffer_lock:
                    self.frame_buffer.append((frame, current_time))
                    self.frames_read += 1
                
                consecutive_failures = 0
                last_read_time = current_time
                
                buffer_size = len(self.frame_buffer)
                if buffer_size < self.min_buffer_size:
                    time.sleep(0.001)
                elif buffer_size < self.target_buffer_size:
                    time.sleep(0.01)
                else:
                    time.sleep(0.03)
                    
            else:
                consecutive_failures += 1
                if consecutive_failures > 15:
                    print("Too many read failures, attempting reconnect...")
                    self.reconnect("Read failures")
                    consecutive_failures = 0
                elif current_time - last_read_time > 5.0:
                    print("No frames received for 5 seconds, reconnecting...")
                    self.reconnect("No frames timeout")
                    last_read_time = current_time
                
                time.sleep(0.1)
    
    def reconnect(self, reason="Manual reconnect"):
        """Reconnect to the stream"""
        if self.cap:
            self.cap.release()
        
        print(f"Reconnecting... ({reason})")
        time.sleep(2)
        
        self.cap = cv.VideoCapture(self.url)
        if self.cap.isOpened():
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            print("Reconnected successfully")
            with self.buffer_lock:
                self.frame_buffer.clear()
            self.buffering = True
            self.last_refresh_time = time.time()
        else:
            print("Reconnection failed")
    
    def scale_frame(self, frame):
        """Scale frame according to settings"""
        if frame is None:
            return None
            
        if self.original_size is None:
            self.original_size = (frame.shape[1], frame.shape[0])
        
        if self.scale_factor == 1.0 and self.target_width is None and self.target_height is None:
            return frame
        
        if self.target_width is not None and self.target_height is not None:
            new_width, new_height = self.target_width, self.target_height
        elif self.target_width is not None:
            aspect_ratio = frame.shape[0] / frame.shape[1]
            new_width = self.target_width
            new_height = int(self.target_width * aspect_ratio)
        elif self.target_height is not None:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        else:
            new_width = int(frame.shape[1] * self.scale_factor)
            new_height = int(frame.shape[0] * self.scale_factor)
        
        scaled_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        return scaled_frame
    
    def get_frame(self):
        """Get a frame from buffer with smart buffering"""
        with self.buffer_lock:
            buffer_size = len(self.frame_buffer)
            
            if self.buffering and buffer_size < self.min_buffer_size:
                return None, True
            elif self.buffering and buffer_size >= self.min_buffer_size:
                self.buffering = False
                print(f"Buffering complete, starting playback with {buffer_size} frames")
            
            if not self.buffering and buffer_size < 3:
                self.buffering = True
                print("Buffer underrun, rebuffering...")
                return None, True
            
            if buffer_size > 0:
                frame_data = self.frame_buffer.popleft()
                return frame_data[0], False
            
            return None, False

    def start(self):
        """Start the streaming"""
        if not self.connect():
            return
            
        self.running = True
        
        self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.read_thread.start()
        
        self.start_async_detection()
        
        print("Press 'q' to quit, 's' to show/hide stats, 'r' to refresh stream, 'd' to toggle detection")
        print(f"Auto-refresh enabled every {self.refresh_interval/60:.1f} minutes")
        print(f"Async frame detection enabled every {self.detection_interval} seconds")
        print("Buffering initial frames...")
        
        last_display_time = time.time()
        frame_interval = 1.0 / 25.0
        
        while self.running:
            frame, is_buffering = self.get_frame()
            if frame is not None:
                frame = self.scale_frame(frame)
                cv.imshow('HLS Stream', frame)
                cv.waitKey(1)
                break
            else:
                print("Waiting for first frame...")
                time.sleep(0.5)

        if not self.running:
            return

        while self.running:
            current_time = time.time()
            
            if current_time - self.last_refresh_time > self.refresh_interval:
                print(f"Periodic refresh after {self.refresh_interval/60:.1f} minutes to stay current")
                self.reconnect("Periodic refresh")
            
            frame, is_buffering = self.get_frame()
            
            if frame is not None:
                frame = self.scale_frame(frame)
                
                if (self.detection_enabled and 
                    current_time - self.last_detection_time >= self.detection_interval):
                    self.add_frame_for_detection(frame)
                    self.last_detection_time = current_time
                
                self.frames_displayed += 1
                
                if current_time - self.last_stats_time >= 1.0:
                    buffer_size = len(self.frame_buffer)
                    if buffer_size > self.target_buffer_size:
                        frame_interval = 1.0 / 30.0
                    elif buffer_size < 10:
                        frame_interval = 1.0 / 20.0
                    else:
                        frame_interval = 1.0 / 25.0
                    self.last_stats_time = current_time
                
                if self.show_stats:
                    elapsed = current_time - self.start_time
                    display_fps = self.frames_displayed / elapsed if elapsed > 0 else 0
                    buffer_size = len(self.frame_buffer)
                    
                    text_scale = STATS_FONT_SCALE
                    text_thickness = STATS_FONT_THICKNESS
                    bottom_y = frame.shape[0] - 10
                    
                    cv.putText(frame, f"FPS: {display_fps:.1f}", (10, bottom_y),
                                cv.FONT_HERSHEY_SIMPLEX, text_scale, STATS_COLOR_FPS, text_thickness)
                    
                    (fps_w, fps_h), _ = cv.getTextSize(f"FPS: {display_fps:.1f}", FONT_HERSHEY, text_scale, text_thickness)
                    (buffer_w, buffer_h), _ = cv.getTextSize(f"Buffer: {buffer_size}", FONT_HERSHEY, text_scale, text_thickness)
                    
                    cv.putText(frame, f"Buffer: {buffer_size}", (10, bottom_y - fps_h - 10),
                                FONT_HERSHEY, text_scale, STATS_COLOR_BUFFER, text_thickness)
                    
                    if self.original_size:
                        size_text = f"Size: {self.original_size[0]}x{self.original_size[1]} -> {frame.shape[1]}x{frame.shape[0]}"
                        (size_w, size_h), _ = cv.getTextSize(size_text, FONT_HERSHEY, 0.5, 1)
                        cv.putText(frame, size_text, (10, bottom_y - fps_h - buffer_h - 20),
                                    FONT_HERSHEY, 0.5, STATS_COLOR_SIZE, 1)
                    
                    detection_text = f"Detection: {'ON' if self.detection_enabled else 'OFF'} ({self.detection_interval}s, ASYNC)"
                    (det_w, det_h), _ = cv.getTextSize(detection_text, FONT_HERSHEY, 0.5, 1)
                    cv.putText(frame, detection_text, (10, bottom_y - fps_h - buffer_h - size_h - 30),
                                FONT_HERSHEY, 0.5, STATS_COLOR_DETECTION, 1)
                    
                    if is_buffering:
                        buffering_text = "BUFFERING..."
                        (buff_w, buff_h), _ = cv.getTextSize(buffering_text, FONT_HERSHEY, text_scale, text_thickness)
                        cv.putText(frame, buffering_text, (10, bottom_y - fps_h - buffer_h - size_h - det_h - 40),
                                    FONT_HERSHEY, text_scale, STATS_COLOR_BUFFERING, text_thickness)
                
                cv.imshow('HLS Stream', frame)
                
                elapsed_since_display = current_time - last_display_time
                if elapsed_since_display < frame_interval:
                    time.sleep(frame_interval - elapsed_since_display)
                last_display_time = time.time()
            
            elif is_buffering:
                if int(current_time) % 2 == 0:
                    buffer_size = len(self.frame_buffer)
                time.sleep(0.1)
            
            else:
                time.sleep(0.01)
            
            key = cv.waitKey(WAITKEY_DELAY_MS) & 0xFF
            
            if key == ord('q') or cv.getWindowProperty('HLS Stream', cv.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Stats display: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('r'):
                print("Manual refresh requested")
                self.reconnect("Manual refresh")
            elif key == ord('d'):
                self.detection_enabled = not self.detection_enabled
                print(f"Frame detection: {'ON' if self.detection_enabled else 'OFF'}")
        
        self.stop()
    
    def stop(self):
        """Stop streaming and cleanup"""
        print("Stopping stream...")
        self.running = False
        
        if self.detection_task and self.detection_loop and not self.detection_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._cancel_async_detection(), 
                self.detection_loop
            )
            try:
                future.result(timeout=2.0)
            except (asyncio.TimeoutError, RuntimeError):
                pass
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3)
        
        if self.cap:
            self.cap.release()
        
        cv.destroyAllWindows()
        print("Stream closed.")
    
    async def _cancel_async_detection(self):
        """Helper to cancel async detection task"""
        if self.detection_task:
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass

# Usage - Always async, super simple!
if __name__ == "__main__":
    
    # Simple usage - detection every 0.5 seconds (2x per second)
    streamer = HLSStreamer(STREAM_URL, scale_factor=SCALE_FACTOR, detection_interval=DETECTION_INTERVAL)
    
    # High frequency detection - no performance impact!
    # streamer = HLSStreamer(STREAM_URL, detection_interval=0.1)  # 10x per second
    
    # Lower frequency detection
    # streamer = HLSStreamer(STREAM_URL, detection_interval=2.0)  # every 2 seconds
    
    try:
        streamer.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        streamer.stop()