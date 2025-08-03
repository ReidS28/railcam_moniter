import sys
import os

# SUPPRESS ALL OUTPUT IMMEDIATELY
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Set all environment variables first
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_PRINT_FIRST_LINE'] = 'False'
os.environ['WDM_LOG'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

import time
import cv2
import torch
import threading
import queue
import logging
import warnings
from datetime import datetime
import numpy as np

# SUPPRESS ALL LOGS AND WARNINGS
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('selenium').setLevel(logging.CRITICAL)
logging.getLogger('webdriver_manager').setLevel(logging.CRITICAL)
logging.getLogger('WDM').setLevel(logging.CRITICAL)
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from ultralytics import YOLO
from slack_sdk import WebClient

# Restore stdout/stderr for essential output only
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

from stream_url_fetcher import get_m3u8

# Slack configuration
slack_token = "xoxb-8678201375584-9289783952051-BanAx2McG4hrWrgf3nCg0D1Z"
user_id = "U08KY5XBKJ4"

# Initialize Slack client
client = WebClient(token=slack_token)

def send_slack_photo(image_path, message="", user_id=None, channel=None):
    if not os.path.exists(image_path):
        return False
    
    try:
        if user_id:
            dm_response = client.conversations_open(users=[user_id])
            if not dm_response["ok"]:
                return False
            target = dm_response["channel"]["id"]
        elif channel:
            target = channel
        else:
            return False
        
        response = client.files_upload_v2(
            file=image_path,
            filename=os.path.basename(image_path),
            title=f"Train Detection - {os.path.basename(image_path)}",
            initial_comment=message,
            channel=target
        )
        
        return response["ok"]
            
    except Exception:
        return False

def send_slack_message(message, user_id=None, channel=None):
    target = user_id if user_id else channel
    if not target:
        return False
    
    try:
        response = client.chat_postMessage(channel=target, text=message)
        return response["ok"]
    except Exception:
        return False

def handle_train(path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"🚂 Train Detected @ {timestamp}"
    
    success = send_slack_photo(path, message, user_id=user_id)
    
    if not success:
        fallback_message = f"{message}\n\n📸 Image location: {path}"
        send_slack_message(fallback_message, user_id=user_id)
    
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# --- YOLO Setup ---
print("Getting stream URL...")
#stream_url = get_m3u8("https://railstream.net/camera-lineup/delaware-railcam")
stream_url = "https://edge01.railstream.net/Live_Web_Free/CPT_CAM01/playlist_dvr_timeshift-0-1800.m3u8"
if not stream_url:
    print("Failed to get stream URL")
    exit()

print(f"Stream URL: {stream_url}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
print("Loading YOLO model...")
temp_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
model = YOLO("yolov8n.pt", verbose=False).to(device)
sys.stdout.close()
sys.stdout = temp_stdout
print("YOLO model loaded")

# Global variables
train_present = False
last_train_seen = 0
frame_count = 0
start_time = time.time()
first_detect_time = None
screenshot_taken = False
last_box = None
last_center = None
last_detection_time = 0
connection_status = "Disconnected"

# Queues
detection_queue = queue.Queue(maxsize=2)
slack_queue = queue.Queue()

class StreamCapture:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.cap = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_reconnect_time = 0
        self.reconnect_delay = 5  # seconds
        
    def connect(self):
        """Attempt to connect to the stream"""
        global connection_status
        
        if self.cap is not None:
            self.cap.release()
            
        print(f"Attempting to connect to stream... (attempt {self.reconnect_attempts + 1})")
        connection_status = "Connecting..."
        
        # Create new capture with optimized settings
        self.cap = cv2.VideoCapture(self.stream_url)
        
        # Set optimal parameters for streaming
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
        
        # Set timeout properties if available
        try:
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
        except:
            pass
        
        # Test the connection
        ret, frame = self.cap.read()
        if ret and frame is not None:
            connection_status = "Connected"
            self.reconnect_attempts = 0
            print("Stream connected successfully")
            return True
        else:
            connection_status = "Failed"
            self.cap.release()
            self.cap = None
            print("Failed to connect to stream")
            return False
    
    def read(self):
        """Read frame with error handling"""
        global connection_status
        
        if self.cap is None:
            return False, None
            
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            connection_status = "Connection Lost"
            return False, None
            
        return True, frame
    
    def should_reconnect(self):
        """Check if we should attempt reconnection"""
        current_time = time.time()
        
        if (self.reconnect_attempts < self.max_reconnect_attempts and 
            current_time - self.last_reconnect_time > self.reconnect_delay):
            self.last_reconnect_time = current_time
            self.reconnect_attempts += 1
            return True
        
        return False
    
    def release(self):
        if self.cap is not None:
            self.cap.release()

class SlackWorker(QThread):
    def run(self):
        while True:
            try:
                image_path = slack_queue.get(timeout=1)
                if image_path is None:
                    break
                handle_train(image_path)
                slack_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

class DetectionWorker(QThread):
    detection_update = pyqtSignal(bool, tuple, tuple)
    
    def run(self):
        global train_present, last_train_seen, last_box, last_center
        global first_detect_time, screenshot_taken, last_detection_time

        while True:
            try:
                frame = detection_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            current_time = time.time()
            
            # Rate limit detection to every 500ms for stability
            if current_time - last_detection_time < 0.5:
                continue
                
            last_detection_time = current_time
            
            try:
                # Resize for faster inference
                height, width = frame.shape[:2]
                if width > 320:  # Even smaller for faster processing
                    scale = 320 / width
                    new_width = 320
                    new_height = int(height * scale)
                    detection_frame = cv2.resize(frame, (new_width, new_height))
                    scale_factor = width / 320
                else:
                    detection_frame = frame
                    scale_factor = 1.0

                results = model(detection_frame, verbose=False, conf=0.4)  # Lower confidence for better detection

                detected = False
                max_area = 0
                best_box = None
                best_center = None

                for box in results[0].boxes:
                    cls = model.names[int(box.cls)]
                    if cls == "train":
                        detected = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        if scale_factor != 1.0:
                            x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
                        
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                            best_box = (x1, y1, x2, y2)

                if detected:
                    if not train_present:
                        first_detect_time = current_time
                        screenshot_taken = False

                    if not screenshot_taken and first_detect_time and (current_time - first_detect_time >= 2.0):  # Wait 2 seconds
                        filename = f"detections/train_{int(current_time)}.jpg"
                        os.makedirs("detections", exist_ok=True)
                        cv2.imwrite(filename, frame)
                        screenshot_taken = True

                        try:
                            slack_queue.put_nowait(filename)
                        except queue.Full:
                            pass

                    train_present = True
                    last_train_seen = current_time
                    last_box = best_box
                    last_center = best_center
                    
                    self.detection_update.emit(True, best_box, best_center)
                else:
                    if train_present and (current_time - last_train_seen) > 3.0:  # Wait 3 seconds before clearing
                        train_present = False
                        last_box = None
                        last_center = None
                        first_detect_time = None
                        screenshot_taken = False
                        
                        self.detection_update.emit(False, None, None)
                        
            except Exception as e:
                print(f"Detection error: {e}")
                continue

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    
    def __init__(self, stream_url):
        super().__init__()
        self.stream_capture = StreamCapture(stream_url)
        self.running = True
        
    def run(self):
        global frame_count
        
        while self.running:
            # Try to connect if not connected
            if self.stream_capture.cap is None:
                if self.stream_capture.should_reconnect():
                    if not self.stream_capture.connect():
                        self.status_update.emit(f"Connection failed (attempt {self.stream_capture.reconnect_attempts})")
                        self.msleep(1000)
                        continue
                else:
                    self.msleep(1000)
                    continue
            
            ret, frame = self.stream_capture.read()
            
            if ret and frame is not None:
                frame_count += 1
                self.status_update.emit("Connected")
                
                # Push to detection queue (skip if full)
                try:
                    # Clear old frames from queue
                    while detection_queue.qsize() > 1:
                        try:
                            detection_queue.get_nowait()
                        except queue.Empty:
                            break
                    detection_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
                
                # Emit frame for UI
                self.frame_ready.emit(frame)
                
                # Reset reconnection attempts on successful read
                self.stream_capture.reconnect_attempts = 0
                
            else:
                self.status_update.emit("Connection lost - reconnecting...")
                self.stream_capture.cap = None
                self.msleep(1000)
    
    def stop(self):
        self.running = False
        self.stream_capture.release()

class RailcamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Detection Camera - Railstream Monitor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Current detection state
        self.current_box = None
        self.current_center = None
        self.current_train_present = False
        
        self.init_ui()
        self.init_threads()
        
        # FPS calculation
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_fps)
        self.status_timer.start(1000)  # Update every second
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Connection status
        self.connection_label = QLabel("Connection: Initializing...")
        self.connection_label.setStyleSheet("color: orange; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.connection_label)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)
        
        # Status bar
        status_layout = QHBoxLayout()
        
        # Train status
        self.train_label = QLabel("Train: False")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.train_label.setFont(font)
        self.train_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.train_label)
        
        status_layout.addStretch()
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(font)
        status_layout.addWidget(self.fps_label)
        
        layout.addLayout(status_layout)
        
    def init_threads(self):
        # Video thread
        self.video_thread = VideoThread(stream_url)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.status_update.connect(self.update_connection_status)
        self.video_thread.start()
        
        # Detection thread
        self.detection_thread = DetectionWorker()
        self.detection_thread.detection_update.connect(self.update_detection)
        self.detection_thread.start()
        
        # Slack thread
        self.slack_thread = SlackWorker()
        self.slack_thread.start()
    
    def update_connection_status(self, status):
        color_map = {
            "Connected": "green",
            "Connecting...": "orange",
            "Connection lost - reconnecting...": "red",
            "Failed": "red"
        }
        color = color_map.get(status, "orange")
        self.connection_label.setText(f"Connection: {status}")
        self.connection_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
    
    def update_detection(self, train_present, box, center):
        self.current_train_present = train_present
        self.current_box = box
        self.current_center = center
        
        # Update train status
        if train_present:
            self.train_label.setText("🚂 Train: DETECTED")
            self.train_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        else:
            self.train_label.setText("Train: Monitoring...")
            self.train_label.setStyleSheet("color: #666; font-weight: bold; font-size: 14px;")
    
    def update_frame(self, frame):
        self.frame_count += 1
        
        # Draw detection overlay if present
        display_frame = frame.copy()
        if self.current_box is not None and self.current_center is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.circle(display_frame, self.current_center, 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "TRAIN DETECTED", (int(x1), int(y1) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Convert to Qt format efficiently
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Reset counters every 10 seconds to keep FPS current
            if elapsed > 10:
                self.frame_count = 0
                self.last_fps_time = current_time
    
    def closeEvent(self, event):
        print("Shutting down...")
        
        # Stop video thread
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
            self.video_thread.wait(2000)
        
        # Stop other threads
        if hasattr(self, 'detection_thread'):
            self.detection_thread.terminate()
        if hasattr(self, 'slack_thread'):
            self.slack_thread.terminate()
        
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication([])
    
    # Set application properties
    app.setApplicationName("Railstream Monitor")
    app.setApplicationVersion("2.0")
    
    window = RailcamApp()
    window.show()
    
    print("Application started - GUI should be visible")
    
    sys.exit(app.exec_())