import sys
import os
import time
import subprocess
import threading
import numpy as np
from collections import deque

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# Stream URL
stream_url = "https://edge01.railstream.net/Live_Web_Free/CPT_CAM01/playlist_dvr_timeshift-0-1800.m3u8"

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = False
        self.process = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        # Frame buffer for smooth playback
        self.buffer_size = 30  # A slightly larger buffer for better resilience
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Stats
        self.frames_received = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()
        
    def run(self):
        self.running = True
        
        # Start buffer consumer thread
        buffer_thread = threading.Thread(target=self.consume_buffer, daemon=True)
        buffer_thread.start()
        
        while self.running:
            try:
                self.connect_stream()
            except Exception as e:
                print(f"Stream error: {e}")
                if self.running:
                    self.status_update.emit("Connection lost - reconnecting...")
                    time.sleep(2)
                    
    def connect_stream(self):
        self.status_update.emit("Connecting...")
        
        # FFmpeg command to read stream and output raw video frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-an',  # no audio
            '-sn',  # no subtitles
            '-vf', 'scale=800:600',  # scale to consistent size
            '-r', '33',  # Set to 33 fps, should match consume_buffer
            '-loglevel', 'error',
            'pipe:1'
        ]
        
        try:
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            self.status_update.emit("Connected")
            self.reconnect_attempts = 0
            
            # Calculate frame size (width * height * 3 channels)
            width, height = 800, 600
            frame_size = width * height * 3

            while self.running and self.process.poll() is None:
                try:
                    # Read frame data
                    raw_frame = self.process.stdout.read(frame_size)
                    
                    if not raw_frame or len(raw_frame) != frame_size:
                        if self.running:
                            print("Stream ended or incomplete frame")
                            break
                        else:
                            return
                    
                    # Convert raw bytes to numpy array
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame.reshape((height, width, 3))
                    
                    # Convert RGB to BGR for consistency
                    frame_bgr = frame[:, :, ::-1].copy()
                    
                    # Add frame to buffer
                    with self.buffer_lock:
                        self.frame_buffer.append(frame_bgr)
                        
                    # --- Re-added Console Logging Block ---
                    self.frames_received += 1
                    current_time = time.time()
                    interval = 1.0
                    if current_time - self.last_stats_time >= interval:
                        input_fps = self.frames_received / (current_time - self.last_stats_time)
                        print(f"Producer FPS: {input_fps:.1f}, Buffer size: {len(self.frame_buffer)}, Dropped: {self.frames_dropped}")
                        self.frames_received = 0
                        self.frames_dropped = 0  # Resetting dropped count for this log interval
                        self.last_stats_time = current_time
                    # --- End of Console Logging Block ---

                except Exception as e:
                    print(f"Frame processing error: {e}")
                    break
                    
        except FileNotFoundError:
            self.status_update.emit("Failed - FFmpeg not found")
            print("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
            return
            
        except Exception as e:
            print(f"FFmpeg error: {e}")
            
        finally:
            self.cleanup_process()
            
        # Handle reconnection
        if self.running:
            self.reconnect_attempts += 1
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.status_update.emit(f"Reconnecting... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                time.sleep(min(2 ** self.reconnect_attempts, 10))  # Exponential backoff
            else:
                self.status_update.emit("Failed - Max reconnection attempts reached")
                self.running = False
    
    def consume_buffer(self):
        """Consume frames from buffer with dynamic frame dropping to prevent drift."""
        target_fps = 60
        frame_time = 1.0 / target_fps
        
        while self.running:
            start_time = time.time()
            
            frame = None
            
            with self.buffer_lock:
                # Get the number of frames currently in the buffer
                buffer_level = len(self.frame_buffer)
                
                # If the buffer is building up, aggressively drop frames
                if buffer_level > 2:
                    while len(self.frame_buffer) > 1:
                        self.frame_buffer.popleft()
                        self.frames_dropped += 1
                
                # Grab the newest frame if available
                if self.frame_buffer:
                    frame = self.frame_buffer.popleft()
            
            if frame is not None:
                self.frame_ready.emit(frame)
            else:
                # If the buffer is empty, yield briefly to avoid a tight loop
                time.sleep(0.001)

            # Maintain steady frame rate
            elapsed = time.time() - start_time
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    def cleanup_process(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1)
                except:
                    pass
            finally:
                self.process = None
                
    def stop(self):
        self.running = False
        self.cleanup_process()

class StreamPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FFmpeg Stream Player")
        self.setGeometry(100, 100, 1200, 800)
        
        # Frame tracking
        self.display_frame_count = 0
        self.last_display_time = time.time()
        
        self.init_ui()
        self.init_video()
        
        # FPS timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_fps)
        self.status_timer.start(2000)
        
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
        
        # FPS counter
        self.fps_label = QLabel("FPS: --")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.fps_label.setFont(font)
        layout.addWidget(self.fps_label)
        
        # Buffer status
        self.buffer_label = QLabel("Buffer: --")
        self.buffer_label.setStyleSheet("color: blue; font-size: 10px;")
        layout.addWidget(self.buffer_label)
        
        # Dropped frames counter
        self.dropped_label = QLabel("Dropped Frames: 0")
        self.dropped_label.setStyleSheet("color: red; font-size: 10px;")
        layout.addWidget(self.dropped_label)
        
        # FFmpeg info
        self.info_label = QLabel("Using FFmpeg for stream processing with frame buffer")
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.info_label)
        
    def init_video(self):
        # Start video thread
        self.video_thread = VideoThread(stream_url)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.status_update.connect(self.update_connection_status)
        self.video_thread.start()
    
    def update_connection_status(self, status):
        color_map = {
            "Connected": "green",
            "Connecting...": "orange",
            "Connection lost - reconnecting...": "red",
            "Failed": "red",
            "Failed - FFmpeg not found": "red",
            "Failed - Max reconnection attempts reached": "red"
        }
        
        if "Reconnecting..." in status:
            color = "orange"
        else:
            color = color_map.get(status, "orange")
            
        self.connection_label.setText(f"Connection: {status}")
        self.connection_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
    
    def update_frame(self, frame):
        try:
            self.display_frame_count += 1
            
            # Convert BGR to RGB for Qt
            rgb_frame = frame[:, :, ::-1]
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Ensure the array is contiguous and convert to bytes
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_display_time
        
        if elapsed > 0 and self.display_frame_count > 0:
            fps = self.display_frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Update buffer status
            if hasattr(self.video_thread, 'frame_buffer'):
                buffer_size = len(self.video_thread.frame_buffer)
                buffer_status = f"Buffer: {buffer_size}/{self.video_thread.buffer_size}"
                if buffer_size < 3:
                    color = "red"
                elif buffer_size > self.video_thread.buffer_size * 0.7:
                    color = "orange"
                else:
                    color = "green"
                self.buffer_label.setText(buffer_status)
                self.buffer_label.setStyleSheet(f"color: {color}; font-size: 10px;")
                
                # Update dropped frames count
                dropped_frames = self.video_thread.frames_dropped
                self.dropped_label.setText(f"Dropped Frames: {dropped_frames}")
                # Reset the dropped frames counter for the next interval
                self.video_thread.frames_dropped = 0
            
            # Reset counters
            self.display_frame_count = 0
            self.last_display_time = current_time
    
    def closeEvent(self, event):
        print("Shutting down...")
        
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
            self.video_thread.wait(3000)
        
        event.accept()

if __name__ == '__main__':
    app = QApplication([])
    
    app.setApplicationName("FFmpeg Stream Player")
    app.setApplicationVersion("1.0")
    
    player = StreamPlayer()
    player.show()
    
    print("FFmpeg stream player started")
    
    sys.exit(app.exec_())