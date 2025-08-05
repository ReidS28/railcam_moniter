import numpy as np
import cv2 as cv
import time
import threading
from collections import deque

# Stream URL
stream_url = "https://edge01.railstream.net/Live_Web_Free/CPT_CAM01/playlist_dvr_timeshift-0-1800.m3u8"

class HLSStreamer:
    def __init__(self, url, scale_factor=1.0, width=None, height=None, refresh_interval=300):
        self.url = url
        self.frame_buffer = deque(maxlen=60)  # Much larger buffer
        self.running = False
        self.cap = None
        self.show_stats = True
        
        # Scaling options
        self.scale_factor = scale_factor
        self.target_width = width
        self.target_height = height
        self.original_size = None
        
        # Auto-refresh to stay current
        self.refresh_interval = refresh_interval  # seconds (default 5 minutes)
        self.last_refresh_time = time.time()
        
        # Buffer management
        self.min_buffer_size = 10  # Don't start displaying until we have this many frames
        self.target_buffer_size = 20  # Try to maintain this many frames
        self.buffering = True
        
        # Stats
        self.frames_read = 0
        self.frames_displayed = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
        # Threading
        self.read_thread = None
        self.buffer_lock = threading.Lock()
        
    def connect(self):
        """Connect to the stream"""
        print("Connecting to stream...")
        self.cap = cv.VideoCapture(self.url)
        
        if not self.cap.isOpened():
            print("Cannot open stream")
            return False
            
        # More aggressive settings for HLS
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
                # Catch the OpenCV C++ exception and treat it as a read failure
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
                
                # Adaptive delay based on buffer size
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
            self.last_refresh_time = time.time()  # Reset refresh timer
        else:
            print("Reconnection failed")
    
    def scale_frame(self, frame):
        """Scale frame according to settings"""
        if frame is None:
            return None
            
        # Store original size for stats
        if self.original_size is None:
            self.original_size = (frame.shape[1], frame.shape[0])  # (width, height)
        
        # No scaling needed
        if self.scale_factor == 1.0 and self.target_width is None and self.target_height is None:
            return frame
        
        # Calculate target dimensions
        if self.target_width is not None and self.target_height is not None:
            # Explicit dimensions
            new_width, new_height = self.target_width, self.target_height
        elif self.target_width is not None:
            # Width specified, maintain aspect ratio
            aspect_ratio = frame.shape[0] / frame.shape[1]  # height/width
            new_width = self.target_width
            new_height = int(self.target_width * aspect_ratio)
        elif self.target_height is not None:
            # Height specified, maintain aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]  # width/height
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        else:
            # Use scale factor
            new_width = int(frame.shape[1] * self.scale_factor)
            new_height = int(frame.shape[0] * self.scale_factor)
        
        # Resize the frame
        scaled_frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        return scaled_frame
    
    def get_frame(self):
        """Get a frame from buffer with smart buffering"""
        with self.buffer_lock:
            buffer_size = len(self.frame_buffer)
            
            # Check if we should be buffering
            if self.buffering and buffer_size < self.min_buffer_size:
                return None, True  # Still buffering
            elif self.buffering and buffer_size >= self.min_buffer_size:
                self.buffering = False
                print(f"Buffering complete, starting playback with {buffer_size} frames")
            
            # If buffer gets too low during playback, start buffering again
            if not self.buffering and buffer_size < 3:
                self.buffering = True
                print("Buffer underrun, rebuffering...")
                return None, True
            
            if buffer_size > 0:
                frame_data = self.frame_buffer.popleft()
                return frame_data[0], False  # Return frame, not buffering
            
            return None, False

    def start(self):
        """Start the streaming"""
        if not self.connect():
            return
            
        self.running = True
        
        # Start background thread for reading frames
        self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.read_thread.start()
        
        print("Press 'q' to quit, 's' to show/hide stats, 'r' to refresh stream")
        print(f"Auto-refresh enabled every {self.refresh_interval/60:.1f} minutes")
        print("Buffering initial frames...")
        
        # Main display loop with adaptive frame rate
        last_display_time = time.time()
        frame_interval = 1.0 / 25.0  # Start with 25 FPS
        
        # Wait for the first frame to be ready and display it to ensure the window is created
        while self.running:
            frame, is_buffering = self.get_frame()
            if frame is not None:
                # Scale the frame and show it once to create the window
                frame = self.scale_frame(frame)
                cv.imshow('HLS Stream', frame)
                cv.waitKey(1) # a small delay to allow the window to render
                break
            else:
                print("Waiting for first frame...")
                time.sleep(0.5)

        if not self.running: # Check if the program was stopped while waiting for the first frame
            return

        while self.running:
            current_time = time.time()
            
            # Check if it's time for periodic refresh
            if current_time - self.last_refresh_time > self.refresh_interval:
                print(f"Periodic refresh after {self.refresh_interval/60:.1f} minutes to stay current")
                self.reconnect("Periodic refresh")
            
            # Get frame
            frame, is_buffering = self.get_frame()
            
            if frame is not None:
                # Scale the frame
                frame = self.scale_frame(frame)
                
                self.frames_displayed += 1
                
                # Calculate and display stats
                if current_time - self.last_stats_time >= 1.0:
                    elapsed = current_time - self.start_time
                    display_fps = self.frames_displayed / elapsed if elapsed > 0 else 0
                    read_fps = self.frames_read / elapsed if elapsed > 0 else 0
                    
                    buffer_size = len(self.frame_buffer)
                    
                    # Adjust frame rate based on buffer health
                    if buffer_size > self.target_buffer_size:
                        frame_interval = 1.0 / 30.0  # Speed up if buffer is full
                    elif buffer_size < 10:
                        frame_interval = 1.0 / 20.0  # Slow down if buffer is low
                    else:
                        frame_interval = 1.0 / 25.0  # Normal rate
                    
                    self.last_stats_time = current_time
                
                # Add stats overlay
                if self.show_stats:
                    elapsed = current_time - self.start_time
                    display_fps = self.frames_displayed / elapsed if elapsed > 0 else 0
                    buffer_size = len(self.frame_buffer)
                    
                    # Use a fixed font scale and position based on the scaled frame's dimensions
                    text_scale = 1.0  # Keep a consistent scale for readability
                    text_thickness = 2
                    
                    # Position the text from the bottom of the frame
                    bottom_y = frame.shape[0] - 10  # Start 10 pixels from the bottom
                    
                    # Draw text, moving up for each line
                    cv.putText(frame, f"FPS: {display_fps:.1f}", (10, bottom_y),
                                cv.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness)
                    
                    # Get text sizes to position them correctly
                    (fps_w, fps_h), _ = cv.getTextSize(f"FPS: {display_fps:.1f}", cv.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                    (buffer_w, buffer_h), _ = cv.getTextSize(f"Buffer: {buffer_size}", cv.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                    
                    cv.putText(frame, f"Buffer: {buffer_size}", (10, bottom_y - fps_h - 10),
                                cv.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness)
                    
                    if self.original_size:
                        size_text = f"Size: {self.original_size[0]}x{self.original_size[1]} -> {frame.shape[1]}x{frame.shape[0]}"
                        (size_w, size_h), _ = cv.getTextSize(size_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv.putText(frame, size_text, (10, bottom_y - fps_h - buffer_h - 20),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if is_buffering:
                        buffering_text = "BUFFERING..."
                        (buff_w, buff_h), _ = cv.getTextSize(buffering_text, cv.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
                        # Position it at the top of the group of text, with a small offset
                        cv.putText(frame, buffering_text, (10, bottom_y - fps_h - buffer_h - size_h - 30),
                                    cv.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness)
                
                # Display frame
                cv.imshow('HLS Stream', frame)
                
                # Frame rate control
                elapsed_since_display = current_time - last_display_time
                if elapsed_since_display < frame_interval:
                    time.sleep(frame_interval - elapsed_since_display)
                last_display_time = time.time()
            
            elif is_buffering:
                # Show buffering message less frequently
                if int(current_time) % 2 == 0:  # Every 2 seconds
                    buffer_size = len(self.frame_buffer)
                time.sleep(0.1)
            
            else:
                # No frame available and not buffering
                time.sleep(0.01)
            
            # Handle key presses AND window close event
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q') or cv.getWindowProperty('HLS Stream', cv.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print(f"Stats display: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('r'):
                print("Manual refresh requested")
                self.reconnect("Manual refresh")
        
        self.stop()
    
    def stop(self):
        """Stop streaming and cleanup"""
        print("Stopping stream...")
        self.running = False
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)
        
        if self.cap:
            self.cap.release()
        
        cv.destroyAllWindows()
        print("Stream closed.")

# Usage examples
if __name__ == "__main__":
    
    # Scale by factor with 20-minute refresh (0.5 = half size)
    streamer = HLSStreamer(stream_url, scale_factor=0.5, refresh_interval=1200)
    
    try:
        streamer.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        streamer.stop()