import asyncio
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2 as cv
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import os

from stream_url_fetcher import get_m3u8
from utils import load_config, print_time
from handle_train import handle_train_detected


# Load config and helper
CONFIG = load_config('config.json')


def _cfg(path, default=None):
    parts = path.split('.')
    cur = CONFIG
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


# Config values (defaults chosen to be safe)
STREAM_NAME = _cfg('general_settings.stream_settings.stream_name')
STREAMS = _cfg('general_settings.stream_settings.streams', [])
STREAM_CONFIG = next((s for s in STREAMS if s.get('name') == STREAM_NAME), STREAMS[0] if STREAMS else {})
STREAM_URL = STREAM_CONFIG.get('url')
if not STREAM_URL:
    raise RuntimeError("No stream URL configured. Please set general_settings.stream_settings in config.json")

SCALE_FACTOR = _cfg('advanced_settings.display_and_performance_settings.scale_factor', 1.0)
TARGET_WIDTH = _cfg('advanced_settings.display_and_performance_settings.target_width')
TARGET_HEIGHT = _cfg('advanced_settings.display_and_performance_settings.target_height')
SHOW_STATS = _cfg('advanced_settings.display_and_performance_settings.show_stats', True)

REFRESH_INTERVAL_SECONDS = _cfg('advanced_settings.buffering_and_refresh.refresh_interval_seconds', 60 * 5)
MIN_BUFFER_SIZE = _cfg('advanced_settings.buffering_and_refresh.min_buffer_size', 5)
BUFFER_MAXLEN = _cfg('advanced_settings.buffering_and_refresh.buffer_maxlen', 120)
DISPLAY_FPS = _cfg('advanced_settings.buffering_and_refresh.display_fps', 30)

DETECTION_INTERVAL = _cfg('advanced_settings.detection_settings.detection_interval', 1.0)
DETECTION_ENABLED = _cfg('advanced_settings.detection_settings.detection_enabled', True)
DETECTION_MODEL_PATH = _cfg('advanced_settings.detection_settings.detection_model_path')
DETECTION_CLASS_ID = _cfg('advanced_settings.detection_settings.detection_class_id', 0)
START_DETECTION_CONFIDENCE_THRESHOLD = _cfg('advanced_settings.detection_settings.start_detection_confidence_threshold', 0.6)
CONTINUOUS_DETECTION_CONFIDENCE_THRESHOLD = _cfg('advanced_settings.detection_settings.continuous_detection_confidence_threshold', 0.5)
DETECTION_MIN_FRAMES = _cfg('advanced_settings.detection_settings.detection_min_frames', 3)
DETECTION_TIMEOUT = _cfg('advanced_settings.detection_settings.detection_timeout', 5)
# Quick filters to reduce false positives
DETECTION_MIN_BBOX_AREA = _cfg('advanced_settings.detection_settings.min_bbox_area', 5000)
DETECTION_ASPECT_RATIO_RANGE = _cfg('advanced_settings.detection_settings.aspect_ratio_range', (0.5, 4.0))
DETECTION_IOU_MATCH_THRESHOLD = _cfg('advanced_settings.detection_settings.iou_match_threshold', 0.4)

WAITKEY_DELAY_MS = _cfg('advanced_settings.ui_and_control.waitkey_delay_ms', 1)
FONT_HERSHEY = cv.FONT_HERSHEY_SIMPLEX
STATS_FONT_SCALE = _cfg('advanced_settings.ui_and_control.stats_font_scale', 0.6)
STATS_FONT_THICKNESS = _cfg('advanced_settings.ui_and_control.stats_font_thickness', 1)
STATS_COLOR_FPS = tuple(_cfg('advanced_settings.ui_and_control.stats_color_fps', (0, 255, 0)))
STATS_COLOR_BUFFER = tuple(_cfg('advanced_settings.ui_and_control.stats_color_buffer', (255, 255, 255)))
STATS_COLOR_SIZE = tuple(_cfg('advanced_settings.ui_and_control.stats_color_size', (200, 200, 200)))
STATS_COLOR_DETECTION = tuple(_cfg('advanced_settings.ui_and_control.stats_color_detection', (0, 255, 255)))
STATS_COLOR_BUFFERING = tuple(_cfg('advanced_settings.ui_and_control.stats_color_buffering', (0, 0, 255)))
BOUNDING_BOX_WEIGHT = _cfg('advanced_settings.ui_and_control.bounding_box_weight', 2)

MAX_PLAYBACK_FPS = _cfg('advanced_settings.playback_settings.max_playback_fps', 30.0)
MIN_PLAYBACK_FPS = _cfg('advanced_settings.playback_settings.min_playback_fps', 1.0)
BUFFER_CRITICAL = _cfg('advanced_settings.playback_settings.buffer_critical', 5)
BUFFER_OPTIMAL = _cfg('advanced_settings.playback_settings.buffer_optimal', 30)


class HLSStreamer:
    """Stream reader with async detection and recording."""

    def __init__(self, url):
        self.url = url
        self.frame_buffer = deque(maxlen=BUFFER_MAXLEN)
        self.buffer_lock = threading.Lock()

        self.scale_factor = SCALE_FACTOR
        self.target_width = TARGET_WIDTH
        self.target_height = TARGET_HEIGHT
        self.original_size = None

        self.running = False
        self.buffering = True
        self.show_stats = SHOW_STATS

        self.frames_read = 0
        self.frames_displayed = 0
        self.incoming_frame_timestamps = deque()
        self.start_time = time.time()

        self.cap = None

        self.refresh_interval = REFRESH_INTERVAL_SECONDS
        self.last_refresh_time = time.time()

        self.detection_interval = DETECTION_INTERVAL
        self.last_detection_time = 0.0
        self.detection_enabled = DETECTION_ENABLED
        self.detection_queue = asyncio.Queue(maxsize=10)
        self.detection_loop = None
        self.detection_task = None
        # Use a dedicated ThreadPoolExecutor for model inference so we can control it and shutdown cleanly
        self._inference_executor = ThreadPoolExecutor(max_workers=1)
        # Validate model path; if missing disable detection with a friendly warning
        self.model = None
        if DETECTION_MODEL_PATH:
            if os.path.exists(DETECTION_MODEL_PATH):
                try:
                    self.model = YOLO(DETECTION_MODEL_PATH)
                except Exception as e:
                    print_time(f"Failed to load detection model: {e}. Detection disabled.")
                    self.model = None
            else:
                print_time(f"Detection model path not found: {DETECTION_MODEL_PATH}. Detection disabled.")

        self.train_detected = False
        self.train_detected_frames = 0
        self.last_train_detected_time = 0.0
        self.train_detected_confidence = 0.0
        self.box_color = None
        self.box_coordinates = None

        self.read_thread = None
        self.detection_thread = None
        self._recording_in_progress = False

    # Detection
    async def _analyze_frame_async(self, frame):
        if frame is None or self.model is None:
            return []

        def _infer():
            try:
                results = self.model(frame, verbose=False)
                detections = []
                for r in results:
                    boxes = getattr(r, 'boxes', None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        class_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if class_id == DETECTION_CLASS_ID:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'confidence': conf})
                return detections
            except Exception as e:
                print_time(f"Detection exception: {e}")
                return []

        loop = asyncio.get_running_loop()
        # run inference using our dedicated executor
        detections = await loop.run_in_executor(self._inference_executor, _infer)

        # Apply quick filtering (area/aspect) and temporal smoothing via IoU matching
        def _passes_filters(box, conf):
            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h
            if area < DETECTION_MIN_BBOX_AREA:
                return False
            ar = w / float(h)
            if not (DETECTION_ASPECT_RATIO_RANGE[0] <= ar <= DETECTION_ASPECT_RATIO_RANGE[1]):
                return False
            if conf < CONTINUOUS_DETECTION_CONFIDENCE_THRESHOLD:
                return False
            return True

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        # Filter detections
        filtered = [d for d in detections if _passes_filters(d['box'], d['confidence'])]
        if filtered:
            best = max(filtered, key=lambda d: d['confidence'])
            self.box_coordinates = best['box']
            self.train_detected_confidence = best['confidence']

            threshold = START_DETECTION_CONFIDENCE_THRESHOLD if not self.train_detected else CONTINUOUS_DETECTION_CONFIDENCE_THRESHOLD
            if self.train_detected_confidence > threshold:
                # temporal smoothing: require that the new box overlaps (IoU) sufficiently with the previous
                if self.box_coordinates and self.box_coordinates is not None and self.train_detected_frames > 0:
                    prev = self.box_coordinates_prev if hasattr(self, 'box_coordinates_prev') else None
                    if prev is not None:
                        overlap = _iou(prev, self.box_coordinates)
                        if overlap < DETECTION_IOU_MATCH_THRESHOLD:
                            # do not count as consecutive detection
                            self.train_detected_frames = 1
                        else:
                            self.train_detected_frames += 1
                    else:
                        self.train_detected_frames += 1
                else:
                    self.train_detected_frames += 1
                self.last_train_detected_time = time.time()
                self.box_color = (255, 255, 0) if not self.train_detected else (0, 255, 0)

                if not self.train_detected and self.train_detected_frames >= DETECTION_MIN_FRAMES:
                    print_time("Train detected")
                    self.train_detected = True
                    if not self._recording_in_progress:
                        self._recording_in_progress = True
                        threading.Thread(target=self._record_thread_wrapper, args=(150,), daemon=True).start()
            # remember previous box for next frame's IoU check
            self.box_coordinates_prev = self.box_coordinates
        else:
            # No detections (or filtered out) -> reset detection counters/state
            self.train_detected_frames = 0
            self.train_detected_confidence = 0.0
            self.box_coordinates = None
            self.box_color = (160, 160, 160)
            if self.train_detected and (time.time() - self.last_train_detected_time) > DETECTION_TIMEOUT:
                self.box_color = None
                self.train_detected = False

        return detections

    async def detection_worker(self):
        print_time("Starting async detection worker...")
        while self.running:
            try:
                frame = await asyncio.wait_for(self.detection_queue.get(), timeout=1.0)
                await self._analyze_frame_async(frame)
                self.detection_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print_time(f"Detection worker error: {e}")
                await asyncio.sleep(0.1)
        print_time("Async detection worker stopped")

    def start_async_detection(self):
        def _loop():
            self.detection_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.detection_loop)
            self.detection_task = self.detection_loop.create_task(self.detection_worker())
            try:
                self.detection_loop.run_until_complete(self.detection_task)
            except Exception as e:
                print_time(f"Detection loop error: {e}")
            finally:
                try:
                    self.detection_loop.close()
                except Exception:
                    pass

        self.detection_thread = threading.Thread(target=_loop, daemon=True)
        self.detection_thread.start()

    def add_frame_for_detection(self, frame):
        if self.detection_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.detection_queue.put(frame.copy()), self.detection_loop)
        except Exception:
            pass

    # Recording
    def _record_thread_wrapper(self, num_frames):
        try:
            output = self.record_stream(num_frames)
            if output:
                print_time(f"Recording saved to: {output}")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(handle_train_detected(str(output)))
                    print_time("Slack notification sent")
                except Exception as e:
                    print_time(f"Notification error: {e}")
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
            else:
                print_time("Recording failed or produced no file")
        except Exception as e:
            print_time(f"Recording thread exception: {e}")
        finally:
            self._recording_in_progress = False

    def record_stream(self, num_frames=150):
        try:
            detections_dir = Path('detections')
            detections_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = detections_dir / f"train_detection_{timestamp}.mp4"

            print_time(f"Starting recording -> {output_path}")

            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            fps = 30.0
            writer = None

            frames_written = 0
            start = time.time()
            timeout = 30

            while frames_written < num_frames and self.running:
                if time.time() - start > timeout:
                    print_time(f"Recording timed out after {timeout}s, wrote {frames_written} frames")
                    break

                with self.buffer_lock:
                    frame = (self.frame_buffer[-1][0].copy() if self.frame_buffer else None)

                if frame is None:
                    time.sleep(0.01)
                    continue

                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv.VideoWriter(str(output_path), fourcc, fps, (w, h))
                    if not writer.isOpened():
                        print_time(f"Unable to open writer for {output_path}")
                        return None

                writer.write(frame)
                frames_written += 1
                time.sleep(1.0 / fps)

            if writer is not None:
                writer.release()
                print_time(f"Recording complete: {frames_written} frames -> {output_path}")
                return output_path
            return None
        except Exception as e:
            print_time(f"Record error: {e}")
            try:
                if writer is not None:
                    writer.release()
            except Exception:
                pass
            return None

    # Capture / Buffer
    def connect(self):
        print_time("Connecting to stream...")
        self.cap = cv.VideoCapture(self.url)
        if not self.cap or not self.cap.isOpened():
            print_time("Cannot open stream")
            return False
        try:
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        print_time("Stream opened")
        return True

    def read_frames(self):
        consecutive_failures = 0
        last_success = time.time()
        while self.running:
            if self.cap is None:
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception as e:
                print_time(f"OpenCV read error: {e}")
                ret, frame = False, None

            now = time.time()
            if ret and frame is not None:
                with self.buffer_lock:
                    self.frame_buffer.append((frame, now))
                    self.incoming_frame_timestamps.append(now)
                    while self.incoming_frame_timestamps and self.incoming_frame_timestamps[0] < now - 10:
                        self.incoming_frame_timestamps.popleft()

                self.frames_read += 1
                consecutive_failures = 0
                last_success = now
                bs = len(self.frame_buffer)
                time.sleep(max(0.001, (bs / float(self.frame_buffer.maxlen)) * 0.05))
            else:
                consecutive_failures += 1
                if consecutive_failures > 15:
                    print_time("Too many read failures, attempting reconnect")
                    self.reconnect("Read failures")
                    consecutive_failures = 0
                elif now - last_success > 5.0:
                    print_time("No frames for 5s, reconnecting")
                    self.reconnect("No frames timeout")
                    last_success = now
                time.sleep(0.1)

    def reconnect(self, reason="manual"):
        if getattr(self, '_reconnecting', False):
            print_time(f"Reconnection already in progress ({reason})")
            return False
        self._reconnecting = True
        try:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            print_time(f"Reconnecting: {reason}")
            time.sleep(2)
            for attempt in range(3):
                print_time(f"Reconnect attempt {attempt+1}/3")
                self.cap = cv.VideoCapture(self.url)
                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    with self.buffer_lock:
                        self.frame_buffer.clear()
                    self.buffering = True
                    self._reconnecting = False
                    print_time("Reconnected")
                    return True
                time.sleep(3)
            print_time("All reconnection attempts failed")
            return False
        except Exception as e:
            print_time(f"Reconnect exception: {e}")
            return False
        finally:
            self._reconnecting = False

    def scale_frame(self, frame):
        if frame is None:
            return None
        if self.original_size is None:
            self.original_size = (frame.shape[1], frame.shape[0])
        if self.target_width and self.target_height:
            new_w, new_h = self.target_width, self.target_height
        elif self.target_width:
            ar = frame.shape[0] / frame.shape[1]
            new_w = self.target_width
            new_h = int(self.target_width * ar)
        elif self.target_height:
            ar = frame.shape[1] / frame.shape[0]
            new_h = self.target_height
            new_w = int(self.target_height * ar)
        else:
            new_w = int(frame.shape[1] * self.scale_factor)
            new_h = int(frame.shape[0] * self.scale_factor)
        try:
            return cv.resize(frame, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        except Exception:
            return frame

    def get_frame(self):
        with self.buffer_lock:
            bs = len(self.frame_buffer)
            if self.buffering and bs < MIN_BUFFER_SIZE:
                return None, True
            if self.buffering and bs >= MIN_BUFFER_SIZE:
                self.buffering = False
                print_time(f"Buffering complete ({bs} frames)")
            if not self.buffering and bs < 3:
                self.buffering = True
                print_time("Buffer underrun, rebuffering...")
                return None, True
            if bs > 0:
                frame, ts = self.frame_buffer.popleft()
                return frame, False
            return None, False

    def update_bounding_box(self, image):
        if self.box_color is None or self.box_coordinates is None:
            return
        x1, y1, x2, y2 = self.box_coordinates
        cv.rectangle(image, (x1, y1), (x2, y2), self.box_color, BOUNDING_BOX_WEIGHT)
        label = f"Train: {self.train_detected_confidence:.2f}"
        (tw, th), baseline = cv.getTextSize(label, FONT_HERSHEY, 0.5, 1)
        cv.rectangle(image, (x1, y1 - th - baseline), (x1 + tw, y1), self.box_color, -1)
        cv.putText(image, label, (x1, y1 - baseline), FONT_HERSHEY, 0.5, (0, 0, 0), 1)

    def start(self):
        if not self.connect():
            return
        self.running = True
        self.read_thread = threading.Thread(target=self.read_frames, daemon=True)
        self.read_thread.start()
        self.start_async_detection()

        print_time("Press 'q' to quit, 's' toggle stats, 'r' refresh, 'd' toggle detection")

        while self.running:
            frame, is_buffering = self.get_frame()
            if frame is not None:
                cv.imshow('HLS Stream', frame)
                cv.waitKey(1)
                break
            print_time("Waiting for first frame...")
            time.sleep(0.5)

        last_display = time.time()

        while self.running:
            now = time.time()
            bs = len(self.frame_buffer)
            if bs <= BUFFER_CRITICAL:
                target_fps = MAX_PLAYBACK_FPS
            elif bs >= BUFFER_OPTIMAL:
                target_fps = MAX_PLAYBACK_FPS
            else:
                ratio = (bs - BUFFER_CRITICAL) / float(max(1, BUFFER_OPTIMAL - BUFFER_CRITICAL))
                target_fps = MIN_PLAYBACK_FPS + ratio * (MAX_PLAYBACK_FPS - MIN_PLAYBACK_FPS)

            frame_interval = 1.0 / max(1.0, target_fps)
            elapsed = now - last_display
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            frame, is_buffering = self.get_frame()
            if frame is not None:
                frame = self.scale_frame(frame)
                h, w = frame.shape[:2]
                cv.rectangle(frame, (w - 100, h - 60), (w, h), (64, 64, 64), -1)

                if self.detection_enabled and (now - self.last_detection_time) >= self.detection_interval:
                    self.add_frame_for_detection(frame)
                    self.last_detection_time = now

                self.frames_displayed += 1
                self.update_bounding_box(frame)

                if self.show_stats:
                    elapsed_total = now - self.start_time
                    display_fps = (self.frames_displayed / elapsed_total) if elapsed_total > 0 else 0.0
                    bottom_y = frame.shape[0] - 10
                    cv.putText(frame, f"FPS: {display_fps:.1f}", (10, bottom_y), FONT_HERSHEY, STATS_FONT_SCALE, STATS_COLOR_FPS, STATS_FONT_THICKNESS)
                    cv.putText(frame, f"Buffer: {bs}", (10, bottom_y - 20), FONT_HERSHEY, STATS_FONT_SCALE, STATS_COLOR_BUFFER, STATS_FONT_THICKNESS)
                    if self.original_size:
                        size_text = f"Size: {self.original_size[0]}x{self.original_size[1]} -> {frame.shape[1]}x{frame.shape[0]}"
                        cv.putText(frame, size_text, (10, bottom_y - 40), FONT_HERSHEY, 0.5, STATS_COLOR_SIZE, 1)
                    det_text = f"Detection: {'ON' if self.detection_enabled else 'OFF'} ({self.detection_interval}s, ASYNC)"
                    cv.putText(frame, det_text, (10, bottom_y - 60), FONT_HERSHEY, 0.5, STATS_COLOR_DETECTION, 1)
                    if is_buffering:
                        cv.putText(frame, "BUFFERING...", (10, bottom_y - 80), FONT_HERSHEY, STATS_FONT_SCALE, STATS_COLOR_BUFFERING, STATS_FONT_THICKNESS)

                cv.imshow('HLS Stream', frame)
                last_display = time.time()
            elif is_buffering:
                time.sleep(0.1)
            else:
                time.sleep(0.01)

            key = cv.waitKey(WAITKEY_DELAY_MS) & 0xFF
            if key == ord('q') or cv.getWindowProperty('HLS Stream', cv.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('s'):
                self.show_stats = not self.show_stats
                print_time(f"Stats: {'ON' if self.show_stats else 'OFF'}")
            elif key == ord('r'):
                print_time('Manual refresh')
                self.reconnect('Manual refresh')
            elif key == ord('d'):
                self.detection_enabled = not self.detection_enabled
                print_time(f"Detection: {'ON' if self.detection_enabled else 'OFF'}")

        self.stop()

    def stop(self):
        print_time('Stopping stream...')
        self.running = False
        if self.detection_task and self.detection_loop:
            try:
                asyncio.run_coroutine_threadsafe(self._cancel_async_detection(), self.detection_loop).result(timeout=2.0)
            except Exception:
                pass

        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=3)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        # Shutdown the inference executor
        try:
            self._inference_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            cv.destroyAllWindows()
        except Exception:
            pass
        print_time('Stream closed.')

    async def _cancel_async_detection(self):
        if self.detection_task:
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass

if __name__ == '__main__':
    streamer = HLSStreamer(STREAM_URL)
    try:
        streamer.start()
    except KeyboardInterrupt:
        print_time('\nInterrupted by user')
        streamer.stop()