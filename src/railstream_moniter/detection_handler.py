import datetime
import os
import subprocess
import tempfile
from collections import deque

import cv2
import imageio_ffmpeg
import numpy as np
from cv2.typing import MatLike
from slack_bot import SlackBot
from utils import print_time

DEFAULT_PRE_ROLL_FRAMES = 30
DEFAULT_VIDEO_SECONDS = 10
DEFAULT_FPS = 30
DEFAULT_VIDEO_FRAMES = DEFAULT_VIDEO_SECONDS * DEFAULT_FPS + 1


class DetectionHandler:
    def __init__(
        self, slack_bot: SlackBot, pre_roll_frames: int = DEFAULT_PRE_ROLL_FRAMES
    ) -> None:

        self._rolling_storage: deque[MatLike] = deque(maxlen=pre_roll_frames)

        self.slack_bot = slack_bot

        self._recording: bool = False
        self._clip_frames: list[MatLike] = []
        self._frames_remaining: int = 0
        self._current_file_path: str | None = None
        self._current_message: str | None = None
        self._fps: int = DEFAULT_FPS

    def add_frame(self, frame: MatLike | None) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print_time("Received invalid/empty frame. Skipping...", "yellow")
            return

        self._rolling_storage.append(frame)

        if self._recording:
            self._clip_frames.append(frame)
            self._frames_remaining -= 1

            if self._frames_remaining <= 0:
                self._finish_recording()

    def new_detection(self):
        now = datetime.datetime.now().astimezone()
        file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        file_path = f"detections/{file_name}"

        self._current_message = f"Train Detected @ {now.strftime('%Y/%m/%d %H:%M:%S')}"

        self._start_recording(file_path)

    def _start_recording(
        self,
        file_path: str,
        video_frames: int = DEFAULT_VIDEO_FRAMES,
        fps: int = DEFAULT_FPS,
    ) -> None:
        if self._recording or not self._rolling_storage:
            return

        self._recording = True
        self._fps = fps
        self._current_file_path = file_path

        self._clip_frames = list(self._rolling_storage)
        self._frames_remaining = video_frames - len(self._clip_frames)

        # print_time(f"Starting recording: '{self._current_file_path}'")

        if self._frames_remaining <= 0:
            self._finish_recording()

    def _finish_recording(self) -> None:
        file_path = self._current_file_path
        frames = self._clip_frames

        self._recording = False
        self._clip_frames = []
        self._frames_remaining = 0

        if file_path and frames:
            self._save_clip(file_path, frames)

        if self._current_file_path is not None:
            self.slack_bot.send_message(
                message=self._current_message,
                file_path=self._current_file_path,
            )

        self._current_file_path = None
        self._current_message = None

    def _save_clip(self, file_path: str, frames: list[MatLike]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        sample_frame = frames[0]
        height, width = sample_frame.shape[:2]

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".mp4":
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter.fourcc(*"XVID")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, f"raw{ext or '.mp4'}")

            writer = cv2.VideoWriter(tmp_path, fourcc, self._fps, (width, height))
            try:
                for frame in frames:
                    self._write_frame_safely(writer, frame, width, height)
            finally:
                writer.release()

            if ext == ".mp4":
                self._transcode_for_compatibility(tmp_path, file_path)
            else:
                os.replace(tmp_path, file_path)

        print_time(f"Video saved successfully to '{file_path}'", "green")

    @staticmethod
    def _transcode_for_compatibility(src_path: str, dest_path: str) -> None:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            src_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            dest_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            print_time(
                "Bundled ffmpeg binary not found; falling back to raw OpenCV output."
                "yellow",
            )
            os.replace(src_path, dest_path)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            print_time(f"ffmpeg transcode failed, using raw output: {stderr}", "red")
            os.replace(src_path, dest_path)

    @staticmethod
    def _write_frame_safely(
        writer: cv2.VideoWriter, frame: np.ndarray, width: int, height: int
    ) -> None:
        try:
            h, w = frame.shape[:2]
            if (w, h) != (width, height):
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)

        except cv2.error as e:
            print_time(f"OpenCV failed to write frame: {e}", "red")
        except Exception as e:  # noqa: BLE001
            print_time(f"Unexpected error writing frame: {e}", "red")
