import asyncio
import datetime
import os
import subprocess
import tempfile
import threading
from collections import deque
from uuid import UUID

import constants
import cv2
import imageio_ffmpeg
import numpy as np
from cv2.typing import MatLike
from slack_bot import SlackBot
from utils import IndexedFrame, print_time


class DetectionHandler:
    def __init__(
        self,
        rolling_storage: deque[IndexedFrame],
        slack_bot: SlackBot,
    ) -> None:

        self.rolling_storage = rolling_storage
        self.slack_bot = slack_bot

        self._recording: bool = False
        self._clip_frames: list[IndexedFrame] = []
        self._frames_remaining: int = 0
        self._current_file_path: str | None = None
        self._current_message: str | None = None
        self._fps: int = constants.OUTPUT_FPS
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def new_detection(
        self,
        start_id: UUID | None = None,
        num_frames: int = constants.OUTPUT_VIDEO_FRAMES,
    ):
        # Use earliest frame id if start_id is none
        if start_id is None:
            if len(self.rolling_storage) > 0:
                start_id = self.rolling_storage[0].id
            else:
                print_time("No frames to start recording", "red")
                return

        fut = asyncio.run_coroutine_threadsafe(
            self.new_detection_async(start_id, num_frames), self._loop
        )
        fut.result()

    async def new_detection_async(self, start_id: UUID, num_frames: int):
        now = datetime.datetime.now().astimezone()
        # file_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        file_name = "detection.mp4"
        file_path = f"detections/{file_name}"

        message = f"Train Detected @ {now.strftime('%Y/%m/%d %H:%M:%S')}"

        frames = await self._gather_frames(start_id, num_frames)
        if frames is None:
            self.slack_bot.send_message(message)
        else:
            self._save_clip(file_path, frames)
            # self.slack_bot.send_message(
            #     message,
            #     file_path,
            # )

    async def _gather_frames(
        self,
        start_id: UUID,
        num_frames: int,
        poll_interval: float = 0.5,
        timeout: float | None = None,
    ) -> list[MatLike] | None:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        frames: list[MatLike] = []
        cursor = start_id
        include_cursor = True

        while True:  # Grab frames in chunks every poll_interval
            snapshot = list(self.rolling_storage)

            idx = next((i for i, f in enumerate(snapshot) if f.id == cursor), None)
            if idx is None:  # If can't find cursor, give up
                return None
            new_items = snapshot[idx if include_cursor else idx + 1 :]
            include_cursor = False

            for item in new_items:
                frames.append(item.frame)
                cursor = item.id
                if len(frames) == num_frames:
                    return frames

            if deadline is not None and loop.time() >= deadline:
                return None

            await asyncio.sleep(poll_interval)  # wait for new frames

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
