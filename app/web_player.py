import asyncio
import base64
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from page_loaders.page_loader import PageLoader
from playwright.async_api import async_playwright
from utils import print_time


class webPlayer:
    def __init__(
        self,
        url: str = "https://railstream.net/live-cameras/item/cptweed-guest",
        frame_callback: Callable[[MatLike]] = (lambda _: None),
        print_console_log: bool = False,
    ):
        # url = "https://www.youtube.com/watch?v=ssuM6NJQ2no"
        self.url = url
        self.frame_callback = frame_callback
        self.print_console_log = print_console_log

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._page_loader: PageLoader | None = None

        # Block until browser is up and screencast is running
        fut = asyncio.run_coroutine_threadsafe(self._setup(), self._loop)
        fut.result()

    async def _setup(self):
        await self._launch_browser()

        self._page_loader = PageLoader.auto_select_loader(
            self._context, self.url, self.print_console_log
        )
        await self._page_loader.create()

        time.sleep(2)  # noqa ASYNC251

        await self.start_video_capture()

    async def _launch_browser(self):
        project_root = Path(__file__).resolve().parents[1]
        extension_path = project_root / "uBOLite"

        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir="",
            headless=False,
            args=[
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=CalculateNativeWinOcclusion,IntersectionObserverV2",
                "--window-position=0,0",
                "--window-size=1280,800",
                "--disable-ipv6",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        await self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def start_video_capture(self, quality: float = 0.8, fps: int = 30):
        self._capture_stop = False
        self._capture_task = asyncio.create_task(self._capture_loop(quality, fps))
        print_time("Video capture started", "green")

    async def _capture_loop(self, quality: float, fps: int):
        if self._page_loader is None or self._page_loader.page is None:
            return

        interval = 1.0 / fps
        while not self._capture_stop:
            t0 = time.perf_counter()
            try:
                page = self._page_loader.page
                b64 = await page.evaluate("q => window.grabVideoFrame(q)", quality)
                if b64:
                    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame_callback(frame)
                        with self._frame_lock:
                            self._latest_frame = frame
            except Exception as e:  # noqa BLE001
                print_time(f"failed to grab video frame: {e}", "red")
                await asyncio.sleep(0.5)

            await asyncio.sleep(max(0.0, interval - (time.perf_counter() - t0)))

    def edit_bounding_box(
        self,
        show: bool | None = False,
        x: float = 0,
        y: float = 0,
        w: float = 0,
        h: float = 0,
        text: str = "",
    ):
        if self._page_loader is None:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._page_loader.edit_bounding_box_async(show, x, y, w, h, text),
            self._loop,
        )
        fut.add_done_callback(
            lambda f: (
                f.exception()
                and print_time(f"bounding box edit failed: {f.exception()}", "red")
            )
        )

    def reset_bounding_box(self):
        if self._page_loader is None:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._page_loader.reset_bounding_box_async(),
            self._loop,
        )
        fut.add_done_callback(
            lambda f: (
                f.exception()
                and print_time(f"bounding box reset failed: {f.exception()}", "red")
            )
        )

    def update_bounding_boxes(self, payload):
        if self._page_loader is None:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._page_loader.update_bounding_boxes_async(payload),
            self._loop,
        )
        fut.add_done_callback(
            lambda f: (
                f.exception()
                and print_time(f"bounding box reset failed: {f.exception()}", "red")
            )
        )

    def end(self):
        async def _shutdown():
            self._capture_stop = True
            task = getattr(self, "_capture_task", None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            if self._page_loader is not None and self._page_loader.page is not None:
                await self._page_loader.page.close()
            await self._context.close()
            await self._playwright.stop()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        fut.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
