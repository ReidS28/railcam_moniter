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
from page_loaders.railstream_page_loader import RailstreamPageLoader
from page_loaders.youtube_page_loader import YoutubePageLoader
from playwright.async_api import async_playwright
from utils import print_time


class webPlayer:
    def __init__(
        self,
        url: str = "https://railstream.net/live-cameras/item/cptweed-guest",
        frame_callback: Callable[[MatLike]] = (lambda _: None),
    ):
        url = "https://www.youtube.com/watch?v=ssuM6NJQ2no"
        self.url = url
        self.frame_callback = frame_callback

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

        # self._page_loader = RailstreamPageLoader(self._context, self.url, True)
        self._page_loader = YoutubePageLoader(self._context, self.url, True)
        await self._page_loader.create()

        time.sleep(2)  # noqa ASYNC251

        await self.start_screencast()

    async def _launch_browser(self):
        project_root = Path(__file__).resolve().parents[2]
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

    async def start_screencast(self, quality: int = 80, every_nth_frame: int = 1):
        if self._page_loader is None or self._page_loader.page is None:
            return

        self._cdp = await self._context.new_cdp_session(self._page_loader.page)
        self._cdp.on("Page.screencastFrame", self._on_screencast_frame)

        await self._cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": quality, "everyNthFrame": every_nth_frame},
        )
        print_time("Screencast started", "green")

    def _on_screencast_frame(self, params):
        try:
            img_bytes = base64.b64decode(params["data"])
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is not None:
                self.frame_callback(frame)
                with self._frame_lock:
                    self._latest_frame = frame

        except Exception as e:  # noqa BLE001
            print_time(f"failed to decode screencast frame: {e}", "red")

        finally:
            asyncio.run_coroutine_threadsafe(
                self._ack_frame(params["sessionId"]), self._loop
            )

    async def _ack_frame(self, session_id):
        try:
            await self._cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception as e:  # noqa BLE001
            print_time(f"failed to ack screencast frame: {e}", "yellow")

    def get_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def edit_bounding_box(self, *args):
        if self._page_loader is None:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._page_loader.edit_bounding_box_async(*args), self._loop
        )
        fut.add_done_callback(
            lambda f: (
                f.exception()
                and print_time(f"bounding box edit failed: {f.exception()}", "red")
            )
        )

    def end(self):
        async def _shutdown():
            try:
                await self._cdp.send("Page.stopScreencast")
            except Exception:  # noqa
                pass
            if self._page_loader is not None and self._page_loader.page is not None:
                await self._page_loader.page.close()
            await self._context.close()
            await self._playwright.stop()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        fut.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
