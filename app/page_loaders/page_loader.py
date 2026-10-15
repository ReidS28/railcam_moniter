from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page
from utils import print_time


class PageLoader:
    @classmethod
    def auto_select_loader(
        cls, context: BrowserContext, url: str, print_console_log: bool = False
    ) -> PageLoader:
        # Import page loaders here
        from page_loaders.railstream_page_loader import RailstreamPageLoader
        from page_loaders.youtube_page_loader import YoutubePageLoader

        host = (urlparse(url).hostname or "").removeprefix("www.")

        loaders = {
            "youtube.com": YoutubePageLoader,
            "youtu.be": YoutubePageLoader,
            "railstream.net": RailstreamPageLoader,
        }
        loader_cls = loaders.get(host, cls)
        return loader_cls(context, url, print_console_log)

    def __init__(
        self,
        context: BrowserContext,
        url,
        print_console_log: bool = False,
    ) -> None:
        self._context: BrowserContext = context
        self.url: str = url
        self.print_console_log = print_console_log
        self.page: Page | None = None

    async def create(self):
        await self._load_page()
        await self.inject_helper_script()
        if self.print_console_log:
            await self._add_console_logging()
        await self._navigate_page()

    async def _load_page(self):
        self.page = await self._context.new_page()
        await self.page.goto(self.url)
        await self.page.bring_to_front()

    async def _navigate_page(self) -> None:
        pass

    async def _add_console_logging(self):
        if self.page is None:
            return

        self.page.on(
            "console",
            lambda msg: print_time(f"[console] {msg.type}: {msg.text}", "cyan"),
        )
        self.page.on("pageerror", lambda exc: print_time(f"[pageerror] {exc}", "red"))
        self.page.on(
            "requestfailed",
            lambda req: print_time(
                f"[requestfailed] {req.method} {req.url} — {req.failure}", "red"
            ),
        )
        self.page.on(
            "response",
            lambda res: (
                print_time(f"[response] {res.status} {res.url}", "yellow")
                if res.status >= 400
                else None
            ),
        )

    async def inject_helper_script(self):
        if self.page is None:
            return

        current_dir = Path(__file__).parent
        js_path = current_dir / "helper_functions.js"
        await self.page.add_init_script(path=js_path)
        await self.page.reload()

        await self.page.evaluate("window.setUpBoundingBoxes()")

    async def edit_bounding_box_async(
        self,
        show: bool | None = False,
        x: float = 0,
        y: float = 0,
        w: float = 0,
        h: float = 0,
        text: str = "",
    ):
        if self.page is None:
            print_time("None")
            return

        await self.page.evaluate(
            f"window.editBoundingBox(NaN, {str(show).lower()}, {x}, {y}, {w}, {h}, '{text}');"
        )

    async def reset_bounding_box_async(self):
        if self.page is None:
            print_time("None")
            return

        await self.page.evaluate("window.resetBoundingBoxes();")

    async def update_bounding_boxes_async(self, payload):
        if self.page is None:
            return

        await self.page.evaluate(
            "(data) => { window.updateBoundingBoxes(data); }",
            payload,
        )
