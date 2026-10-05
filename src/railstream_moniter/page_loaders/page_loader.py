from playwright.async_api import BrowserContext, Page
from utils import print_time


class PageLoader:
    def __init__(
        self, context: BrowserContext, url, print_console_log: bool = False
    ) -> None:
        self._context: BrowserContext = context
        self.url: str = url
        self.print_console_log = print_console_log
        self.page: Page | None = None

    async def create(self):

        await self._load_page()
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

    # async def add_bounding_helper_functions(self):
    #     bypass_trusted_types = """
    #     if (window.trustedTypes && !window.trustedTypes.defaultPolicy) {
    #         window.trustedTypes.createPolicy('default', {
    #             createHTML: (string) => string,
    #             createScriptURL: (string) => string,
    #             createScript: (string) => string
    #         });
    #     }
    #     """
    #     await self._context.evaluate(bypass_trusted_types)

    #     current_dir = Path(__file__).parent
    #     js_path = current_dir / "helper_functions.js"
    #     await self._context.add_script_tag(path=js_path)

    #     await self._context.evaluate("window.createBoundingBoxContiner()")
