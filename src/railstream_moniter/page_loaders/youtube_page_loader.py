from page_loaders.page_loader import PageLoader
from playwright.async_api import BrowserContext


class YoutubePageLoader(PageLoader):
    def __init__(
        self, context: BrowserContext, url, print_console_log: bool = False, 
    ) -> None:
        super().__init__(context, url, print_console_log)
