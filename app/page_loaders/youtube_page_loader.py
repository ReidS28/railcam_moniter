import asyncio

from page_loaders.page_loader import PageLoader
from playwright.async_api import BrowserContext
from utils import print_time

PLAYER = "#movie_player"
FULLSCREEN_BTN = "#movie_player .ytp-fullscreen-button"
CHAT_TOGGLE = "ytd-live-chat-frame #show-hide-button button"


class YoutubePageLoader(PageLoader):
    def __init__(
        self,
        context: BrowserContext,
        url,
        print_console_log: bool = False,
    ) -> None:
        super().__init__(context, url, print_console_log)

    async def _navigate_page(self) -> None:
        if self.page is None:
            return

        await self.page.wait_for_selector(PLAYER, state="visible")
        if await self.check_for_live():
            await self.close_chat()
        else:
            await asyncio.sleep(1)
        await self.enter_fullscreen()

    async def check_for_live(self) -> bool:
        if self.page is None:
            return False

        try:
            player_locator = self.page.locator(PLAYER)
            await player_locator.hover()

            await asyncio.sleep(1)
            live_badge = self.page.locator("button.ytp-live-badge")
            return await live_badge.is_visible(timeout=2000)
        except Exception:  # noqa BLE001
            return False

    async def close_chat(self) -> None:
        if self.page is None:
            return

        close_button = self.page.frame_locator("iframe#chatframe").locator(
            "#close-button button"
        )
        await close_button.click()

    async def enter_fullscreen(self) -> None:
        if self.page is None:
            return

        fullscreen_button = self.page.locator("button.ytp-fullscreen-button")
        await fullscreen_button.click()
