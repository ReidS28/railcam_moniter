from page_loaders.page_loader import PageLoader
from playwright.async_api import BrowserContext


class RailstreamPageLoader(PageLoader):
    def __init__(self, context: BrowserContext, url) -> None:
        super().__init__(context, url)

    async def _navigate_page(self) -> None:
        await self.click_play()
        await self.enter_fullscreen()

    async def click_play(self):
        if self.page is None:
            return

        play_button_selector = ".brid-overlay-play-button.brid-button"
        locator = self.page.locator(play_button_selector)
        await locator.evaluate(
            "el => el.scrollIntoView({block: 'center', inline: 'center'})"
        )
        await self.page.click(play_button_selector)

    async def enter_fullscreen(self):
        if self.page is None:
            return

        await self.page.click(".brid-fullscreen-control.brid-control.brid-button")
