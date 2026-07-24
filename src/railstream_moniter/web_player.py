from pathlib import Path

from playwright.sync_api import sync_playwright


class webPlayer:
    def __init__(
        self, url: str = "https://railstream.net/live-cameras/item/cptweed-guest"
    ):
        self.url = url

        project_root = Path(__file__).resolve().parents[2]
        extension_path = project_root / "uBOLite"

        self._playwright = sync_playwright().start()
        context = self._playwright.chromium.launch_persistent_context(
            user_data_dir="",
            headless=False,
            args=[
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
            ],
        )
        self._page = context.new_page()

        self._page.goto(self.url)

        self.click_play()
        self.enter_fullscreen()

    def click_play(self):

        play_button_selector = ".brid-overlay-play-button.brid-button"

        # Canter on page
        locator = self._page.locator(play_button_selector)
        locator.evaluate("el => el.scrollIntoView({block: 'center', inline: 'center'})")

        self._page.click(play_button_selector)

    def enter_fullscreen(self):
        self._page.click(".brid-fullscreen-control.brid-control.brid-button")

    def end(self):
        self._page.close()
