import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from utils import print_time


class webPlayer:
    def __init__(
        self, url: str = "https://railstream.net/live-cameras/item/cptweed-guest"
    ):
        self.url = url

        options = Options()

        project_root = Path(__file__).resolve().parents[1]
        extension_path = project_root / "uBOLite"
        # extension_path = "C:/Users/straw/Desktop/Code/railstream_moniter/uBOLite"4

        options.add_argument(f"--load-extension={extension_path}")

        self._driver = webdriver.Chrome(options=options)

        time.sleep(2)

        self._driver.get(self.url)

        time.sleep(2)
        self.click_play()
        # time.sleep(2)
        # self.skip_ad()

    def click_play(self):
        try:
            play_button = WebDriverWait(self._driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".brid-overlay-play-button.brid-button")
                )
            )

            # Center video
            self._driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", play_button
            )
            time.sleep(0.5)
            try:
                play_button.click()
            except Exception:
                self._driver.execute_script("arguments[0].click();", play_button)

        finally:
            pass

    def remove_ad_container(self):
        script = """
            var adContainer = document.querySelector('.brid-advert-container');
            if (adContainer) {
                adContainer.remove();
            }
        """
        self._driver.execute_script(script)

    def skip_ad(self):
        script = """
            var adVideo = document.querySelector('.brid-advert-container video');
            
            if (adVideo && !isNaN(adVideo.duration)) {
                adVideo.currentTime = adVideo.duration; 
            }
        """
        try:
            self._driver.execute_script(script)
            print_time("Ad skipped to end", "green")
        except Exception:
            print_time("Could not fast-forward ad.", "yellow")

    def end(self):
        self._driver.close()
        self._driver.quit()
