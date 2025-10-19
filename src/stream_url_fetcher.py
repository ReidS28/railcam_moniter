import sys
import re
import time
import logging
import warnings
import os

# SUPPRESS ALL LOGS BEFORE ANY IMPORTS
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('selenium').setLevel(logging.CRITICAL)
logging.getLogger('webdriver_manager').setLevel(logging.CRITICAL)
logging.getLogger('WDM').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Set environment variables to suppress WebDriver Manager
os.environ['WDM_LOG_LEVEL'] = '0'
os.environ['WDM_PRINT_FIRST_LINE'] = 'False' 
os.environ['WDM_LOG'] = 'False'

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def get_m3u8(url="https://railstream.net/camera-lineup/delaware-railcam"):
    # Setup Selenium with maximum silence
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--silent")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Redirect stderr to suppress remaining logs
    import subprocess
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        
        # Click play button if present
        try:
            play_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".brid-overlay-play-button"))
            )
            play_button.click()
        except:
            pass  # Silently continue if no play button
        
        time.sleep(5)
        
        # Parse HTML
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Search for m3u8
        m3u8_url = None
        for script in soup.find_all("script"):
            if script.string and ".m3u8" in script.string:
                match = re.search(r'(https?://[^\s\'"]+\.m3u8)', script.string)
                if match:
                    m3u8_url = match.group(1)
                    break
        
        driver.quit()
        print("M3U8 URL: " + m3u8_url)
        return m3u8_url
        
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr

if __name__ == "__main__":
    # Use command line argument or default URL
    url = sys.argv[1] if len(sys.argv) > 1 else None
    url = "https://railstream.net/live-cameras/item/atlanta-guest"
    result = get_m3u8(url) if url else get_m3u8()
    if result:
        print(f"M3U8 Stream URL: {result}")
    else:
        print("M3U8 URL not found.")

#STREAM_URL = get_m3u8("https://railstream.net/live-cameras/item/chesterton-in-east-guest")
#STREAM_URL = "https://edge01.railstream.net/Live_Web_Free/CPT_CAM01/playlist_dvr_timeshift-0-1800.m3u8"