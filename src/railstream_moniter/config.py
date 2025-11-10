import json

CONFIG_FILE = "config.json"
# Placeholder defaults in case of loading error
DEFAULT_STREAM_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_MAX_QUEUE_SIZE = 15

# --- Load the full config object once ---
def load_config():
    """Loads and returns the full configuration dictionary."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {CONFIG_FILE} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"ERROR: {CONFIG_FILE} is not valid JSON.")
        return {}

# The main configuration object available globally to other modules
CONFIG = load_config()

# --- Extract useful, high-level variables for cleaner access ---

try:
    # Use the first stream defined in the 'streams' array
    STREAM_URL = CONFIG["general_settings"]["stream_settings"]["streams"][0]["url"]
    MODEL_PATH = CONFIG["advanced_settings"]["detection_settings"]["detection_model_path"]
    MAX_QUEUE_SIZE = CONFIG["advanced_settings"]["buffering_and_refresh"]["target_buffer_size"]
    
    # You can also define the full settings objects for convenience
    DETECTION_SETTINGS = CONFIG["advanced_settings"]["detection_settings"]
    SLACK_TOKEN = CONFIG["general_settings"]["slack_settings"]["slack_token"]

except (KeyError, IndexError, TypeError) as e:
    print(f"Configuration Error: Could not extract key setting. Using defaults. Error: {e}")
    STREAM_URL = DEFAULT_STREAM_URL
    MODEL_PATH = DEFAULT_MODEL_PATH
    MAX_QUEUE_SIZE = DEFAULT_MAX_QUEUE_SIZE
    DETECTION_SETTINGS = {}
    SLACK_TOKEN = "invalid-token"