import configparser
import threading
from stream_handler import StreamHandler
from ui import StreamUI
import time

def load_config():
    """Reads the config.ini file."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def main():
    config = load_config()

    data_lock = threading.Lock()

    shared_data = {
        'latest_frame': None
    }

    streamer = StreamHandler(config, shared_data, data_lock)
    streamer.daemon = True
    streamer.start()

    ui = StreamUI(config)
    ui.create_ui()

    print("UI loop starting. Close the window to exit.")

    while ui.is_running():
        current_frame = None

        with data_lock:
            if shared_data['latest_frame'] is not None:
                current_frame = shared_data['latest_frame'].copy()

        if current_frame is not None:
            ui.update_frame(current_frame)

        ui.render()

        if current_frame is None:
            time.sleep(0.01) 

    # --- Cleanup ---
    print("Window closed, stopping stream thread...")
    streamer.stop()
    streamer.join()
    ui.cleanup()
    
    print("Program exited gracefully.")

if __name__ == "__main__":
    main()