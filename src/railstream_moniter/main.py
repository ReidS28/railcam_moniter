import configparser
import threading
import time
import dearpygui.dearpygui as dpg

# We assume you have imported StreamHandler and StreamUI correctly here
from stream_handler import StreamHandler
from ui import StreamUI 
# Note: Since the full imports aren't visible, I'll use placeholders 
# for brevity, but assume you have them.

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def main():
    config = load_config()

    # --- Shared Data and Threading Setup ---
    data_lock = threading.Lock()
    shared_data = {'latest_frame': None}

    streamer = StreamHandler(config, shared_data, data_lock)
    streamer.daemon = True 
    streamer.start()

    # --- UI Setup ---
    ui = StreamUI(config)
    ui.create_ui()

    print("UI loop starting. Close the window to exit.")

    # --- Playback Timer Setup (Target 30 FPS) ---
    TARGET_FPS = 30
    FRAME_DURATION_SEC = 1 / TARGET_FPS 
    
    last_render_time = time.time()

    # --- Main UI Loop (Consumer) ---
    while ui.is_running():
        
        current_time = time.time()

        if (current_time - last_render_time) >= FRAME_DURATION_SEC:
            
            current_frame = None
            with data_lock:
                if shared_data['latest_frame'] is not None:
                    current_frame = shared_data['latest_frame'].copy()
            
            if current_frame is not None:
                ui.update_frame(current_frame)

            last_render_time = current_time

        ui.render()

        time.sleep(0.001) 

    # --- Cleanup ---
    print("Window closed, stopping stream thread...")
    streamer.stop()
    streamer.join()
    ui.cleanup()
    
    print("Program exited gracefully.")

if __name__ == "__main__":
    main()