from stream_handler import stream_reader
from ui import start_gui
import threading


def main():
    # Start stream reader thread (feeds raw_frame_queue)
    threading.Thread(target=stream_reader, daemon=True).start()

    # Start DearPyGui GUI which now shows the live preview internally
    start_gui()


if __name__ == "__main__":
    main()
