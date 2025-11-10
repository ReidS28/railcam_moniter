"""Railstream Monitor UI: Video scales to fill container while preserving aspect ratio."""

import cv2
import time
import threading
import numpy as np
import dearpygui.dearpygui as dpg
from stream_handler import raw_frame_queue

def _frame_updater(texture_tag: str, image_tag: str, base_size=(640, 360)):
    """Thread that updates the video texture and resizes the image widget to fill container."""
    base_w, base_h = base_size
    aspect_ratio = base_w / base_h

    while dpg.is_dearpygui_running():
        try:
            frame = raw_frame_queue.get(timeout=1)
        except Exception:
            continue
        if frame is None:
            continue

        try:
            # Get container size
            container_w = dpg.get_item_width("video_container")
            container_h = dpg.get_item_height("video_container")

            if container_w <= 0 or container_h <= 0:
                container_w, container_h = base_w, base_h

            # Calculate scaled size maintaining aspect ratio
            scale_w = container_w
            scale_h = int(scale_w / aspect_ratio)
            if scale_h > container_h:
                scale_h = container_h
                scale_w = int(scale_h * aspect_ratio)

            # Resize the frame
            resized = cv2.resize(frame, (scale_w, scale_h))
            rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
            arr = rgba.astype(np.float32) / 255.0
            flat = arr.flatten()

            # Update the dynamic texture
            dpg.set_value(texture_tag, flat)
            # Update image widget size
            dpg.configure_item(image_tag, width=scale_w, height=scale_h)

        except Exception:
            pass

        try:
            raw_frame_queue.task_done()
        except Exception:
            pass


def start_gui():
    """Start DearPyGui window with a video that fills container while preserving aspect ratio."""
    dpg.create_context()
    dpg.create_viewport(title="Railstream Monitor", width=920, height=560)

    # Theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 18, 18, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (28, 28, 28, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 120, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 60, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220, 255))
    dpg.bind_theme(global_theme)

    base_w, base_h = 640, 360
    blank = np.zeros((base_h, base_w, 4), dtype=np.float32).flatten()
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(base_w, base_h, blank, tag="video_tex")

    # Main window
    with dpg.window(label="Railstream Monitor", tag="main_window", width=900, height=520):
        # Child container that takes full available space
        with dpg.child_window(width=-1, height=-1, tag="video_container"):
            dpg.add_image("video_tex", tag="video_image", width=base_w, height=base_h)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Start the video update thread
    updater = threading.Thread(target=_frame_updater, args=("video_tex", "video_image", (base_w, base_h)), daemon=True)
    updater.start()

    # Set primary window
    try:
        dpg.set_primary_window("main_window", True)
    except Exception:
        pass

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    start_gui()
