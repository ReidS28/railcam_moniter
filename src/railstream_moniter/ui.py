import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from utils import print_time
import time

class StreamUI:
    def __init__(self, config):
        self.title = config.get('UI', 'title')
        self.width = config.getint('UI', 'width')
        self.height = config.getint('UI', 'height')
        
        self.texture_id = "stream_texture"
        self.image_id = "stream_image"
        self.registry_id = "stream_texture_registry"

        self.window_tag = "Video Window"
        self.drawlist_id = "stream_drawlist"
        self.draw_image_id = "stream_draw_image"

        self._current_frame_dims = (0, 0)

    def _convert_frame_to_texture_data(self, frame):
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return (frame_rgba.astype(np.float32) / 255.0)
    
    def _on_window_resize(self, sender, app_data, user_data):

        dpg.configure_item(
            self.drawlist_id, 
            width = -1,
            height = -1
        )
        
        time.sleep(0.001)

        drawlist_size = dpg.get_item_rect_size(self.drawlist_id)
        drawlist_width = drawlist_size[0]
        drawlist_height = drawlist_size[1]

        dpg.configure_item(
            self.drawlist_id,
            width=drawlist_width,
            height=drawlist_height
        )

        dpg.split_frame()

        dpg.configure_item(
            self.draw_image_id, 
            pmax=(drawlist_width, drawlist_height)
        )

        print_time(f"Width: {drawlist_width}, Height: {drawlist_height}")

    def create_ui(self):
        dpg.create_context()
        placeholder = np.zeros((self.height, self.width, 4), dtype=np.float32)

        with dpg.texture_registry(show=False, tag=self.registry_id):
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=placeholder,
                tag=self.texture_id
            )
        initial_window_height = self.height + 40

        with dpg.window( # Inner Window
            tag=self.window_tag, 
            label="Video Stream", 
            width=self.width,
            height=initial_window_height,
            no_close=True,
            no_resize=False,
            no_collapse=False
        ):
            
            with dpg.drawlist(
                tag=self.drawlist_id,
                width=-1,#self.width, 
                height=-1#self.height
            ):
                dpg.draw_rectangle(
                    pmin=(0, 0), 
                    # pmax is set to a large value to ensure it covers the entire area.
                    # This will be clipped by the drawlist boundaries.
                    pmax=(9999, 9999), 
                    color=(0, 0, 0, 0),  # Transparent border
                    fill=(20, 30, 80, 255), # Dark Blue fill (RGBA)
                    tag="drawlist_background_rect" # Give it a tag to configure it later
                )
                
                dpg.draw_image(
                    self.texture_id,
                    pmin=(0, 0),
                    #pmax=(self.width, self.height),
                    pmax=(300, 300),
                    tag=self.draw_image_id
                )

        with dpg.item_handler_registry(tag="window_resize_handler_reg"):
            # Add a resize handler to the registry
            dpg.add_item_resize_handler(callback=self._on_window_resize)
        
        # Bind the registry to your window
        dpg.bind_item_handler_registry(self.window_tag, "window_resize_handler_reg")

        dpg.create_viewport(title=self.title, width=1280, height=720) 
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update_frame(self, frame):
        
        h, w, _ = frame.shape
        if w == 0 or h == 0:
            return
        
        frame_dims = (w, h)
        texture_data = self._convert_frame_to_texture_data(frame)

        new_texture_tag = self.texture_id
        if frame_dims != self._current_frame_dims:
            print_time(f"Resizing texture and draw item for {frame_dims}")

            new_texture_tag = dpg.generate_uuid()
            dpg.delete_item(self.texture_id)

            dpg.add_dynamic_texture(
                width=w,
                height=h,
                default_value=texture_data,
                tag=new_texture_tag,
                parent=self.registry_id
            )
            #scale = min(self.width / w, self.height / h)
            #new_w = int(w * scale)
            #new_h = int(h * scale)
            #x_offset = (self.width - new_w) // 2
            #y_offset = (self.height - new_h) // 2
            #
            #pmin = (x_offset, y_offset)
            #pmax = (x_offset + new_w, y_offset + new_h)

            drawlist_width = dpg.get_item_rect_size(self.drawlist_id)[0]
            drawlist_height = dpg.get_item_rect_size(self.drawlist_id)[1]

            # Fallback just in case (shouldn't be needed, but safe)
            if drawlist_width == 0: drawlist_width = self.width
            if drawlist_height == 0: drawlist_height = self.height

            dpg.configure_item(
                self.draw_image_id, 
                texture_tag=new_texture_tag, 
                pmin=(0, 0), 
                pmax=(drawlist_width, drawlist_height) # Set pmax to the drawlist size
            )

            self.texture_id = new_texture_tag
            self._current_frame_dims = frame_dims
        else:
            dpg.set_value(self.texture_id, texture_data)

    def is_running(self):
        return dpg.is_dearpygui_running()

    def render(self):
        dpg.render_dearpygui_frame()

    def cleanup(self):
        print_time("Cleaning up UI...")
        dpg.destroy_context() 