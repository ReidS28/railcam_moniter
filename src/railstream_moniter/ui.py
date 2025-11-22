import dearpygui.dearpygui as dpg
import numpy as np
import cv2

class StreamUI:
    def __init__(self, config):
        self.title = config.get('UI', 'title')
        self.width = config.getint('UI', 'width')
        self.height = config.getint('UI', 'height')
        
        self.texture_id = "stream_texture"
        self.image_id = "stream_image"
        self.registry_id = "stream_texture_registry"

        self._current_frame_dims = (0, 0)

    def _convert_frame_to_texture_data(self, frame):
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return (frame_rgba.astype(np.float32) / 255.0)

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

        with dpg.window(tag="Primary Window"):
            dpg.add_image(
                self.texture_id,
                tag=self.image_id,
                width=self.width,
                height=self.height
            )

        dpg.create_viewport(title=self.title, width=self.width, height=self.height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

    def update_frame(self, frame):
        h, w, _ = frame.shape
        frame_dims = (w, h)

        texture_data = self._convert_frame_to_texture_data(frame)

        if frame_dims != self._current_frame_dims:
            print(f"Resizing texture from {self._current_frame_dims} to {frame_dims}")
            
            # Generate a new, unique ID for the texture
            new_texture_id = dpg.generate_uuid()

            dpg.delete_item(self.texture_id)
            
            dpg.add_dynamic_texture(
                width=w,
                height=h,
                default_value=texture_data,
                tag=new_texture_id, # Use the new unique ID
                parent=self.registry_id
            )
            
            # Point the image widget to the new texture
            dpg.configure_item(self.image_id, texture_tag=new_texture_id, width=w, height=h)
            
            # Update self.texture_id to the new ID for the next resize
            self.texture_id = new_texture_id
            
            self._current_frame_dims = frame_dims
        else:
            dpg.set_value(self.texture_id, texture_data)

    def is_running(self):
        return dpg.is_dearpygui_running()

    def render(self):
        dpg.render_dearpygui_frame()

    def cleanup(self):
        print("Cleaning up UI...")
        dpg.destroy_context()