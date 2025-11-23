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

        self.window_tag = "Video Window"
        self.drawlist_id = "stream_drawlist"
        self.draw_image_id = "stream_draw_image"

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
        initial_window_height = self.height + 40

        with dpg.window(
            tag=self.window_tag, 
            label="Video Stream", 
            width=self.width, 
            height=initial_window_height,
            no_close=True,
            no_resize=True,
            no_collapse=False
        ):
            with dpg.drawlist(
                tag=self.drawlist_id,
                width=self.width, 
                height=self.height
            ):
                dpg.draw_image(
                    self.texture_id,
                    pmin=(0, 0),
                    pmax=(self.width, self.height),
                    tag=self.draw_image_id
                )

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
            print(f"Resizing texture and draw item for {frame_dims}")

            new_texture_tag = dpg.generate_uuid()
            dpg.delete_item(self.texture_id)

            dpg.add_dynamic_texture(
                width=w,
                height=h,
                default_value=texture_data,
                tag=new_texture_tag,
                parent=self.registry_id
            )
            scale = min(self.width / w, self.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            x_offset = (self.width - new_w) // 2
            y_offset = (self.height - new_h) // 2
            
            pmin = (x_offset, y_offset)
            pmax = (x_offset + new_w, y_offset + new_h)

            dpg.configure_item(
                self.draw_image_id, 
                texture_tag=new_texture_tag, 
                pmin=pmin, 
                pmax=pmax
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
        print("Cleaning up UI...")
        dpg.destroy_context() 