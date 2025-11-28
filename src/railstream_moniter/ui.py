import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from utils import print_time, BoundingBoxColor

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
        self.bounding_box_id = "stream_bounding_box"

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

        dpg.split_frame()

        drawlist_size = dpg.get_item_rect_size(self.drawlist_id)
        drawlist_width = drawlist_size[0]
        drawlist_height = drawlist_size[1]

        # This doesn't work unless it it told to be the size of itself
        dpg.configure_item(
            self.drawlist_id,
            width=drawlist_width,
            height=drawlist_height
        )

        w, h = self._current_frame_dims

        if(w != 0 and h != 0):

            scale = min(drawlist_width / w, drawlist_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            x_offset = (drawlist_width - new_w) // 2
            y_offset = (drawlist_height - new_h) // 2
            
            pmin = (x_offset, y_offset)
            pmax = (x_offset + new_w, y_offset + new_h)

            dpg.configure_item(
                self.draw_image_id, 
                pmin = pmin,
                pmax = pmax
            )

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

        with dpg.window( # Inner Window
            tag=self.window_tag, 
            label="Video Stream", 
            width=self.width * 0.8,
            height=self.height * 0.8,
            no_close=True,
            no_resize=False,
            no_collapse=False
        ):
            
            with dpg.drawlist(
                tag=self.drawlist_id,
                width=-1,
                height=-1
            ):
                dpg.draw_rectangle(
                    pmin=(0, 0), 
                    pmax=(99999, 99999), 
                    color=(0, 0, 0, 0),
                    fill=(0, 0, 0, 100),
                    tag="drawlist_background_rect"
                )
                
                dpg.draw_image(
                    self.texture_id,
                    pmin=(0, 0),
                    pmax=(0, 0),
                    tag=self.draw_image_id
                )

                dpg.draw_rectangle(
                    pmin=(0, 0),
                    pmax=(0, 0),
                    color=(0, 255, 0, 255),
                    thickness=2,
                    show=False,
                    tag=self.bounding_box_id
                )

        with dpg.item_handler_registry(tag="window_resize_handler_reg"):
            dpg.add_item_resize_handler(callback=self._on_window_resize)

        dpg.bind_item_handler_registry(self.window_tag, "window_resize_handler_reg")

        dpg.create_viewport(title=self.title, width=1280, height=720) 
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def update_frame(self, frame, detection_result = None):
        
        # Texture

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

            drawlist_width = dpg.get_item_rect_size(self.drawlist_id)[0]
            drawlist_height = dpg.get_item_rect_size(self.drawlist_id)[1]

            if drawlist_width == 0: drawlist_width = self.width
            if drawlist_height == 0: drawlist_height = self.height

            dpg.configure_item(
                self.draw_image_id, 
                texture_tag=new_texture_tag, 
                pmin=(0, 0), 
                pmax=(drawlist_width, drawlist_height)
            )

            self.texture_id = new_texture_tag
            self._current_frame_dims = frame_dims
        else:
            dpg.set_value(self.texture_id, texture_data)

        # Detection Box
        confidence = detection_result['confidence']
        box = detection_result['box']
        color = detection_result['color']

        if detection_result and box and confidence:

            image_config = dpg.get_item_configuration(self.draw_image_id)
            draw_pmin = image_config['pmin']
            draw_pmax = image_config['pmax']

            draw_width = draw_pmax[0] - draw_pmin[0]
            draw_height = draw_pmax[1] - draw_pmin[1]

            if draw_width == 0 or draw_height == 0:
                return

            w_frame, h_frame = self._current_frame_dims

            cx, cy, box_w_norm, box_h_norm = box

            norm_xmin = cx - box_w_norm / 2.0
            norm_ymin = cy - box_h_norm / 2.0
            norm_xmax = cx + box_w_norm / 2.0
            norm_ymax = cy + box_h_norm / 2.0

            box_xmin = draw_pmin[0] + norm_xmin * draw_width
            box_ymin = draw_pmin[1] + norm_ymin * draw_height
            box_xmax = draw_pmin[0] + norm_xmax * draw_width
            box_ymax = draw_pmin[1] + norm_ymax * draw_height

            dpg.configure_item(
                self.bounding_box_id,
                pmin=(box_xmin, box_ymin),
                pmax=(box_xmax, box_ymax),
                show=True,
                color=color.value,
                thickness=3
            )
        else:
            if dpg.does_item_exist(self.bounding_box_id):
                dpg.configure_item(self.bounding_box_id, show=False)

    def is_running(self):
        return dpg.is_dearpygui_running()

    def render(self):
        dpg.render_dearpygui_frame()

    def cleanup(self):
        print_time("Cleaning up UI...")
        dpg.destroy_context() 