import cv2
import numpy as np
import utils.constants as constants

class Drawer:
    def __init__(self):
        self.font = constants.FONT
        self.font_scale = constants.FONT_SCALE
        self.font_thickness = constants.FONT_THICKNESS
        self.text_offset_y = constants.TEXT_OFFSET_Y

    def draw_bbox(self, frame, bbox, color, thickness=None):
        x1, y1, x2, y2 = map(int, bbox)
        if thickness is None:
            thickness = self.font_thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def draw_text(self, frame, text, position, color, font_scale=None, font_thickness=None, offset_y=None):
        if font_scale is None: font_scale = self.font_scale
        if font_thickness is None: font_thickness = self.font_thickness
        if offset_y is None: offset_y = self.text_offset_y

        text_x, text_y = position[0], position[1] - offset_y
        
        text_size = cv2.getTextSize(text, self.font, font_scale, font_thickness)[0]
        if text_y < text_size[1] + 5:
            text_y = position[1] + text_size[1] + 5
        
        cv2.putText(frame, text, (int(text_x), int(text_y)), self.font, font_scale, color, font_thickness)

    def draw_text_with_background(self, frame, text, position, text_color, bg_color=None, padding=5, font_scale=None, font_thickness=None):
        if font_scale is None: font_scale = self.font_scale
        if font_thickness is None: font_thickness = self.font_thickness

        text_size = cv2.getTextSize(text, self.font, font_scale, font_thickness)[0]
        text_x, text_y = position[0], position[1]

        x1_bg = int(text_x - padding)
        y1_bg = int(text_y - text_size[1] - padding)
        x2_bg = int(text_x + text_size[0] + padding)
        y2_bg = int(text_y + padding)

        if bg_color is None:
            bg_color = (max(0, text_color[0]-50), max(0, text_color[1]-50), max(0, text_color[2]-50))

        cv2.rectangle(frame, (x1_bg, y1_bg), (x2_bg, y2_bg), bg_color, -1)
        
        cv2.putText(frame, text, (int(text_x), int(text_y)), self.font, font_scale, text_color, font_thickness)

    def draw_zone_boundary(self, frame, zone_polygon, color, thickness=None):
        if thickness is None: thickness = self.font_thickness
        cv2.polylines(frame, [zone_polygon], isClosed=True, color=color, thickness=thickness)

    def draw_keypoints(self, frame, keypoints, color=(0, 255, 255), radius=3, thickness=1):
        for kp in keypoints:
            if kp[2] > constants.CONF_THRESHOLD:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, color, thickness)

    def draw_fps(self, frame, fps_value, position=(10, 30)):
        cv2.putText(frame, f"FPS: {int(fps_value)}", position, self.font, self.font_scale, constants.ANNOTATION_COLOR_NORMAL, self.font_thickness)
