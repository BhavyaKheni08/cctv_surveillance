import cv2
import numpy as np
import time
import os
from utils.helpers import is_point_in_zone
from config import VIOLATION_IMAGE_SAVE_DIR, FORBIDDEN_ZONE_COOLDOWN_SECONDS, ANNOTATION_COLOR_VIOLATION, ANNOTATION_COLOR_WARNING, ANNOTATION_COLOR_ZONE_BOUNDARY

class ForbiddenZoneDetector:
    def __init__(self, zones_data, violation_object_classes, violation_text="VIOLATION IN FORBIDDEN ZONE!", violation_image_prefix="forbidden_zone_violation_frame"):
        self.zones = []
        if zones_data:
            for zone_coords in zones_data:
                self.zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        if not self.zones:
            print(f"Warning: No forbidden zones provided for {violation_text.split(' ')[0]} detector.")

        self.violation_object_classes = violation_object_classes
        self.violation_text = violation_text
        self.violation_image_prefix = violation_image_prefix
        self.last_violation_time = 0

    def process_frame(self, frame, detections, frame_count):
        violation_detected_this_frame = False
        display_frame = frame.copy()

        for zone in self.zones:
            cv2.polylines(display_frame, [zone], isClosed=True, color=ANNOTATION_COLOR_ZONE_BOUNDARY, thickness=2)
            cv2.fillPoly(display_frame, [zone], (0, 0, 255))

        for det in detections:
            cls_id = det['cls']
            box = det['box']
            x1, y1, x2, y2 = box

            if cls_id in self.violation_object_classes:
                center_x = int((x1 + x2) / 2)
                bottom_center_y = int(y2)
                object_point = (center_x, bottom_center_y)

                for zone in self.zones:
                    if is_point_in_zone(object_point, zone):
                        violation_detected_this_frame = True
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), ANNOTATION_COLOR_VIOLATION, 2)
                        cv2.putText(display_frame, self.violation_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ANNOTATION_COLOR_VIOLATION, 2)
                        
                        current_time = time.time()
                        if current_time - self.last_violation_time > FORBIDDEN_ZONE_COOLDOWN_SECONDS:
                            output_filename = os.path.join(VIOLATION_IMAGE_SAVE_DIR, f"{self.violation_image_prefix}_{int(current_time)}.jpg")
                            cv2.imwrite(output_filename, frame)
                            print(f"ALERT! {self.violation_text} at frame {frame_count} - image saved as {output_filename}")
                            self.last_violation_time = current_time

                        break

        return display_frame, violation_detected_this_frame
