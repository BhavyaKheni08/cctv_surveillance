import cv2
import numpy as np
import time
import os
from utils.helpers import is_point_in_zone
from utils.drawer import Drawer
import utils.constants as constants

class ForbiddenZoneDetector:
    def __init__(self, zones_data, violation_object_classes, violation_text="VIOLATION IN FORBIDDEN ZONE!", violation_image_prefix="forbidden_zone_violation_frame"):
        self.zones = []
        if zones_data:
            for zone_coords in zones_data:
                self.zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        self.violation_object_classes = violation_object_classes
        self.violation_text = violation_text
        self.violation_image_prefix = violation_image_prefix
        self.last_violation_time = 0

        self.object_entry_times = {} 
        self.WARNING_DURATION_SECONDS = 4 

        self.untracked_obj_temp_ids = {} 
        self._next_temp_id = 100000 

        self.drawer = Drawer()

    def _get_or_assign_temp_id(self, bbox):
        for existing_bbox_key, temp_id in list(self.untracked_obj_temp_ids.items()):
            x1_prev, y1_prev, x2_prev, y2_prev = existing_bbox_key
            x1_curr, y1_curr, x2_curr, y2_curr = bbox

            if abs(x1_prev - x1_curr) < 50 and \
               abs(y1_prev - y1_curr) < 50 and \
               abs(x2_prev - x2_curr) < 50 and \
               abs(y2_prev - y2_curr) < 50:
                return temp_id
        
        new_temp_id = self._next_temp_id
        self._next_temp_id += 1
        self.untracked_obj_temp_ids[bbox] = new_temp_id
        return new_temp_id


    def process_frame(self, frame, detections, frame_count):
        violation_detected_this_frame = False
        display_frame = frame.copy()
        current_time = time.time()

        if display_frame is None: 
            return display_frame, False

        objects_currently_in_zone = set()
        
        current_frame_untracked_bboxes = [] 

        for zone_idx, zone in enumerate(self.zones):
            self.drawer.draw_zone_boundary(display_frame, zone, constants.ANNOTATION_COLOR_ZONE_BOUNDARY)

            for det in detections:
                cls_id = det['cls']
                box = det['box']
                track_id = det['id']
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                bottom_center_y = int(y2)
                object_point = (center_x, bottom_center_y)

                if cls_id in self.violation_object_classes:
                    if is_point_in_zone(object_point, zone):
                        violation_detected_this_frame = True

                        actual_id = track_id
                        if actual_id is None:
                            actual_id = self._get_or_assign_temp_id(box)
                            current_frame_untracked_bboxes.append(box)

                        objects_currently_in_zone.add(actual_id)

                        if actual_id not in self.object_entry_times:
                            self.object_entry_times[actual_id] = current_time
                            duration_in_zone = 0 
                        else:
                            duration_in_zone = current_time - self.object_entry_times[actual_id]

                        annotation_color = constants.ANNOTATION_COLOR_NORMAL
                        annotation_text = f"IN ZONE ({int(duration_in_zone)}s)"
                        
                        if duration_in_zone >= constants.FORBIDDEN_ZONE_COOLDOWN_SECONDS:
                            annotation_color = constants.ANNOTATION_COLOR_VIOLATION 
                            annotation_text = f"{self.violation_text} ({int(duration_in_zone)}s)"
                            if current_time - self.last_violation_time > constants.FORBIDDEN_ZONE_COOLDOWN_SECONDS:
                                output_filename = os.path.join(constants.VIOLATION_IMAGE_SAVE_DIR, f"{self.violation_image_prefix}_{int(current_time)}.jpg")
                                cv2.imwrite(output_filename, frame)
                                print(f"ALERT! {self.violation_text} at frame {frame_count} - image saved as {output_filename}")
                                self.last_violation_time = current_time
                        elif duration_in_zone >= self.WARNING_DURATION_SECONDS:
                            annotation_color = constants.ANNOTATION_COLOR_WARNING
                            annotation_text = f"WARNING! {self.violation_text} ({int(duration_in_zone)}s)"
                            print(f"WARNING! {det.get('label', 'Object')} (ID: {actual_id}) in forbidden zone for {int(duration_in_zone)} seconds at frame {frame_count}")
                        
                        self.drawer.draw_bbox(display_frame, box, annotation_color)
                        self.drawer.draw_text(display_frame, annotation_text, (x1, y1), annotation_color)
                        
                        break

        keys_to_remove = [id for id in self.object_entry_times if id not in objects_currently_in_zone]
        for id in keys_to_remove:
            if id in self.object_entry_times:
                del self.object_entry_times[id]
            temp_bbox_to_remove = None
            for bbox_key, temp_id in self.untracked_obj_temp_ids.items():
                if temp_id == id:
                    temp_bbox_to_remove = bbox_key
                    break
            if temp_bbox_to_remove:
                del self.untracked_obj_temp_ids[temp_bbox_to_remove]
        
        temp_ids_to_keep_in_map = set()
        for det in detections:
            if det['id'] is None:
                temp_id_for_current_det = self._get_or_assign_temp_id(det['box'])
                temp_ids_to_keep_in_map.add(temp_id_for_current_det)

        bboxes_to_remove_from_temp_ids = [
            bbox_key for bbox_key, temp_id in list(self.untracked_obj_temp_ids.items())
            if temp_id not in temp_ids_to_keep_in_map and temp_id not in objects_currently_in_zone
        ]
        for bbox_key in bboxes_to_remove_from_temp_ids:
            del self.untracked_obj_temp_ids[bbox_key]

        return display_frame, violation_detected_this_frame
