import cv2
import numpy as np
import time
from utils.helpers import calculate_keypoint_distance
from utils.drawer import Drawer
import utils.constants as constants

class VandalismDetector:
    def __init__(self, dist_thresh=constants.VANDALISM_DIST_THRESH, motion_thresh=constants.VANDALISM_MOTION_THRESH, max_age=constants.VANDALISM_TRACKER_MAX_AGE):
        self.prev_gray = None
        self.DIST_THRESH = dist_thresh
        self.MOTION_THRESH = motion_thresh

        self._PERSON_CLASS_ID = constants.PERSON_CLASS_ID
        self._VEHICLE_CLASS_IDS = constants.VEHICLE_CLASS_IDS
        self.drawer = Drawer()

        self.person_last_known_positions = {} 
        self._next_temp_person_id = 200000

    def _get_or_assign_temp_person_id(self, det_bbox):
        curr_x1, curr_y1, curr_x2, curr_y2 = det_bbox
        curr_cx, curr_cy = (curr_x1 + curr_x2) // 2, (curr_y1 + curr_y2) // 2

        for temp_id, last_info in list(self.person_last_known_positions.items()):
            prev_bbox = last_info['bbox']
            prev_cx, prev_cy = (prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2
            
            if calculate_keypoint_distance((prev_cx, prev_cy), (curr_cx, curr_cy)) < 70:
                return temp_id
        
        new_temp_id = f"v_temp_{self._next_temp_person_id}"
        self._next_temp_person_id += 1
        return new_temp_id

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        return blur

    def calculate_motion(self, current_blur, bbox):
        if self.prev_gray is None:
            return 0

        diff = cv2.absdiff(self.prev_gray, current_blur)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x1, y1, x2, y2 = map(int, bbox)
        motion_area = 0
        for c in contours:
            if cv2.contourArea(c) < 50:
                continue
            cx, cy, cw, ch = cv2.boundingRect(c)
            
            if x1 < cx + cw // 2 < x2 and y1 < cy + ch // 2 < y2:
                motion_area += cv2.contourArea(c)
        return motion_area

    def process_frame(self, frame, person_detections, car_detections):
        annotated_frame = frame.copy()
        current_blur = self.preprocess_frame(annotated_frame)
        violation_detected_this_frame = False

        if self.prev_gray is None:
            self.prev_gray = current_blur
            return annotated_frame, False
        
        current_frame_person_ids = set()
        for p_data in person_detections:
            p_id = p_data.get('id')
            if p_id is None:
                p_id = self._get_or_assign_temp_person_id(p_data['box'])
                p_data['id'] = p_id
            current_frame_person_ids.add(p_id)
            self.person_last_known_positions[p_id] = {'bbox': p_data['box'], 'time': time.time()}

        car_centers = [((c_data['box'][0] + c_data['box'][2]) // 2, (c_data['box'][1] + c_data['box'][3]) // 2)
                       for c_data in car_detections]

        for p_data in person_detections:
            p_id = p_data['id']
            x1, y1, x2, y2 = p_data['box']
            label = p_data['label']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            near_car = any(calculate_keypoint_distance((cx, cy), cc) < self.DIST_THRESH for cc in car_centers)

            motion = self.calculate_motion(current_blur, (x1, y1, x2, y2))

            color, display_label = constants.ANNOTATION_COLOR_NORMAL, f"ID {p_id}"
            
            if near_car and motion > self.MOTION_THRESH:
                color, display_label = constants.ANNOTATION_COLOR_VIOLATION, f"VANDALISM! ID {p_id}"
                violation_detected_this_frame = True
                print(f"[Vandalism] ALERT: Person ID {p_id} near car with motion {motion:.2f} > {self.MOTION_THRESH}")

            self.drawer.draw_bbox(annotated_frame, (x1, y1, x2, y2), color)
            self.drawer.draw_text(annotated_frame, display_label, (x1, y1), color)
            
            if violation_detected_this_frame:
                self.drawer.draw_text_with_background(annotated_frame, "VANDALISM ALERT!", (x1, y2 + 20),
                                                    constants.ANNOTATION_COLOR_VIOLATION)

        persons_to_remove = []
        for p_id, info in list(self.person_last_known_positions.items()):
            # Use GENERAL_TRACK_INACTIVE_SECONDS for cleanup
            if p_id not in current_frame_person_ids and (time.time() - info['time']) > constants.GENERAL_TRACK_INACTIVE_SECONDS:
                persons_to_remove.append(p_id)
        for p_id in persons_to_remove:
            del self.person_last_known_positions[p_id]

        self.prev_gray = current_blur
        return annotated_frame, violation_detected_this_frame

