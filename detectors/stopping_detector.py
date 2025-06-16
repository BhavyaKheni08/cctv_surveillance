import cv2
import numpy as np
import time
from utils.helpers import is_point_in_zone, calculate_keypoint_distance, get_midpoint
from utils.drawer import Drawer
import utils.constants as constants

class StoppingDetector:
    def __init__(self, no_stopping_zones_data=None, stop_duration_sec=constants.STOP_DURATION_SECONDS, 
                 person_proximity_threshold_pixels=constants.PERSON_PROXIMITY_THRESHOLD_PIXELS, 
                 person_detection_grace_sec=constants.PERSON_GRACE_PERIOD_SECONDS):
        self.no_stopping_zones = []
        if no_stopping_zones_data:
            for zone_coords in no_stopping_zones_data:
                self.no_stopping_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        self.stop_duration_sec = stop_duration_sec
        self.person_proximity_threshold_pixels = person_proximity_threshold_pixels
        self.person_detection_grace_sec = person_detection_grace_sec

        self.vehicle_states = {} 
        self.person_last_known_positions = {} 

        self.MOVEMENT_THRESHOLD_PIXELS = 5

        self.drawer = Drawer()
        self._next_temp_vehicle_id = 300000
        self._next_temp_person_id = 400000

    def _get_or_assign_temp_id(self, det_data, object_type='vehicle'):
        yolo_id = det_data.get('id')
        current_bbox = det_data['box']
        
        if yolo_id is not None:
            return yolo_id

        tracking_dict = self.vehicle_states if object_type == 'vehicle' else self.person_last_known_positions
        next_temp_id_counter = self._next_temp_vehicle_id if object_type == 'vehicle' else self._next_temp_person_id
        prefix = 'v_temp_' if object_type == 'vehicle' else 'p_temp_'

        curr_cx, curr_cy = (current_bbox[0] + current_bbox[2]) // 2, (current_bbox[1] + current_bbox[3]) // 2

        for obj_id, obj_state in list(tracking_dict.items()):
            if isinstance(obj_id, str) and obj_id.startswith(prefix):
                prev_bbox = obj_state.get('last_bbox', obj_state['bbox'])
                if prev_bbox:
                    prev_cx, prev_cy = (prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2
                    if calculate_keypoint_distance((prev_cx, prev_cy), (curr_cx, curr_cy)) < 70:
                        return obj_id
        
        new_temp_id = f"{prefix}{next_temp_id_counter}"
        if object_type == 'vehicle':
            self._next_temp_vehicle_id += 1
        else:
            self._next_temp_person_id += 1
        return new_temp_id

    def process_frame(self, frame, vehicle_detections, person_detections):
        annotated_frame = frame.copy()
        current_time = time.time()
        violation_detected_this_frame = False

        if annotated_frame is None:
            return annotated_frame, False

        for zone in self.no_stopping_zones:
            self.drawer.draw_zone_boundary(annotated_frame, zone, constants.ANNOTATION_COLOR_ZONE_BOUNDARY)

        current_frame_person_ids = set()
        for p_data in person_detections:
            p_id = self._get_or_assign_temp_id(p_data, 'person')
            p_data['id'] = p_id 
            
            center_x, center_y = (p_data['box'][0] + p_data['box'][2]) // 2, (p_data['box'][1] + p_data['box'][3]) // 2
            current_frame_person_ids.add(p_id)
            self.person_last_known_positions[p_id] = {'center': (center_x, center_y), 'bbox': p_data['box'], 'label': p_data['label'], 'time': current_time}

        current_frame_tracked_vehicle_ids = set()
        for v_data in vehicle_detections:
            v_id = self._get_or_assign_temp_id(v_data, 'vehicle')
            v_data['id'] = v_id

            object_type_name = v_data.get('label', 'Vehicle')
            center_x, center_y = (v_data['box'][0] + v_data['box'][2]) // 2, (v_data['box'][1] + v_data['box'][3]) // 2
            current_frame_tracked_vehicle_ids.add(v_id)

            is_currently_in_zone = False
            for zone in self.no_stopping_zones:
                if is_point_in_zone((center_x, center_y), zone):
                    is_currently_in_zone = True
                    break

            if v_id not in self.vehicle_states:
                self.vehicle_states[v_id] = {
                    'type': object_type_name,
                    'last_bbox': v_data['box'],
                    'last_center': (center_x, center_y),
                    'in_zone': is_currently_in_zone,
                    'is_stopped': False,
                    'stopped_start_time': current_time, 
                    'violation_triggered': False,
                    'last_person_interaction_time': 0.0,
                    'last_active_time': current_time
                }
            else:
                v_state = self.vehicle_states[v_id]
                prev_center = v_state['last_center']
                movement = calculate_keypoint_distance((center_x, center_y), prev_center)

                v_state['last_bbox'] = v_data['box']
                v_state['last_center'] = (center_x, center_y)
                v_state['in_zone'] = is_currently_in_zone
                v_state['type'] = object_type_name
                v_state['last_active_time'] = current_time

                if v_state['in_zone'] and movement < self.MOVEMENT_THRESHOLD_PIXELS:
                    if not v_state['is_stopped']:
                        v_state['is_stopped'] = True
                        v_state['stopped_start_time'] = current_time
                else:
                    v_state['is_stopped'] = False

                person_interaction_in_this_frame = False
                for p_id, p_data in self.person_last_known_positions.items():
                    if calculate_keypoint_distance(v_state['last_center'], p_data['center']) < self.person_proximity_threshold_pixels:
                        person_interaction_in_this_frame = True
                        break
                
                if person_interaction_in_this_frame:
                    v_state['last_person_interaction_time'] = current_time
                
                is_vehicle_stopped_long_enough = v_state['in_zone'] and \
                                                  v_state['is_stopped'] and \
                                                  (current_time - v_state['stopped_start_time'] >= self.stop_duration_sec)
                
                person_condition_met_for_violation = (v_state['last_person_interaction_time'] > 0 and \
                                                      (current_time - v_state['last_person_interaction_time'] <= self.person_detection_grace_sec))

                if is_vehicle_stopped_long_enough and person_condition_met_for_violation:
                    if not v_state['violation_triggered']:
                        print(f"[Stopping] VIOLATION! {v_state['type']} ID: {v_id} stopped for "
                              f"{(current_time - v_state['stopped_start_time']):.1f}s with person interaction.")
                        v_state['violation_triggered'] = True
                        violation_detected_this_frame = True
                else:
                    v_state['violation_triggered'] = False

        vehicles_to_remove = []
        for v_id, v_state in list(self.vehicle_states.items()):
            if v_id not in current_frame_tracked_vehicle_ids and \
               (time.time() - v_state['last_active_time'] > self.stop_duration_sec * 2):
                vehicles_to_remove.append(v_id)

        for v_id in vehicles_to_remove:
            del self.vehicle_states[v_id]
        
        persons_to_remove = []
        for p_id, p_data in list(self.person_last_known_positions.items()):
            if p_id not in current_frame_person_ids and \
               (time.time() - p_data['time'] > constants.MAX_TRACK_INACTIVE_SECONDS):
                persons_to_remove.append(p_id)
        for p_id in persons_to_remove:
            del self.person_last_known_positions[p_id]


        for v_id, v_state in self.vehicle_states.items():
            x1, y1, x2, y2 = v_state['last_bbox']
            
            color = constants.ANNOTATION_COLOR_NORMAL
            label = f"{v_state['type']} ID: {v_id}"
            
            if v_state['in_zone']:
                if v_state['violation_triggered']:
                    color = constants.ANNOTATION_COLOR_VIOLATION
                    label += f" - STOPPING VIOLATION!"
                    self.drawer.draw_text_with_background(annotated_frame, "NO STOPPING/DROPPING!", (x1, y2 + 25),
                                constants.ANNOTATION_COLOR_VIOLATION)
                elif v_state['is_stopped']:
                    color = constants.ANNOTATION_COLOR_WARNING
                    duration_stopped = current_time - v_state['stopped_start_time']
                    label += f" (Stopped: {duration_stopped:.1f}s)"
                    self.drawer.draw_text(annotated_frame, f"Stopped: {duration_stopped:.1f}s", (x1, y2 + 25),
                                constants.ANNOTATION_COLOR_WARNING)
                else:
                    color = (255, 0, 0)
                    label += f" (In Zone)"

            self.drawer.draw_bbox(annotated_frame, (x1, y1, x2, y2), color)
            self.drawer.draw_text(annotated_frame, label, (x1, y1), color)

        for p_id, p_data in self.person_last_known_positions.items():
            x1, y1, x2, y2 = p_data['bbox']
            self.drawer.draw_bbox(annotated_frame, (x1, y1, x2, y2), (0, 200, 0), thickness=1)
            self.drawer.draw_text(annotated_frame, f"P ID: {p_id}", (x1, y1), (0, 200, 0), offset_y=constants.TEXT_OFFSET_Y + 15)

        return annotated_frame, violation_detected_this_frame

