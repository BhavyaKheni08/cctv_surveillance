import cv2
import numpy as np
import time
from utils.helpers import is_point_in_zone, calculate_keypoint_distance, get_midpoint
from config import VEHICLE_CLASS_IDS, PERSON_CLASS_ID, STOP_DURATION_SECONDS,PERSON_PROXIMITY_THRESHOLD_PIXELS, PERSON_GRACE_PERIOD_SECONDS,ANNOTATION_COLOR_NORMAL, ANNOTATION_COLOR_WARNING, ANNOTATION_COLOR_VIOLATION, ANNOTATION_COLOR_ZONE_BOUNDARY

class StoppingDetector:
    def __init__(self, no_stopping_zones_data=None, stop_duration_sec=STOP_DURATION_SECONDS, 
                 person_proximity_threshold_pixels=PERSON_PROXIMITY_THRESHOLD_PIXELS, 
                 person_detection_grace_sec=PERSON_GRACE_PERIOD_SECONDS):
        self.no_stopping_zones = []
        if no_stopping_zones_data:
            for zone_coords in no_stopping_zones_data:
                self.no_stopping_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        if not self.no_stopping_zones:
            print("Warning: No 'No Stopping' zones provided for StoppingDetector.")

        self.stop_duration_sec = stop_duration_sec
        self.person_proximity_threshold_pixels = person_proximity_threshold_pixels
        self.person_detection_grace_sec = person_detection_grace_sec

        self.vehicle_states = {}
        self.person_tracked_data = {}

        self.MOVEMENT_THRESHOLD_PIXELS = 5

    def process_frame(self, frame, vehicle_detections, person_deteations):
        annotated_frame = frame.copy()
        current_time = time.time()

        for zone in self.no_stopping_zones:
            cv2.polylines(annotated_frame, [zone], isClosed=True, color=ANNOTATION_COLOR_ZONE_BOUNDARY, thickness=2)
            cv2.fillPoly(annotated_frame, [zone], (0, 0, 255))

        current_frame_person_data = {}
        for p_data in person_deteations:
            x1, y1, x2, y2 = p_data['box']
            p_id = p_data.get('id')
            if p_id is None: continue

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            current_frame_person_data[p_id] = {'center': (center_x, center_y), 'bbox': p_data['box']}

        current_frame_tracked_vehicle_ids = set()
        for v_data in vehicle_detections:
            x1, y1, x2, y2 = v_data['box']
            v_id = v_data.get('id')
            if v_id is None: continue

            object_type_name = v_data.get('label', 'Vehicle')
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
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
                for p_id, p_data in current_frame_person_data.items():
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
                else:
                    v_state['violation_triggered'] = False

        vehicles_to_remove = []
        for v_id, v_state in list(self.vehicle_states.items()):
            if v_id not in current_frame_tracked_vehicle_ids and \
               (current_time - v_state['last_active_time'] > self.stop_duration_sec * 2):
                vehicles_to_remove.append(v_id)

        for v_id in vehicles_to_remove:
            del self.vehicle_states[v_id]

        for v_id, v_state in self.vehicle_states.items():
            x1, y1, x2, y2 = v_state['last_bbox']
            
            color = ANNOTATION_COLOR_NORMAL
            label = f"{v_state['type']} ID: {v_id}"
            
            if v_state['in_zone']:
                if v_state['violation_triggered']:
                    color = ANNOTATION_COLOR_VIOLATION
                    label += f" - STOPPING VIOLATION!"
                    cv2.putText(annotated_frame, "NO STOPPING/DROPPING!", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ANNOTATION_COLOR_VIOLATION, 3)
                elif v_state['is_stopped']:
                    color = ANNOTATION_COLOR_WARNING
                    duration_stopped = current_time - v_state['stopped_start_time']
                    label += f" (Stopped: {duration_stopped:.1f}s)"
                    cv2.putText(annotated_frame, f"Stopped: {duration_stopped:.1f}s", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ANNOTATION_COLOR_WARNING, 2)
                else:
                    color = (255, 0, 0)
                    label += f" (In Zone)"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for p_data in person_deteations:
            x1, y1, x2, y2 = p_data['box']
            p_id = p_data.get('id', 'N/A')
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(annotated_frame, f"P ID: {p_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        return annotated_frame
