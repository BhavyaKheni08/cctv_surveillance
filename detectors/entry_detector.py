import cv2
import numpy as np
import time
from collections import deque
from utils.helpers import is_point_in_zone, calculate_iou
from utils.drawer import Drawer
import utils.constants as constants

class EntryZoneDetector:
    def __init__(self, entry_zones_data=None):
        self.entry_zones = []
        if entry_zones_data:
            for zone_coords in entry_zones_data:
                self.entry_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        self.all_tracked_cars = {} 
        self.objects_in_zone_last_frame = {} 

        self.IOU_THRESHOLD_FOR_MATCHING = 0.4
        self.INACTIVE_CLEANUP_SECONDS = 60.0 

        self.drawer = Drawer()
        self.untracked_id_counter = 100000

    def _get_or_assign_temp_id(self, det):
        yolo_id = det.get('id')
        current_bbox = det['box']
        current_time = time.time()

        if yolo_id is not None:
            if yolo_id in self.all_tracked_cars:
                self.all_tracked_cars[yolo_id]['last_bbox'] = current_bbox
                self.all_tracked_cars[yolo_id]['last_seen_time'] = current_time
                return yolo_id

        best_match_id = None
        max_iou = 0.0
        
        for tracked_id, tracked_info in list(self.all_tracked_cars.items()):
            prev_bbox = tracked_info['last_bbox']
            iou = calculate_iou(current_bbox, prev_bbox)
            
            if iou > max_iou and iou >= self.IOU_THRESHOLD_FOR_MATCHING and \
               (current_time - tracked_info['last_seen_time']) < self.INACTIVE_CLEANUP_SECONDS:
                max_iou = iou
                best_match_id = tracked_id
        
        if best_match_id:
            self.all_tracked_cars[best_match_id]['last_bbox'] = current_bbox
            self.all_tracked_cars[best_match_id]['last_seen_time'] = current_time
            return best_match_id
        
        new_temp_id = f"temp_{self.untracked_id_counter}"
        self.untracked_id_counter += 1
        
        self.all_tracked_cars[new_temp_id] = {
            'last_bbox': current_bbox,
            'last_seen_time': current_time,
            'entry_x': None,
            'exit_x': None,
            'is_violation_flagged': False,
        }
        return new_temp_id


    def process_frame(self, frame, all_detections):
        annotated_frame = frame.copy()
        current_time = time.time()
        violation_detected_this_frame = False
        
        if annotated_frame is None: 
            return annotated_frame, False

        for zone in self.entry_zones:
            self.drawer.draw_zone_boundary(annotated_frame, zone, constants.ANNOTATION_COLOR_ZONE_BOUNDARY)

        current_frame_objects_in_zone_data = {} 
        current_frame_car_ids_detected = set() 
        
        # First pass: Process all car detections to update states and identify entry/exit events
        for det in all_detections:
            cls_id = det['cls']
            bbox = det['box']
            label = det['label']

            if cls_id not in constants.VEHICLE_CLASS_IDS:
                continue 

            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center = (cx, cy)

            actual_id = self._get_or_assign_temp_id(det) 
            current_frame_car_ids_detected.add(actual_id) 

            car_state = self.all_tracked_cars.get(actual_id) 
            if car_state is None: 
                continue

            car_state['last_bbox'] = bbox 
            car_state['last_seen_time'] = current_time 

            is_currently_in_zone = False
            for zone in self.entry_zones:
                if is_point_in_zone(center, zone):
                    is_currently_in_zone = True
                    break
            
            was_in_zone_prev_frame = actual_id in self.objects_in_zone_last_frame
            
            if is_currently_in_zone:
                current_frame_objects_in_zone_data[actual_id] = {'bbox': bbox} 
                
                if not was_in_zone_prev_frame: 
                    car_state['entry_x'] = cx 
                    car_state['exit_x'] = None 
                    print(f"  [EntryZone] Car (ID: {actual_id}) entered zone at X={cx}.")
            
            else: 
                if was_in_zone_prev_frame:
                    car_state['exit_x'] = cx
                    print(f"  [EntryZone] Car (ID: {actual_id}) exited zone at X={cx}.")

                    if car_state['entry_x'] is not None and car_state['exit_x'] is not None and \
                       car_state['exit_x'] < car_state['entry_x']:
                        
                        car_state['is_violation_flagged'] = True
                        violation_detected_this_frame = True
                        print(f"  [EntryZone] VIOLATION! Car (ID: {actual_id}) exited left (Entry X: {car_state['entry_x']}, Exit X: {car_state['exit_x']}).")
                    
        ids_to_remove_from_all_tracked = []
        for car_id, state in list(self.all_tracked_cars.items()):
            if car_id not in current_frame_car_ids_detected and \
               (current_time - state['last_seen_time']) > self.INACTIVE_CLEANUP_SECONDS:
                ids_to_remove_from_all_tracked.append(car_id)
        for car_id in ids_to_remove_from_all_tracked:
            del self.all_tracked_cars[car_id]

        self.objects_in_zone_last_frame = current_frame_objects_in_zone_data

        # --- Drawing Loop for ALL Cars detected in current frame ---
        for det in all_detections:
            cls_id = det['cls']
            if cls_id not in constants.VEHICLE_CLASS_IDS:
                continue
            
            bbox = det['box']
            x1, y1, x2, y2 = bbox
            center = (int((x1+x2)/2), int((y1+y2)/2)) # Recalculate center for drawing

            actual_id = self._get_or_assign_temp_id(det) 

            car_state = self.all_tracked_cars.get(actual_id) 

            # ALWAYS draw red if flagged, regardless of current zone status
            if car_state and car_state['is_violation_flagged']:
                self.drawer.draw_bbox(annotated_frame, bbox, constants.ANNOTATION_COLOR_VIOLATION) 
                self.drawer.draw_text(annotated_frame, f"ID {actual_id} VIOLATION: Exited Left!", (x1, y1), constants.ANNOTATION_COLOR_VIOLATION)
            else:
                # If not flagged, then draw green ONLY if the car's current center is within ANY of the entry zones
                is_currently_in_any_entry_zone_for_drawing = False
                for zone in self.entry_zones:
                    if is_point_in_zone(center, zone):
                        is_currently_in_any_entry_zone_for_drawing = True
                        break
                
                if is_currently_in_any_entry_zone_for_drawing:
                    self.drawer.draw_bbox(annotated_frame, bbox, constants.ANNOTATION_COLOR_NORMAL)
                    self.drawer.draw_text(annotated_frame, f"ID {actual_id} In Zone", (x1, y1), constants.ANNOTATION_COLOR_NORMAL)
                # If not flagged AND not currently in any entry zone, this detector does not draw anything for it.

        return annotated_frame, violation_detected_this_frame


        return annotated_frame
