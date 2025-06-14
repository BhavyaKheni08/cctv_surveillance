import cv2
import numpy as np
from utils.helpers import is_point_in_zone

class EntryZoneDetector:
    def __init__(self, entry_zones_data=None):
        self.entry_zones = []
        if entry_zones_data:
            for zone_coords in entry_zones_data:
                self.entry_zones.append(np.array(zone_coords, np.int32).reshape((-1, 1, 2)))

        if not self.entry_zones:
            print("Warning: No entry zones provided for EntryZoneDetector.")

    def process_frame(self, frame, person_detections):
        annotated_frame = frame.copy()
        
        for zone in self.entry_zones:
            cv2.polylines(annotated_frame, [zone], isClosed=True, color=(255, 255, 0), thickness=2)
            cv2.fillPoly(annotated_frame, [zone], (255, 255, 0))

        entry_detected_this_frame = False

        for person_data in person_detections:
            x1, y1, x2, y2 = person_data['box']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center = (cx, cy)

            for idx, zone in enumerate(self.entry_zones):
                if is_point_in_zone(center, zone):
                    print(f"  [EntryZone] Person ID: {person_data.get('id', 'N/A')} entered zone {idx + 1}") 
                    cv2.putText(annotated_frame, f"Entered Zone {idx+1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    entry_detected_this_frame = True
                    break

        return annotated_frame
