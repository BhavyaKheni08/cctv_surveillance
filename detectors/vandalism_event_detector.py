import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import VANDALISM_DIST_THRESH, VANDALISM_MOTION_THRESH, VANDALISM_TRACKER_MAX_AGE,PERSON_CLASS_ID, ANNOTATION_COLOR_NORMAL, ANNOTATION_COLOR_VIOLATION

class VandalismDetector:
    def __init__(self, dist_thresh=VANDALISM_DIST_THRESH, motion_thresh=VANDALISM_MOTION_THRESH, max_age=VANDALISM_TRACKER_MAX_AGE):
        self.tracker = DeepSort(max_age=max_age)
        self.prev_gray = None
        self.DIST_THRESH = dist_thresh
        self.MOTION_THRESH = motion_thresh

        self._PERSON_CLASS_ID = PERSON_CLASS_ID

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

        if self.prev_gray is None:
            self.prev_gray = current_blur
            return annotated_frame
        
        ds_person_detections = []
        for p_data in person_detections:
            x1, y1, x2, y2 = p_data['box']
            conf = p_data['conf'] 
            ds_person_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks = self.tracker.update_tracks(ds_person_detections, frame=annotated_frame)

        car_centers = [((c_data['box'][0] + c_data['box'][2]) // 2, (c_data['box'][1] + c_data['box'][3]) // 2)
                       for c_data in car_detections]

        for t in tracks:
            if not t.is_confirmed():
                continue

            x1, y1, x2, y2 = t.to_ltrb()
            track_id = t.track_id
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            near_car = any(np.hypot(cx - ccx, cy - ccy) < self.DIST_THRESH for ccx, ccy in car_centers)

            motion = self.calculate_motion(current_blur, (x1, y1, x2, y2))

            color, label = ANNOTATION_COLOR_NORMAL, f"ID {track_id}"
            vandalism_detected = False

            if near_car and motion > self.MOTION_THRESH:
                color, label = ANNOTATION_COLOR_VIOLATION, f"VANDALISM! ID {track_id}"
                vandalism_detected = True
                print(f"[Vandalism] ALERT: Person ID {track_id} near car with motion {motion:.2f} > {self.MOTION_THRESH}")

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if vandalism_detected:
                cv2.putText(annotated_frame, "VANDALISM ALERT!", (int(x1), int(y2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, ANNOTATION_COLOR_VIOLATION, 2)

        self.prev_gray = current_blur
        return annotated_frame
