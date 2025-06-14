import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

from config import (
    DEFAULT_VIDEO_SOURCE, YOLO_GENERAL_MODEL_PATH, YOLO_POSE_MODEL_PATH,
    CLASS_NAMES, PERSON_CLASS_ID, VEHICLE_CLASS_IDS, BIKE_CLASS_IDS,
    DISPLAY_WINDOW_NAME_PREFIX, ANNOTATION_COLOR_VIOLATION, ANNOTATION_COLOR_NORMAL,
    ANNOTATION_COLOR_WARNING, ANNOTATION_COLOR_ZONE_BOUNDARY, ANNOTATION_COLOR_ZONE_FILL,
    FORBIDDEN_ZONE_COOLDOWN_SECONDS, VANDALISM_DIST_THRESH, VANDALISM_MOTION_THRESH,
    VANDALISM_TRACKER_MAX_AGE, STOP_DURATION_SECONDS, PERSON_PROXIMITY_THRESHOLD_PIXELS,
    PERSON_GRACE_PERIOD_SECONDS, VIOLATION_IMAGE_SAVE_DIR,
    NO_BIKE_ZONES, NO_PARKING_ZONES, ENTRY_ZONES, NO_STOPPING_ZONES
)

from detectors.entry_detector import EntryZoneDetector
from detectors.forbidden_area_detector import ForbiddenZoneDetector
from detectors.behavior_detector import HumanBehaviorDetector
from detectors.vandalism_event_detector import VandalismDetector
from detectors.stopping_detector import StoppingDetector

general_object_model = None
pose_estimation_model = None

def load_yolo_models():
    global general_object_model, pose_estimation_model
    try:
        if general_object_model is None:
            general_object_model = YOLO(YOLO_GENERAL_MODEL_PATH)
            print(f"Loaded general object detection model: {YOLO_GENERAL_MODEL_PATH}")
        if pose_estimation_model is None:
            pose_estimation_model = YOLO(YOLO_POSE_MODEL_PATH)
            print(f"Loaded pose estimation model: {YOLO_POSE_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading YOLO models: {e}. Make sure paths are correct and models exist.")
        exit()

def run_camera_surveillance(video_source, camera_name="DefaultCamera"):
    load_yolo_models()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}' for '{camera_name}'.")
        print("Please check the path/URL or ensure camera is available.")
        return

    print(f"\n Initializing detectors for {camera_name} with predefined zones ")

    forbidden_bike_zone_detector = ForbiddenZoneDetector(
        zones_data=NO_BIKE_ZONES,
        violation_object_classes=BIKE_CLASS_IDS,
        violation_text="BIKE FORBIDDEN!",
        violation_image_prefix="bike_zone_violation"
    )

    forbidden_car_zone_detector = ForbiddenZoneDetector(
        zones_data=NO_PARKING_ZONES,
        violation_object_classes=VEHICLE_CLASS_IDS,
        violation_text="VEHICLE NO PARKING!",
        violation_image_prefix="car_zone_violation"
    )

    entry_zone_detector = EntryZoneDetector(entry_zones_data=ENTRY_ZONES)
    human_behavior_detector = HumanBehaviorDetector()
    vandalism_detector = VandalismDetector(
        dist_thresh=VANDALISM_DIST_THRESH,
        motion_thresh=VANDALISM_MOTION_THRESH,
        max_age=VANDALISM_TRACKER_MAX_AGE
    )
    stopping_detector = StoppingDetector(
        no_stopping_zones_data=NO_STOPPING_ZONES,
        stop_duration_sec=STOP_DURATION_SECONDS,
        person_proximity_threshold_pixels=PERSON_PROXIMITY_THRESHOLD_PIXELS,
        person_detection_grace_sec=PERSON_GRACE_PERIOD_SECONDS
    )

    frame_count = 0
    display_window_name = f"{DISPLAY_WINDOW_NAME_PREFIX} - {camera_name}"
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)

    print(f"\nStarting video processing for '{camera_name}'. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error reading frame for '{camera_name}'.")
            if isinstance(video_source, str) and (video_source.startswith("http") or video_source.startswith("rtsp")):
                print(f"Attempting to reconnect to stream '{video_source}'...")
                cap.release()
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    print(f"Failed to reconnect to stream '{video_source}'. Exiting '{camera_name}'.")
                    break
                else:
                    print(f"Reconnected to stream '{video_source}'.")
                    continue
            else: 
                break

        frame_count += 1
        frame_height, frame_width = frame.shape[:2]

        results_general = general_object_model(frame, verbose=False)[0]
        
        all_detections_info = []
        person_detections = []
        car_detections = []
        bike_detections_for_forbidden_zone = []

        class_names = general_object_model.names

        if results_general.boxes is not None:
            for box in results_general.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                label = class_names.get(cls_id, f"unknown_class_{cls_id}")
                
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                detection_data = {
                    'box': (x1, y1, x2, y2),
                    'cls': cls_id,
                    'conf': conf,
                    'label': label,
                    'id': track_id
                }
                all_detections_info.append(detection_data)

                if cls_id == PERSON_CLASS_ID:
                    person_detections.append(detection_data)
                elif cls_id in VEHICLE_CLASS_IDS:
                    car_detections.append(detection_data)
                elif cls_id in BIKE_CLASS_IDS:
                    bike_detections_for_forbidden_zone.append(detection_data)

        person_detections_with_keypoints = []
        results_pose = pose_estimation_model(frame, verbose=False)[0]

        if results_pose.boxes is not None and results_pose.keypoints is not None:
            for i, box in enumerate(results_pose.boxes):
                if int(box.cls[0]) == PERSON_CLASS_ID: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    kps_xy = results_pose.keypoints.xy.cpu().numpy()[i]
                    kps_conf = results_pose.keypoints.conf.cpu().numpy()[i]
                    
                    formatted_kps = []
                    for j in range(len(kps_xy)):
                        formatted_kps.append([kps_xy[j][0], kps_xy[j][1], kps_conf[j]])

                    matched_id = None
                    for p_det in person_detections:
                        p_x1, p_y1, p_x2, p_y2 = p_det['box']
                        if box.id is not None and p_det['id'] == int(box.id[0]):
                             matched_id = int(box.id[0])
                             break
                    if matched_id is None:
                        matched_id = i

                    person_detections_with_keypoints.append({
                        'box': (x1, y1, x2, y2),
                        'keypoints': formatted_kps,
                        'id': matched_id
                    })

        zones_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for zone in forbidden_bike_zone_detector.zones:
            cv2.fillPoly(zones_overlay, [zone], (0, 0, 255))
        for zone in forbidden_car_zone_detector.zones:
            cv2.fillPoly(zones_overlay, [zone], (0, 0, 255))

        for zone in entry_zone_detector.entry_zones:
            cv2.fillPoly(zones_overlay, [zone], (255, 255, 0))

        for zone in stopping_detector.no_stopping_zones:
            cv2.fillPoly(zones_overlay, [zone], (255, 0, 0))

        current_annotated_frame = cv2.addWeighted(frame, 1, zones_overlay, 0.2, 0)

        current_annotated_frame, bike_violation_detected = forbidden_bike_zone_detector.process_frame(
            current_annotated_frame, bike_detections_for_forbidden_zone, frame_count
        )

        current_annotated_frame, car_violation_detected = forbidden_car_zone_detector.process_frame(
            current_annotated_frame, car_detections, frame_count
        )

        current_annotated_frame = entry_zone_detector.process_frame(
            current_annotated_frame, person_detections
        )

        current_annotated_frame, stable_fight_alert = human_behavior_detector.process_frame(
            current_annotated_frame, person_detections_with_keypoints, frame_width, frame_height
        )
        if stable_fight_alert:
            cv2.putText(current_annotated_frame, "FIGHT ALERT!", (frame_width - 300, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, ANNOTATION_COLOR_VIOLATION, 3)

        current_annotated_frame = vandalism_detector.process_frame(
            current_annotated_frame, person_detections, car_detections
        )
        
        current_annotated_frame = stopping_detector.process_frame(
            current_annotated_frame, car_detections, person_detections
        )

        cv2.imshow(display_window_name, current_annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Quit key 'q' pressed for '{camera_name}'. Stopping surveillance.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSurveillance for '{camera_name}' stopped.")

if __name__ == '__main__':
    run_camera_surveillance(DEFAULT_VIDEO_SOURCE, camera_name="Main Entrance")

