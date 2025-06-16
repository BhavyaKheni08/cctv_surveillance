import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import importlib
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR 

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils.constants as constants
from utils.drawer import Drawer

from detectors.entry_detector import EntryZoneDetector
from detectors.forbidden_area_detector import ForbiddenZoneDetector
from detectors.behavior_detector import HumanBehaviorDetector
from detectors.vandalism_event_detector import VandalismDetector
from detectors.stopping_detector import StoppingDetector

general_object_model = None
pose_estimation_model = None

def load_yolo_models(yolo_general_path, yolo_pose_path):
    global general_object_model, pose_estimation_model
    try:
        if general_object_model is None:
            general_object_model = YOLO(yolo_general_path)
            print(f"Loaded general object detection model: {yolo_general_path}")
        if pose_estimation_model is None:
            pose_estimation_model = YOLO(yolo_pose_path)
            print(f"Loaded pose estimation model: {yolo_pose_path}")
    except Exception as e:
        print(f"Error loading YOLO models: {e}. Make sure paths are correct and models exist.")
        sys.exit(1)

def run_surveillance_scenario(config_module_name):
    try:
        scenario_config = importlib.import_module(config_module_name)
        print(f"Loaded configuration from: {config_module_name}.py")
    except ImportError:
        print(f"Error: Configuration file '{config_module_name}.py' not found.")
        print("Please ensure it's in the correct directory within your project structure.")
        sys.exit(1)

    video_source = scenario_config.DEFAULT_VIDEO_SOURCE
    camera_name = os.path.splitext(os.path.basename(config_module_name))[0].replace("config_", "").replace("_", " ").title() + " Scenario"

    load_yolo_models(constants.YOLO_GENERAL_MODEL_PATH, constants.YOLO_POSE_MODEL_PATH)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}' for '{camera_name}'.")
        print("Please check the path/URL or ensure camera is available. Exiting this scenario.")
        return

    print(f"\n--- Initializing detectors for '{camera_name}' ---")

    violation_event_log = []
    drawer = Drawer()

    forbidden_bike_zone_detector = None
    if hasattr(scenario_config, 'NO_BIKE_ZONES') and scenario_config.NO_BIKE_ZONES:
        forbidden_bike_zone_detector = ForbiddenZoneDetector(
            zones_data=scenario_config.NO_BIKE_ZONES,
            violation_object_classes=constants.BIKE_CLASS_IDS,
            violation_text="BIKE FORBIDDEN!",
            violation_image_prefix="bike_zone_violation"
        )
        print("  Forbidden Bike Zone Detector ENABLED.")
    else:
        print("  Forbidden Bike Zone Detector DISABLED (No NO_BIKE_ZONES in config).")


    forbidden_car_zone_detector = None
    if hasattr(scenario_config, 'NO_PARKING_ZONES') and scenario_config.NO_PARKING_ZONES:
        forbidden_car_zone_detector = ForbiddenZoneDetector(
            zones_data=scenario_config.NO_PARKING_ZONES,
            violation_object_classes=constants.VEHICLE_CLASS_IDS,
            violation_text="VEHICLE NO PARKING!",
            violation_image_prefix="car_zone_violation"
        )
        print("  Forbidden Car Parking Zone Detector ENABLED.")
    else:
        print("  Forbidden Car Parking Zone Detector DISABLED (No NO_PARKING_ZONES in config).")


    entry_zone_detector = None
    if hasattr(scenario_config, 'ENTRY_ZONES') and scenario_config.ENTRY_ZONES:
        entry_zone_detector = EntryZoneDetector(entry_zones_data=scenario_config.ENTRY_ZONES)
        print("  Entry Zone Detector ENABLED.")
    else:
        print("  Entry Zone Detector DISABLED (No ENTRY_ZONES in config).")

    
    human_behavior_detector = HumanBehaviorDetector()
    print("  Human Behavior Detector ENABLED.")
    
    vandalism_detector = VandalismDetector(
        dist_thresh=constants.VANDALISM_DIST_THRESH,
        motion_thresh=constants.VANDALISM_MOTION_THRESH,
        max_age=constants.VANDALISM_TRACKER_MAX_AGE
    )
    print("  Vandalism Detector ENABLED.")

    stopping_detector = None
    if hasattr(scenario_config, 'NO_STOPPING_ZONES') and scenario_config.NO_STOPPING_ZONES:
        stopping_detector = StoppingDetector(
            no_stopping_zones_data=scenario_config.NO_STOPPING_ZONES,
            stop_duration_sec=constants.STOP_DURATION_SECONDS,
            person_proximity_threshold_pixels=constants.PERSON_PROXIMITY_THRESHOLD_PIXELS,
            person_detection_grace_sec=constants.PERSON_GRACE_PERIOD_SECONDS
        )
        print("  Stopping Detector ENABLED.")
    else:
        print("  Stopping Detector DISABLED (No NO_STOPPING_ZONES in config).")

    frame_count = 0
    start_time_fps = time.time()
    fps = 0

    display_window_name = f"{constants.DISPLAY_WINDOW_NAME_PREFIX} - {camera_name}"
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)

    print(f"\nStarting video processing for '{camera_name}'. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
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
        current_time_fps = time.time()
        if (current_time_fps - start_time_fps) > 1:
            fps = frame_count / (current_time_fps - start_time_fps)
            start_time_fps = current_time_fps
            frame_count = 0

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

                if cls_id == constants.PERSON_CLASS_ID:
                    person_detections.append(detection_data)
                elif cls_id in constants.VEHICLE_CLASS_IDS:
                    car_detections.append(detection_data)
                elif cls_id in constants.BIKE_CLASS_IDS:
                    bike_detections_for_forbidden_zone.append(detection_data)

        person_detections_with_keypoints = []
        results_pose = pose_estimation_model(frame, verbose=False)[0]

        if results_pose.boxes is not None and results_pose.keypoints is not None:
            for i, box in enumerate(results_pose.boxes):
                if int(box.cls[0]) == constants.PERSON_CLASS_ID: 
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    kps_xy = results_pose.keypoints.xy.cpu().numpy()[i]
                    kps_conf = results_pose.keypoints.conf.cpu().numpy()[i]
                    
                    formatted_kps = []
                    for j in range(len(kps_xy)):
                        formatted_kps.append([kps_xy[j][0], kps_xy[j][1], kps_conf[j]])

                    matched_id = None
                    person_label = constants.CLASS_NAMES.get(constants.PERSON_CLASS_ID, 'person') 
                    
                    for p_det in person_detections:
                        p_x1, p_y1, p_x2, p_y2 = p_det['box']
                        if box.id is not None and p_det['id'] == int(box.id[0]):
                             matched_id = int(box.id[0])
                             person_label = p_det['label'] 
                             break
                    if matched_id is None:
                        matched_id = i

                    person_detections_with_keypoints.append({
                        'box': (x1, y1, x2, y2),
                        'keypoints': formatted_kps,
                        'id': matched_id,
                        'label': person_label 
                    })

        current_annotated_frame = frame.copy()
        zones_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Corrected fillPoly arguments: convert zone_coords to numpy array
        if forbidden_bike_zone_detector and hasattr(scenario_config, 'NO_BIKE_ZONES') and scenario_config.NO_BIKE_ZONES:
            for zone_coords in scenario_config.NO_BIKE_ZONES:
                polygon = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zones_overlay, [polygon], constants.ANNOTATION_COLOR_ZONE_FILL)
        if forbidden_car_zone_detector and hasattr(scenario_config, 'NO_PARKING_ZONES') and scenario_config.NO_PARKING_ZONES:
            for zone_coords in scenario_config.NO_PARKING_ZONES:
                polygon = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zones_overlay, [polygon], constants.ANNOTATION_COLOR_ZONE_FILL)

        if entry_zone_detector and hasattr(scenario_config, 'ENTRY_ZONES') and scenario_config.ENTRY_ZONES:
            for zone_coords in scenario_config.ENTRY_ZONES:
                polygon = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zones_overlay, [polygon], constants.ANNOTATION_COLOR_ZONE_BOUNDARY)

        if stopping_detector and hasattr(scenario_config, 'NO_STOPPING_ZONES') and scenario_config.NO_STOPPING_ZONES:
            for zone_coords in scenario_config.NO_STOPPING_ZONES:
                polygon = np.array(zone_coords, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(zones_overlay, [polygon], (255, 0, 0)) # Blue fill

        current_annotated_frame = cv2.addWeighted(current_annotated_frame, 1, zones_overlay, 0.08, 0)

        bike_violation_detected = False
        if forbidden_bike_zone_detector:
            current_annotated_frame, bike_violation_detected = forbidden_bike_zone_detector.process_frame(
                current_annotated_frame, bike_detections_for_forbidden_zone, frame_count
            )

        car_violation_detected = False
        if forbidden_car_zone_detector:
            current_annotated_frame, car_violation_detected = forbidden_car_zone_detector.process_frame(
                current_annotated_frame, car_detections, frame_count
            )

        entry_violation_detected = False
        if entry_zone_detector:
            current_annotated_frame, entry_violation_detected = entry_zone_detector.process_frame(
                current_annotated_frame, all_detections_info
            )

        stable_fight_alert = False
        if human_behavior_detector:
            current_annotated_frame, stable_fight_alert = human_behavior_detector.process_frame(
                current_annotated_frame, person_detections_with_keypoints, frame_width, frame_height
            )

        vandalism_detected = False
        if vandalism_detector:
            current_annotated_frame, vandalism_detected = vandalism_detector.process_frame(
                current_annotated_frame, person_detections, car_detections
            )
        
        stopping_violation_detected = False
        if stopping_detector:
            current_annotated_frame, stopping_violation_detected = stopping_detector.process_frame(
                current_annotated_frame, car_detections, person_detections_with_keypoints
            )

        drawer.draw_fps(current_annotated_frame, fps)

        if current_annotated_frame is not None:
            cv2.imshow(display_window_name, current_annotated_frame)
        else:
            print(f"Warning: current_annotated_frame is None. Skipping imshow for frame {frame_count}.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Quit key 'q' pressed for '{camera_name}'. Stopping surveillance.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSurveillance for '{camera_name}' stopped.")

    print("\n--- VIOLATION EVENT LOG ---")
    if violation_event_log:
        for i, event in enumerate(violation_event_log):
            print(f"{i+1}. [{event['timestamp']}] Detector: {event['detector']}, Details: {event['details']}, Frame: {event['frame']}")
    else:
        print("No violations detected during this session.")

if __name__ == '__main__':
    CONFIG_FILES_DIRECTORY = PROJECT_ROOT
    
    try:
        os.chdir(CONFIG_FILES_DIRECTORY)
        print(f"Changed current working directory to: {os.getcwd()}")
    except OSError as e:
        print(f"Error changing directory to {CONFIG_FILES_DIRECTORY}: {e}")
        print("Please verify the path. Exiting.")
        sys.exit(1)

    print("Available scenario config files:")
    config_files_available = [f for f in os.listdir('.') if f.startswith('config_') and f.endswith('.py')]
    
    if not config_files_available:
        print("No 'config_*.py' files found in the current directory.")
        print(f"Please ensure your config files (e.g., config_entry.py) are in: {CONFIG_FILES_DIRECTORY}")
        sys.exit(1)

    config_files_available.sort()

    for i, file_name in enumerate(config_files_available):
        display_name = file_name.replace('.py', '')
        print(f"[{i+1}] {display_name}")

    selected_index = -1
    while selected_index < 0 or selected_index >= len(config_files_available):
        try:
            choice = input("Enter the number of the scenario you want to run: ")
            selected_index = int(choice) - 1
            if selected_index < 0 or selected_index >= len(config_files_available):
                print("Invalid choice. Please enter a valid number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    chosen_config_file_name = config_files_available[selected_index]
    chosen_config_module_name = chosen_config_file_name.replace('.py', '')

    run_surveillance_scenario(chosen_config_module_name)
