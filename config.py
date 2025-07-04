import os


YOLO_GENERAL_MODEL_PATH = "/Users/bhavyakheni/Desktop/cctv_surveillance/models/yolo11l.pt"
YOLO_POSE_MODEL_PATH = "/Users/bhavyakheni/Desktop/cctv_surveillance/models/yolov8l-pose.pt"

DEFAULT_VIDEO_SOURCE = "/Users/bhavyakheni/Desktop/cctv_surveillance/VIdoes/No_Parking.mov"

CLASS_NAMES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}

PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = [2, 5, 7]
BIKE_CLASS_IDS = [1, 3]

FORBIDDEN_ZONE_COOLDOWN_SECONDS = 5
VIOLATION_IMAGE_SAVE_DIR = "violations"
if not os.path.exists(VIOLATION_IMAGE_SAVE_DIR):
    os.makedirs(VIOLATION_IMAGE_SAVE_DIR)

VANDALISM_DIST_THRESH = 100
VANDALISM_MOTION_THRESH = 1500
VANDALISM_TRACKER_MAX_AGE = 30

STOP_DURATION_SECONDS = 3
PERSON_PROXIMITY_THRESHOLD_PIXELS = 100
PERSON_GRACE_PERIOD_SECONDS = 3.0

DISPLAY_WINDOW_NAME_PREFIX = "Surveillance Feed"
ANNOTATION_COLOR_VIOLATION = (0, 0, 255)
ANNOTATION_COLOR_WARNING = (0, 165, 255)
ANNOTATION_COLOR_NORMAL = (0, 255, 0)
ANNOTATION_COLOR_ZONE_BOUNDARY = (255, 255, 0)
ANNOTATION_COLOR_ZONE_FILL = (255, 0, 0)

NO_BIKE_ZONES = [
    [[100, 100], [300, 100], [300, 300], [100, 300]],
]

NO_PARKING_ZONES = [
    [[400, 600], [800, 600], [800, 700], [400, 700]],
]

ENTRY_ZONES = [
    [[900, 50], [1200, 50], [1200, 200], [900, 200]],
]

NO_STOPPING_ZONES = [
    [[50, 300], [250, 300], [250, 450], [50, 450]],
]
