import cv2
import numpy as np
import math
import utils.constants as constants

def is_point_in_zone(point, zone):
    if zone is None or len(zone) == 0:
        return False
    return cv2.pointPolygonTest(zone, point, False) >= 0

def calculate_keypoint_distance(kp1, kp2):
    if kp1 is None or kp2 is None:
        return 0
    p1_coords = kp1[:2] if isinstance(kp1, (list, tuple)) and len(kp1) > 1 else kp1
    p2_coords = kp2[:2] if isinstance(kp2, (list, tuple)) and len(kp2) > 1 else kp2
    return np.linalg.norm(np.array(p1_coords) - np.array(p2_coords))

def get_midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    p1_coords = p1[:2] if isinstance(p1, (list, tuple)) and len(p1) > 1 else p1
    p2_coords = p2[:2] if isinstance(p2, (list, tuple)) and len(p2) > 1 else p2
    return ((p1_coords[0] + p2_coords[0]) // 2, (p1_coords[1] + p2_coords[1]) // 2)

def visible(kp_data):
    return kp_data is not None and len(kp_data) > 2 and kp_data[2] > constants.CONF_THRESHOLD

def get_vector(p1_coords, p2_coords):
    if p1_coords is None or p2_coords is None:
        return None
    return np.array([p2_coords[0] - p1_coords[0], p2_coords[1] - p1_coords[1]])

def get_angle(v1, v2):
    if v1 is None or v2 is None:
        return None
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = math.acos(cosine_angle)
    return math.degrees(angle_rad)

def get_person_height(keypoints_full):
    if not visible(keypoints_full[constants.NOSE]) or not visible(keypoints_full[constants.LEFT_ANKLE]) or not visible(keypoints_full[constants.RIGHT_ANKLE]):
        return None
    
    lowest_ankle_y = max(keypoints_full[constants.LEFT_ANKLE][1], keypoints_full[constants.RIGHT_ANKLE][1])
    return lowest_ankle_y - keypoints_full[constants.NOSE][1]

def calculate_centroid(keypoints_full):
    visible_kps = [kp for kp in keypoints_full if visible(kp)]
    if not visible_kps:
        return None
    x_sum = sum(kp[0] for kp in visible_kps)
    y_sum = sum(kp[1] for kp in visible_kps)
    return (x_sum / len(visible_kps), y_sum / len(visible_kps))

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in (x1, y1, x2, y2) format.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If no intersection, IoU is 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # The union area is the sum of both areas minus the intersection area
    union_area = float(box1_area + box2_area - intersection_area)

    # Handle case where union_area is zero to avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area
        return 0.0
    
    dist_pixels = calculate_keypoint_distance(prev_kps[kp_idx], curr_kps[kp_idx])
    return dist_pixels / frame_dimension if frame_dimension > 0 else 0.0
