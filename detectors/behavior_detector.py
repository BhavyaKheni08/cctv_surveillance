import cv2
import numpy as np
from collections import deque
from utils.helpers import (
    calculate_keypoint_distance, visible, get_midpoint,
    get_vector, get_angle, get_person_height, calculate_centroid
) # Removed calculate_velocity as it's not directly causing the IndexError here, but logic needs to be mindful
from utils.drawer import Drawer
import utils.constants as constants

class HumanBehaviorDetector:
    def __init__(self):
        self.PROXIMITY_WRIST_HEAD_RATIO = 0.04
        self.PROXIMITY_LIMB_TORSO_RATIO = 0.06
        self.PROXIMITY_HIPS_RATIO = 0.10
        self.PROXIMITY_KNEES_ANKLES_RATIO = 0.08
        self.ARM_RAISED_ANGLE_THRESHOLD = 25
        self.ELBOW_PUNCH_BEND_ANGLE_MIN = 60
        self.ELBOW_PUNCH_BEND_ANGLE_MAX = 100
        self.KNEE_KICK_ANGLE_THRESHOLD = 90
        self.MIN_FIGHT_INDICATORS = constants.HUMAN_FIGHT_DETECTION_THRESHOLD
        self.FIGHT_DETECTION_WINDOW = constants.HUMAN_FIGHT_HISTORY_SIZE
        self.VELOCITY_THRESHOLD_FIGHT = constants.HUMAN_VELOCITY_THRESHOLD_FIGHT

        self.FALL_HEAD_TO_HIP_Y_RATIO = constants.HUMAN_FALL_HEAD_TO_HIP_Y_RATIO
        self.FALL_BODY_ALIGNMENT_X_DEVIATION = constants.HUMAN_FALL_BODY_ALIGNMENT_X_DEVIATION
        self.FALL_VERTICAL_DROP_RATIO = constants.HUMAN_FALL_VERTICAL_DROP_RATIO
        self.FALL_DETECTION_HISTORY_FRAMES = constants.HUMAN_FALL_DETECTION_HISTORY_FRAMES
        self.FALL_STILLNESS_THRESHOLD = constants.HUMAN_FALL_STILLNESS_THRESHOLD
        self.MIN_FALL_INDICATORS = constants.HUMAN_MIN_FALL_INDICATORS

        self.person_pose_histories = {} 
        self.fight_buffer = deque(maxlen=self.FIGHT_DETECTION_WINDOW)
        self.drawer = Drawer()

    def _analyze_pose_for_fight(self, kp1_full, kp2_full, frame_width, frame_height, history1: deque, history2: deque):
        if kp1_full is None or kp2_full is None:
            return False

        indicators = 0

        # Helper to check if two keypoints are close based on a ratio of frame width
        # Ensure that this is only used with full (x,y,conf) keypoints
        def is_close_full_kp(p1_full, p2_full, ratio_factor):
            if not visible(p1_full) or not visible(p2_full):
                return False
            return calculate_keypoint_distance(p1_full, p2_full) < frame_width * ratio_factor

        target_kps_head = [constants.NOSE, constants.LEFT_EYE, constants.RIGHT_EYE, constants.LEFT_EAR, constants.RIGHT_EAR]
        target_kps_upper_body = [constants.LEFT_SHOULDER, constants.RIGHT_SHOULDER, constants.LEFT_HIP, constants.RIGHT_HIP]
        
        for current_kp_full, other_kp_full, current_history, other_history in [(kp1_full, kp2_full, history1, history2), (kp2_full, kp1_full, history2, history1)]:
            for shoulder_idx, elbow_idx, wrist_idx in [(constants.LEFT_SHOULDER, constants.LEFT_ELBOW, constants.LEFT_WRIST), (constants.RIGHT_SHOULDER, constants.RIGHT_ELBOW, constants.RIGHT_WRIST)]:
                if all(visible(current_kp_full[idx]) for idx in [shoulder_idx, elbow_idx, wrist_idx] if idx < len(current_kp_full)):
                    arm_vec = get_vector(current_kp_full[shoulder_idx], current_kp_full[elbow_idx])
                    forearm_vec = get_vector(current_kp_full[elbow_idx], current_kp_full[wrist_idx])
                    
                    if arm_vec is not None and forearm_vec is not None:
                        elbow_angle = get_angle(arm_vec, forearm_vec)
                        vertical_vec = np.array([0, -1])

                        if get_angle(arm_vec, vertical_vec) is not None and get_angle(arm_vec, vertical_vec) < self.ARM_RAISED_ANGLE_THRESHOLD and \
                           elbow_angle is not None and self.ELBOW_PUNCH_BEND_ANGLE_MIN < elbow_angle < self.ELBOW_PUNCH_BEND_ANGLE_MAX:
                            if current_kp_full[wrist_idx][1] < current_kp_full[shoulder_idx][1]:
                                for target_idx in target_kps_head + target_kps_upper_body:
                                    if target_idx < len(other_kp_full) and is_close_full_kp(current_kp_full[wrist_idx], other_kp_full[target_idx], 
                                                           self.PROXIMITY_WRIST_HEAD_RATIO if target_idx in target_kps_head else self.PROXIMITY_LIMB_TORSO_RATIO):
                                        indicators += 1.5

        target_kps_lower_body = [constants.LEFT_HIP, constants.RIGHT_HIP, constants.LEFT_KNEE, constants.RIGHT_KNEE, constants.LEFT_ANKLE, constants.RIGHT_ANKLE]
        
        for current_kp_full, other_kp_full in [(kp1_full, kp2_full), (kp2_full, kp1_full)]:
            for hip_idx, knee_idx, ankle_idx in [(constants.LEFT_HIP, constants.LEFT_KNEE, constants.LEFT_ANKLE), (constants.RIGHT_HIP, constants.RIGHT_KNEE, constants.RIGHT_ANKLE)]:
                if all(visible(current_kp_full[idx]) for idx in [hip_idx, knee_idx, ankle_idx] if idx < len(current_kp_full)):
                    hip_knee_vec = get_vector(current_kp_full[hip_idx], current_kp_full[knee_idx])
                    knee_ankle_vec = get_vector(current_kp_full[knee_idx], current_kp_full[ankle_idx])
                    if hip_knee_vec is not None and knee_ankle_vec is not None:
                        leg_angle = get_angle(hip_knee_vec, knee_ankle_vec)
                        if leg_angle is not None and leg_angle < self.KNEE_KICK_ANGLE_THRESHOLD:
                            for target_idx in target_kps_lower_body + target_kps_upper_body:
                                if target_idx < len(other_kp_full) and is_close_full_kp(current_kp_full[ankle_idx], other_kp_full[target_idx], self.PROXIMITY_KNEES_ANKLES_RATIO):
                                    indicators += 1.0

        mid1_hips_coords = get_midpoint(kp1_full[constants.LEFT_HIP], kp1_full[constants.RIGHT_HIP])
        mid2_hips_coords = get_midpoint(kp2_full[constants.LEFT_HIP], kp2_full[constants.RIGHT_HIP])
        if mid1_hips_coords and mid2_hips_coords and calculate_keypoint_distance(mid1_hips_coords, mid2_hips_coords) < frame_width * self.PROXIMITY_HIPS_RATIO:
            indicators += 0.5

        mid1_shoulders_coords = get_midpoint(kp1_full[constants.LEFT_SHOULDER], kp1_full[constants.RIGHT_SHOULDER])
        mid2_shoulders_coords = get_midpoint(kp2_full[constants.LEFT_SHOULDER], kp2_full[constants.RIGHT_SHOULDER])
        
        if mid1_shoulders_coords and mid2_shoulders_coords:
            p1_center_coords = calculate_centroid(kp1_full)
            p2_center_coords = calculate_centroid(kp2_full)
            
            if p1_center_coords and p2_center_coords:
                vec_p1_shoulders_to_p2_center = get_vector(mid1_shoulders_coords, p2_center_coords)
                p1_forward_vec = get_vector(mid1_hips_coords, mid1_shoulders_coords) if mid1_hips_coords else None

                vec_p2_shoulders_to_p1_center = get_vector(mid2_shoulders_coords, p1_center_coords)
                p2_forward_vec = get_vector(mid2_hips_coords, mid2_shoulders_coords) if mid2_hips_coords else None

                if p1_forward_vec is not None and vec_p1_shoulders_to_p2_center is not None:
                    angle_p1_facing_p2 = get_angle(p1_forward_vec, vec_p1_shoulders_to_p2_center)
                    if angle_p1_facing_p2 is not None and angle_p1_facing_p2 < 30:
                        indicators += 0.5

                if p2_forward_vec is not None and vec_p2_shoulders_to_p1_center is not None:
                    angle_p2_facing_p1 = get_angle(p2_forward_vec, vec_p2_shoulders_to_p1_center)
                    if angle_p2_facing_p1 is not None and angle_p2_facing_p1 < 30:
                        indicators += 0.5

        # IMPORTANT: calculate_velocity from utils/helpers.py expects full keypoint data (x, y, conf)
        # Ensure that history elements are also full keypoint data.
        if len(history1) >= 2 and len(history2) >= 2:
            prev_kp1_full = history1[-2]
            prev_kp2_full = history2[-2]

            for current_kp_iter_full, prev_kp_iter_full, other_kp_full in [(kp1_full, prev_kp1_full, kp2_full), (kp2_full, prev_kp2_full, kp1_full)]:
                for limb_idx in [constants.LEFT_WRIST, constants.RIGHT_WRIST, constants.LEFT_ANKLE, constants.RIGHT_ANKLE]:
                    # Check visibility before calculating velocity
                    if visible(current_kp_iter_full[limb_idx]) and visible(prev_kp_iter_full[limb_idx]):
                        velocity_limb = calculate_keypoint_distance(prev_kp_iter_full[limb_idx], current_kp_iter_full[limb_idx]) / frame_width
                        if velocity_limb > self.VELOCITY_THRESHOLD_FIGHT:
                            # Use (x,y) coords for vectors if get_vector takes them
                            limb_movement_vec = get_vector(prev_kp_iter_full[limb_idx][:2], current_kp_iter_full[limb_idx][:2])
                            
                            other_person_target_center_coords = get_midpoint(get_midpoint(other_kp_full[constants.LEFT_SHOULDER], other_kp_full[constants.RIGHT_SHOULDER]), other_kp_full[constants.NOSE])
                            if other_person_target_center_coords is None: other_person_target_center_coords = other_kp_full[constants.NOSE][:2] # Fallback to nose coords
                            
                            if other_person_target_center_coords and limb_movement_vec is not None:
                                limb_to_target_vec = get_vector(current_kp_iter_full[limb_idx][:2], other_person_target_center_coords)
                                if limb_to_target_vec is not None:
                                    angle_to_target = get_angle(limb_movement_vec, limb_to_target_vec)
                                    if angle_to_target is not None and angle_to_target < 20: 
                                        indicators += 1.0

        return indicators >= self.MIN_FIGHT_INDICATORS

    def _detect_fall(self, current_kp_full, frame_width, frame_height, pose_history: deque):
        if current_kp_full is None:
            return False

        pose_history.append(current_kp_full) 

        if len(pose_history) < 2:
            return False

        fall_indicators = 0

        # These points are (x, y) tuples from get_midpoint, so we check for None directly
        current_head_coords = get_midpoint(current_kp_full[constants.NOSE], get_midpoint(current_kp_full[constants.LEFT_EAR], current_kp_full[constants.RIGHT_EAR]))
        current_neck_coords = get_midpoint(current_kp_full[constants.LEFT_SHOULDER], current_kp_full[constants.RIGHT_SHOULDER])
        current_hips_coords = get_midpoint(current_kp_full[constants.LEFT_HIP], current_kp_full[constants.RIGHT_HIP])
        current_torso_center_coords = get_midpoint(current_neck_coords, current_hips_coords)

        # 1. Head below hips (or close to) - indicates horizontal posture
        # Check for None for the midpoint coordinates before using them
        if current_head_coords and current_hips_coords and \
           current_head_coords[1] > current_hips_coords[1] - (get_person_height(current_kp_full) * self.FALL_HEAD_TO_HIP_Y_RATIO if get_person_height(current_kp_full) else 0):
            fall_indicators += 1

        # 2. Body alignment (torso vector significantly off vertical)
        # Check for None for the coordinates before calling get_vector
        if current_neck_coords and current_hips_coords:
            body_vec = get_vector(current_neck_coords, current_hips_coords) # These are (x,y) tuples
            if body_vec is not None:
                vertical_orientation = np.array([0, 1])
                angle = get_angle(body_vec, vertical_orientation)
                if angle is not None and angle > 90 - (90 * self.FALL_BODY_ALIGNMENT_X_DEVIATION):
                    fall_indicators += 1

        # 3. Sudden significant vertical drop (requires history)
        if len(pose_history) >= self.FALL_DETECTION_HISTORY_FRAMES:
            oldest_kp_full = pose_history[0]
            oldest_torso_center_coords = get_midpoint(get_midpoint(oldest_kp_full[constants.LEFT_SHOULDER], oldest_kp_full[constants.RIGHT_SHOULDER]),
                                               get_midpoint(oldest_kp_full[constants.LEFT_HIP], oldest_kp_full[constants.RIGHT_HIP]))

            # Check for None for the coordinates before using them
            if current_torso_center_coords and oldest_torso_center_coords:
                vertical_drop = current_torso_center_coords[1] - oldest_torso_center_coords[1]
                person_height = get_person_height(current_kp_full)

                if person_height and vertical_drop > person_height * self.FALL_VERTICAL_DROP_RATIO:
                    fall_indicators += 1

        # 4. Stillness after potential fall (requires full history)
        if fall_indicators >= 1 and len(pose_history) >= self.FALL_DETECTION_HISTORY_FRAMES:
            is_still = True
            for i in range(1, self.FALL_DETECTION_HISTORY_FRAMES):
                prev_kp_hist_full = pose_history[i-1]
                curr_kp_hist_full = pose_history[i]
                
                prev_centroid_coords = calculate_centroid(prev_kp_hist_full)
                curr_centroid_coords = calculate_centroid(curr_kp_hist_full)

                # Check for None for centroids before calculating distance
                if prev_centroid_coords is None or curr_centroid_coords is None:
                    is_still = False
                    break
                
                displacement = calculate_keypoint_distance(prev_centroid_coords, curr_centroid_coords) / frame_width 
                if displacement > self.FALL_STILLNESS_THRESHOLD:
                    is_still = False
                    break
            
            if is_still:
                fall_indicators += 1

        return fall_indicators >= self.MIN_FALL_INDICATORS

    def process_frame(self, frame, person_detections_with_keypoints, frame_width, frame_height):
        annotated_frame = frame.copy()
        
        fight_detected_this_frame = False
        
        active_person_ids = set()
        for p_data in person_detections_with_keypoints:
            p_id = p_data.get('id')
            if p_id is None: 
                p_id = str(p_data['box']) 
            active_person_ids.add(p_id)

            if p_id not in self.person_pose_histories:
                self.person_pose_histories[p_id] = deque(maxlen=self.FALL_DETECTION_HISTORY_FRAMES)
            self.person_pose_histories[p_id].append(p_data['keypoints']) # Append full keypoint data

        inactive_ids = [pid for pid in self.person_pose_histories if pid not in active_person_ids]
        for pid in inactive_ids:
            del self.person_pose_histories[pid]

        person_ids = list(self.person_pose_histories.keys())
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                p1_id = person_ids[i]
                p2_id = person_ids[j]

                p1_data = next((p for p in person_detections_with_keypoints if p.get('id') == p1_id or (p.get('id') is None and str(p['box']) == p1_id)), None)
                p2_data = next((p for p in person_detections_with_keypoints if p.get('id') == p2_id or (p.get('id') is None and str(p['box']) == p2_id)), None)

                if p1_data is None or p2_data is None:
                    continue

                p1_kps_history = self.person_pose_histories[p1_id]
                p2_kps_history = self.person_pose_histories[p2_id]

                center1 = ((p1_data['box'][0] + p1_data['box'][2]) // 2, (p1_data['box'][1] + p1_data['box'][3]) // 2)
                center2 = ((p2_data['box'][0] + p2_data['box'][2]) // 2, (p2_data['box'][1] + p2_data['box'][3]) // 2)
                dist_centers = calculate_keypoint_distance(center1, center2)

                if dist_centers < frame_width * 0.4: 
                    if self._analyze_pose_for_fight(p1_data['keypoints'], p2_data['keypoints'], frame_width, frame_height,
                                                   p1_kps_history, p2_kps_history):
                        fight_detected_this_frame = True
                        self.drawer.draw_bbox(annotated_frame, p1_data['box'], constants.ANNOTATION_COLOR_VIOLATION)
                        self.drawer.draw_bbox(annotated_frame, p2_data['box'], constants.ANNOTATION_COLOR_VIOLATION)
                        self.drawer.draw_text(annotated_frame, "FIGHT",
                                    (min(p1_data['box'][0], p2_data['box'][0]), min(p1_data['box'][1], p2_data['box'][1])),
                                    constants.ANNOTATION_COLOR_VIOLATION)

        self.fight_buffer.append(fight_detected_this_frame)
        stable_fight_alert = sum(self.fight_buffer) > self.FIGHT_DETECTION_WINDOW * 0.6
        if stable_fight_alert:
            self.drawer.draw_text_with_background(annotated_frame, "Stable Fight Alert!", (20, 30), constants.ANNOTATION_COLOR_VIOLATION)


        for person_data in person_detections_with_keypoints:
            person_id = person_data.get('id')
            if person_id is None: person_id = str(person_data['box']) 
            kp_full = person_data['keypoints']

            if person_id not in self.person_pose_histories:
                self.person_pose_histories[person_id] = deque(maxlen=self.FALL_DETECTION_HISTORY_FRAMES)
                self.person_pose_histories[person_id].append(kp_full)

            fall_detected = self._detect_fall(kp_full, frame_width, frame_height, self.person_pose_histories[person_id])
            
            if fall_detected:
                self.drawer.draw_bbox(annotated_frame, person_data['box'], constants.ANNOTATION_COLOR_WARNING) 
                self.drawer.draw_text(annotated_frame, "FALL DETECTED", (person_data['box'][0], person_data['box'][1]), constants.ANNOTATION_COLOR_WARNING)

        for person_data in person_detections_with_keypoints:
            kps_full = person_data['keypoints']
            if kps_full:
                self.drawer.draw_keypoints(annotated_frame, kps_full, color=(0, 255, 255))

        return annotated_frame, stable_fight_alert
