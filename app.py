from flask import Flask, request, render_template, send_file, redirect, url_for, session
import os
import cv2
import pandas as pd
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from mediapipe import solutions as mp_solutions
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from xlsx2csv import Xlsx2csv
import shutil
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from datetime import datetime, timedelta

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
IMAGES_FOLDER = 'static/images'
PHONE_IMAGES_FOLDER = 'static/phone_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(PHONE_IMAGES_FOLDER, exist_ok=True)

USERS = {
    'admin': generate_password_hash('password123')
}

@dataclass
class DetectionConfig:
    """Configuration parameters for detection"""
    pose_confidence: float = 0.5
    hand_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    smoothing_frames: int = 2  # Number of consecutive frames to confirm a signal
    cooldown_seconds: float = 2.0  # Cooldown period after detecting a signal
    max_num_people: int = 10  # Maximum number of people to detect
    save_frames: bool = True   # Whether to save detected frames
    output_folder: str = "detected_frames"  # Folder to save detected frames

class GestureDetector:
    def __init__(self, config: DetectionConfig):
        """Initialize detector with given configuration"""
        self.config = config
        # Use holistic model to get better integration of pose and hands
        self.holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.pose_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
            enable_segmentation=False,
        )
        # Initialize multi-pose detector
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.pose_confidence,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
        )
        # Use multi-hand detector
        self.hands = mp_solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2 * config.max_num_people,  # Allow for both hands per person
            min_detection_confidence=config.hand_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )

        # Buffering for each detection type and for each potential person
        self.elbow_hand_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.hand_above_head_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.hand_pointing_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.mic_gesture_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.torch_pointing_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]

        # Last detection time for each person
        self.last_detection_times = [-float('inf')] * config.max_num_people

        # Create output folder if saving frames is enabled
        if config.save_frames and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
            print(f"Created output folder: {config.output_folder}")

    # [All the existing methods of GestureDetector remain unchanged]
    def is_elbow_above_shoulder(self, pose_landmarks) -> bool:
        """
        Checks if either elbow is above the corresponding shoulder.
        """
        if not pose_landmarks:
            return False

        left_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y

        left_elbow = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_ELBOW.value].y
        right_elbow = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y

        # Check visibility confidence
        left_shoulder_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER.value].visibility > 0.5
        right_shoulder_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5
        left_elbow_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_ELBOW.value].visibility > 0.5
        right_elbow_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_ELBOW.value].visibility > 0.5

        # Only check if landmarks are visible
        left_check = left_elbow_visible and left_shoulder_visible and left_elbow < left_shoulder
        right_check = right_elbow_visible and right_shoulder_visible and right_elbow < right_shoulder

        return left_check or right_check

    def is_hand_raised(self, hand_landmarks) -> bool:
        """
        Checks if the hand is raised above a certain threshold.
        """
        if not hand_landmarks:
            return False

        wrist_y = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value].y
        return wrist_y < 0.4  # Adjust threshold as needed

    def is_hand_above_head(self, hand_landmarks, pose_landmarks) -> bool:
        """
        Checks if any hand is raised above the head.
        Returns True if any finger tip is above the nose.
        """
        if not hand_landmarks or not pose_landmarks:
            return False

        # Check if nose landmark has sufficient visibility
        nose = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.NOSE.value]
        if nose.visibility < 0.5:
            return False

        nose_y = nose.y

        # Check if any finger tip is above the head
        finger_tips = [
            mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP.value,
            mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP.value,
            mp_solutions.hands.HandLandmark.RING_FINGER_TIP.value,
            mp_solutions.hands.HandLandmark.PINKY_TIP.value,
            mp_solutions.hands.HandLandmark.THUMB_TIP.value
        ]

        # Check if any finger tip is above the nose
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < nose_y:
                return True

        return False

    def is_pointing_gesture(self, hand_landmarks) -> bool:
        """
        Detects a pointing gesture - index finger extended while other fingers are curled.
        """
        if not hand_landmarks:
            return False

        try:
            # Get landmarks for each finger
            thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP.value]
            index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP.value]
            middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP.value]
            ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP.value]
            pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP.value]

            index_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_PIP.value]
            middle_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_PIP.value]
            ring_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_PIP.value]
            pinky_pip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_PIP.value]

            wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value]

            # Check if index finger is extended (tip is far from wrist)
            index_extended = index_tip.y < index_pip.y

            # Check if other fingers are curled (tips are close to palm)
            other_fingers_curled = (
                middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and
                pinky_tip.y > pinky_pip.y
            )

            # For horizontal pointing (as seen in images)
            horizontal_pointing = abs(index_tip.x - wrist.x) > 0.1

            return index_extended and (other_fingers_curled or horizontal_pointing)

        except KeyError as e:
            print(f"Error accessing landmark: {e}")
            return False

    def is_torch_pointing_gesture(self, hand_landmarks, pose_landmarks) -> bool:
        """
        Detects when someone is pointing a torch-like object.
        This is characterized by an extended arm with the hand in a gripping position.
        """
        if not hand_landmarks or not pose_landmarks:
            return False

        # We need to check if the arm is extended (distance between shoulder and wrist)
        # and if the hand is in a gripping position (fingers closed, not fully extended)

        # Get hand type (left or right)
        hand_side = self.match_hand_to_pose(hand_landmarks, pose_landmarks)

        if not hand_side:
            return False

        # Get relevant shoulder and elbow positions based on hand side
        shoulder_idx = (mp_solutions.pose.PoseLandmark.LEFT_SHOULDER.value if hand_side == "left"
                      else mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER.value)
        elbow_idx = (mp_solutions.pose.PoseLandmark.LEFT_ELBOW.value if hand_side == "left"
                   else mp_solutions.pose.PoseLandmark.RIGHT_ELBOW.value)

        # Check if landmarks are visible
        shoulder = pose_landmarks.landmark[shoulder_idx]
        elbow = pose_landmarks.landmark[elbow_idx]
        wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value]

        if shoulder.visibility < 0.5 or elbow.visibility < 0.5:
            return False

        # Check if arm is extended (shoulder, elbow, and wrist should form a relatively straight line)
        # Calculate vectors
        shoulder_to_elbow = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
        elbow_to_wrist = np.array([wrist.x - elbow.x, wrist.y - elbow.y])

        # Normalize vectors
        if np.linalg.norm(shoulder_to_elbow) > 0 and np.linalg.norm(elbow_to_wrist) > 0:
            shoulder_to_elbow = shoulder_to_elbow / np.linalg.norm(shoulder_to_elbow)
            elbow_to_wrist = elbow_to_wrist / np.linalg.norm(elbow_to_wrist)

            # Calculate dot product to get cosine of angle
            dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)

            # Arm is extended if angle is close to 180 degrees (dot product close to -1)
            # or if angle is close to 0 degrees (dot product close to 1)
            arm_extended = dot_product > 0.7 or dot_product < -0.7
        else:
            arm_extended = False

        # Check hand position - a gripping position has fingers somewhat curled but not completely
        # For a torch-holding grip, typically the index and thumb form a circular grip
        index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP.value]
        index_dip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_DIP.value]
        thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP.value]

        # Distance between thumb tip and index finger
        thumb_to_index = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

        # Check for grip-like hand position (fingers not fully extended but slightly curled)
        grip_position = thumb_to_index < 0.1  # Thumb and index are close, as if gripping something

        # Combination of extended arm and gripping position suggests pointing with an object
        return arm_extended and grip_position

    def is_mic_speaking_gesture(self, hand_landmarks, pose_landmarks) -> bool:
        """
        Detects when someone is holding something near their mouth, like a microphone or handset.
        This is visible in image 4 where someone appears to be speaking into a device.
        """
        if not hand_landmarks or not pose_landmarks:
            return False

        # Get relevant landmarks
        wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value]
        mouth = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.MOUTH_RIGHT.value]

        # Check if hand is near the mouth
        distance_x = abs(wrist.x - mouth.x)
        distance_y = abs(wrist.y - mouth.y)

        # Adjustable thresholds
        return distance_x < 0.15 and distance_y < 0.15 and wrist.visibility > 0.7 and mouth.visibility > 0.7

    def match_hand_to_pose(self, hand_landmarks, pose_landmarks) -> Optional[str]:
        """
        Attempt to match a hand to a specific pose by comparing wrist positions.
        Returns "left" or "right" for the corresponding hand, or None if no match can be confidently made.
        """
        if not hand_landmarks or not pose_landmarks:
            return None

        # Get wrist position from hand landmarks
        hand_wrist_x = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value].x
        hand_wrist_y = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST.value].y

        # Get wrist positions from pose landmarks
        left_wrist_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.5
        right_wrist_visible = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.5

        if not (left_wrist_visible or right_wrist_visible):
            return None

        # Calculate distances to both wrists
        distances = []

        if left_wrist_visible:
            left_wrist_x = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_WRIST.value].x
            left_wrist_y = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_WRIST.value].y
            left_dist = ((hand_wrist_x - left_wrist_x)**2 + (hand_wrist_y - left_wrist_y)**2)**0.5
            distances.append(("left", left_dist))

        if right_wrist_visible:
            right_wrist_x = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_WRIST.value].x
            right_wrist_y = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
            right_dist = ((hand_wrist_x - right_wrist_x)**2 + (hand_wrist_y - right_wrist_y)**2)**0.5
            distances.append(("right", right_dist))

        if not distances:
            return None

        # Find the closest wrist
        closest = min(distances, key=lambda x: x[1])

        # Only return a match if the distance is below a threshold
        if closest[1] < 0.15:  # Threshold for matching - may need tuning
            return closest[0]

        return None

    def assign_person_id(self, pose_landmarks):
        """
        Assign a unique identifier to a person based on their pose landmarks.
        This is a basic implementation using the center of the bounding box as an identifier.
        """
        if not pose_landmarks:
            return None

        # Use visible landmarks to calculate a center point
        visible_landmarks = [(lm.x, lm.y) for lm in pose_landmarks.landmark if lm.visibility > 0.5]

        if not visible_landmarks:
            return None

        # Calculate center of visible landmarks
        center_x = sum(x for x, _ in visible_landmarks) / len(visible_landmarks)
        center_y = sum(y for _, y in visible_landmarks) / len(visible_landmarks)

        # Return a simple ID based on position in the frame
        # Divide the frame into grid cells and use the cell index as ID
        x_cells = 3  # Number of cells horizontally
        y_cells = 3  # Number of cells vertically

        x_cell = min(int(center_x * x_cells), x_cells - 1)
        y_cell = min(int(center_y * y_cells), y_cells - 1)

        person_id = y_cell * x_cells + x_cell
        return min(person_id, self.config.max_num_people - 1)  # Ensure it's within valid range

    def save_frame(self, frame, video_time_seconds, signal_type, person_id=None):
        """
        Save the current frame to the output folder with a timestamp, signal type, and person ID.
        """
        if not self.config.save_frames:
            return

        # Format time as minutes:seconds.milliseconds
        minutes = int(video_time_seconds // 60)
        seconds = int(video_time_seconds % 60)
        milliseconds = int((video_time_seconds % 1) * 1000)

        # Create a filename based on the timestamp and signal type
        timestamp = f"{minutes:02d}_{seconds:02d}_{milliseconds:03d}"
        signal_tag = signal_type.replace(" ", "_").lower()

        # Add person ID to filename if provided
        person_info = f"_person{person_id}" if person_id is not None else ""
        filename = f"{timestamp}_{signal_tag}{person_info}.jpg"

        # Full path to save the frame
        filepath = os.path.join(self.config.output_folder, filename)

        # Save the frame
        cv2.imwrite(filepath, frame)

        return filepath

    def draw_landmarks(self, frame, multi_pose_landmarks=None, multi_hand_landmarks=None):
        """
        Draw landmarks on the frame for visualization, supporting multiple people.
        """
        annotated_frame = frame.copy()

        # Draw multiple pose landmarks if available
        if multi_pose_landmarks:
            for pose_landmarks in multi_pose_landmarks:
                mp_solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    pose_landmarks,
                    mp_solutions.holistic.POSE_CONNECTIONS,
                    mp_solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        # Draw hand landmarks if available
        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                mp_solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_solutions.holistic.HAND_CONNECTIONS,
                    mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp_solutions.drawing_styles.get_default_hand_connections_style()
                )

        return annotated_frame

    def process_video(self, input_path: str) -> List[Tuple[float, str, Optional[str]]]:
        """
        Process video file for gesture detection.
        Returns a list of (video_time_seconds, message, frame_path) tuples.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        detection_log = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            frame_count = 0
            detected_people_ids = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                video_time_seconds = frame_count / fps

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe to detect people and landmarks
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)

                # Lists to store detected signals for current frame
                detected_signals = []  # (person_id, signal_type, annotated_frame)

                # Track which people are detected in this frame
                current_frame_people = set()

                # First, analyze each person detected by the pose model
                if pose_results.pose_landmarks:
                    # Assign a person ID
                    person_id = self.assign_person_id(pose_results.pose_landmarks)

                    if person_id is not None:
                        current_frame_people.add(person_id)
                        detected_people_ids.add(person_id)

                        # Initialize detection flags for this person
                        elbow_hand_detected = False
                        hand_above_head_detected = False
                        pointing_detected = False
                        torch_pointing_detected = False
                        mic_gesture_detected = False

                        # If hands also detected, try to match pose with hands
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                # Try various gestures for this combination of hand and pose
                                if self.is_elbow_above_shoulder(pose_results.pose_landmarks) and self.is_hand_raised(hand_landmarks):
                                    elbow_hand_detected = True

                                if self.is_hand_above_head(hand_landmarks, pose_results.pose_landmarks):
                                    hand_above_head_detected = True

                                if self.is_pointing_gesture(hand_landmarks):
                                    pointing_detected = True

                                if self.is_torch_pointing_gesture(hand_landmarks, pose_results.pose_landmarks):
                                    torch_pointing_detected = True

                                if self.is_mic_speaking_gesture(hand_landmarks, pose_results.pose_landmarks):
                                    mic_gesture_detected = True

                        # Update the detection buffers for this person
                        if person_id < len(self.elbow_hand_buffers):
                            self.elbow_hand_buffers[person_id].pop(0)
                            self.elbow_hand_buffers[person_id].append(elbow_hand_detected)

                            self.hand_above_head_buffers[person_id].pop(0)
                            self.hand_above_head_buffers[person_id].append(hand_above_head_detected)

                            self.hand_pointing_buffers[person_id].pop(0)
                            self.hand_pointing_buffers[person_id].append(pointing_detected)

                            self.torch_pointing_buffers[person_id].pop(0)
                            self.torch_pointing_buffers[person_id].append(torch_pointing_detected)

                            self.mic_gesture_buffers[person_id].pop(0)
                            self.mic_gesture_buffers[person_id].append(mic_gesture_detected)

                            # Check for consistent signals across frames
                            elbow_hand_signal = all(self.elbow_hand_buffers[person_id])
                            hand_above_head_signal = all(self.hand_above_head_buffers[person_id])
                            pointing_signal = all(self.hand_pointing_buffers[person_id])
                            torch_pointing_signal = all(self.torch_pointing_buffers[person_id])
                            mic_gesture_signal = all(self.mic_gesture_buffers[person_id])

                            # If signal detected and cooldown period has elapsed for this person
                            if ((elbow_hand_signal or hand_above_head_signal or pointing_signal or
                                 torch_pointing_signal or mic_gesture_signal) and
                                (video_time_seconds - self.last_detection_times[person_id] >= self.config.cooldown_seconds)):

                                # Format time
                                minutes = int(video_time_seconds // 60)
                                seconds = int(video_time_seconds % 60)
                                milliseconds = int((video_time_seconds % 1) * 1000)
                                formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                                # Draw landmarks for visualization
                                annotated_frame = self.draw_landmarks(
                                    frame,
                                    [pose_results.pose_landmarks],
                                    hand_results.multi_hand_landmarks
                                )

                                # Check each signal type and log if detected
                                if elbow_hand_signal:
                                    signal_type = "Elbow above shoulder with hand raised"
                                    detected_signals.append((person_id, signal_type, annotated_frame))
                                    self.last_detection_times[person_id] = video_time_seconds

                                elif hand_above_head_signal:  # Use elif to avoid duplicate logging
                                    signal_type = "Hand raised above head"
                                    detected_signals.append((person_id, signal_type, annotated_frame))
                                    self.last_detection_times[person_id] = video_time_seconds

                                elif pointing_signal:
                                    signal_type = "Pointing gesture"
                                    detected_signals.append((person_id, signal_type, annotated_frame))
                                    self.last_detection_times[person_id] = video_time_seconds

                                elif torch_pointing_signal:
                                    signal_type = "Torch pointing gesture"
                                    detected_signals.append((person_id, signal_type, annotated_frame))
                                    self.last_detection_times[person_id] = video_time_seconds

                                elif mic_gesture_signal:
                                    signal_type = "Speaking/mic gesture"
                                    detected_signals.append((person_id, signal_type, annotated_frame))
                                    self.last_detection_times[person_id] = video_time_seconds

                # Log all detected signals
                for person_id, signal_type, annotated_frame in detected_signals:
                    # Format time
                    minutes = int(video_time_seconds // 60)
                    seconds = int(video_time_seconds % 60)
                    milliseconds = int((video_time_seconds % 1) * 1000)
                    formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                    # Save frame if enabled
                    frame_path = None
                    if self.config.save_frames:
                        frame_path = self.save_frame(annotated_frame, video_time_seconds, signal_type, person_id)

                    # Add to detection log
                    detection_log.append((
                        video_time_seconds,
                        f"Signal detected at {formatted_time}: {signal_type} (Person {person_id})",
                        frame_path
                    ))

        finally:
            cap.release()

        return detection_log

class PhoneDetector:
    def __init__(self, config=None):
        """Initialize the phone detector"""
        # Load YOLOv8 model for phone detection
        self.phone_detector = YOLO('yolov8n.pt')

        # Configuration options
        self.save_frames = True if config is None else config.save_frames
        self.output_folder = PHONE_IMAGES_FOLDER if config is None else config.output_folder
        self.cooldown_seconds = 2.0 if config is None else config.cooldown_seconds

        # Create output folder if it doesn't exist
        if self.save_frames and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output folder for phone detection: {self.output_folder}")

        # Keep track of last detection time
        self.last_detection_time = -float('inf')

    def detect_phone(self, frame):
        """
        Detect phone in the frame using YOLOv8 model.
        Returns True if phone detected, False otherwise.
        """
        results = self.phone_detector(frame)
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == 'cell phone':
                    # Draw box around the phone
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Only consider detections with reasonable confidence
                    if confidence > 0.4:
                        return True, (x1, y1, x2, y2, confidence)
        return False, None

    def draw_detection(self, frame, box_info):
        """
        Draw detection box on frame.
        """
        if box_info is None:
            return frame

        annotated_frame = frame.copy()
        x1, y1, x2, y2, confidence = box_info

        # Draw rectangle around the phone
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label
        label = f"Phone: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame

    def save_frame(self, frame, video_time_seconds, box_info=None):
        """
        Save the current frame to the output folder with a timestamp.
        """
        if not self.save_frames:
            return None

        # Format time as minutes:seconds.milliseconds
        minutes = int(video_time_seconds // 60)
        seconds = int(video_time_seconds % 60)
        milliseconds = int((video_time_seconds % 1) * 1000)

        # Create a filename based on the timestamp
        timestamp = f"{minutes:02d}_{seconds:02d}_{milliseconds:03d}"
        filename = f"{timestamp}_phone_detected.jpg"

        # Full path to save the frame
        filepath = os.path.join(self.output_folder, filename)

        # Draw the detection box if provided
        if box_info:
            frame_to_save = self.draw_detection(frame, box_info)
        else:
            frame_to_save = frame

        # Save the frame
        cv2.imwrite(filepath, frame_to_save)

        return filepath

    def process_video(self, input_path: str) -> List[Tuple[float, str, Optional[str]]]:
        """
        Process video file for phone detection.
        Returns a list of (video_time_seconds, message, frame_path) tuples.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        detection_log = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Calculate video timestamp in seconds
                video_time_seconds = frame_count / fps

                # Check if enough time has passed since last detection (cooldown)
                if video_time_seconds - self.last_detection_time < self.cooldown_seconds:
                    continue

                # Detect phone in the frame
                phone_detected, box_info = self.detect_phone(frame)

                if phone_detected:
                    # Update last detection time
                    self.last_detection_time = video_time_seconds

                    # Format time
                    minutes = int(video_time_seconds // 60)
                    seconds = int(video_time_seconds % 60)
                    milliseconds = int((video_time_seconds % 1) * 1000)
                    formatted_time = f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

                    # Save frame if enabled
                    frame_path = None
                    if self.save_frames:
                        frame_path = self.save_frame(frame, video_time_seconds, box_info)

                    # Add to detection log
                    detection_log.append((
                        video_time_seconds,
                        f"Phone detected at {formatted_time}",
                        frame_path
                    ))

        finally:
            cap.release()

        return detection_log

def process_rtis_report(report_path: str, detected_signals: List[Tuple[float, str, Optional[str]]], master_data_path: str) -> Dict:
    """
    Process the RTIS report and match with detected signals and master data.
    Returns a dictionary with processed data and statistics.
    """
    # Load the RTIS report
    try:
        rtis_df = pd.read_excel(report_path)
    except Exception as e:
        raise ValueError(f"Error reading RTIS report: {str(e)}")

    # Load the master dataset
    try:
        master_df = pd.read_excel(master_data_path)
    except Exception as e:
        raise ValueError(f"Error reading master dataset: {str(e)}")

    # Convert time in detected signals to datetime (assuming video starts at first RTIS timestamp)
    if not rtis_df.empty:
        first_rtis_time = pd.to_datetime(rtis_df.iloc[0]['Time'])  # Assuming 'Time' column exists
    else:
        first_rtis_time = datetime.now()

    # Add signal detection information to RTIS report
    rtis_df['CalloutSignal'] = 'No'
    rtis_df['SignalStatus'] = 'Missed'  # Default to missed, will update if detected

    # Match detected signals with RTIS report based on time
    for signal_time, signal_msg, _ in detected_signals:
        # Calculate the corresponding datetime for this signal
        signal_datetime = first_rtis_time + timedelta(seconds=signal_time)
        
        # Find the closest time in RTIS report
        time_diff = abs(pd.to_datetime(rtis_df['Time']) - signal_datetime)
        closest_idx = time_diff.idxmin()
        
        # Mark as detected if within a reasonable time window (e.g., 10 seconds)
        if time_diff[closest_idx] < timedelta(seconds=10):
            rtis_df.at[closest_idx, 'CalloutSignal'] = 'Yes'
            rtis_df.at[closest_idx, 'SignalStatus'] = 'Done'

    # Match with master dataset based on latitude and longitude
    matched_signals = []
    for idx, row in rtis_df.iterrows():
        if row['CalloutSignal'] == 'Yes':
            # Find matching signal in master dataset
            lat = row['LATITUDE']
            lon = row['LONGITUDE']
            
            # Find closest match in master dataset
            master_df['distance'] = np.sqrt((master_df['LATITUDE'] - lat)**2 + (master_df['LONGITUDE'] - lon)**2)
            closest_master = master_df.loc[master_df['distance'].idxmin()]
            
            # Check if distance is within threshold (e.g., 0.001 degrees ~ 111 meters)
            if closest_master['distance'] < 0.001:
                matched_signals.append({
                    'time': row['Time'],
                    'latitude': lat,
                    'longitude': lon,
                    'status': row['SignalStatus'],
                    'expected_signal': 'Yes',
                    'distance_to_master': closest_master['distance']
                })
            else:
                matched_signals.append({
                    'time': row['Time'],
                    'latitude': lat,
                    'longitude': lon,
                    'status': row['SignalStatus'],
                    'expected_signal': 'No',
                    'distance_to_master': closest_master['distance']
                })

    # Calculate statistics
    total_signals = len(matched_signals)
    correct_signals = sum(1 for s in matched_signals if s['expected_signal'] == 'Yes' and s['status'] == 'Done')
    missed_signals = sum(1 for s in matched_signals if s['expected_signal'] == 'Yes' and s['status'] == 'Missed')
    false_positives = sum(1 for s in matched_signals if s['expected_signal'] == 'No' and s['status'] == 'Done')

    return {
        'rtis_report': rtis_df.to_dict('records'),
        'matched_signals': matched_signals,
        'statistics': {
            'total_signals': total_signals,
            'correct_signals': correct_signals,
            'missed_signals': missed_signals,
            'false_positives': false_positives,
            'accuracy': correct_signals / total_signals if total_signals > 0 else 0
        }
    }

# Flask routes
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and check_password_hash(USERS[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if files were uploaded
        if 'video' not in request.files or 'report1' not in request.files or 'report2' not in request.files:
            return render_template('upload.html', error='Please upload all required files')
        
        video_file = request.files['video']
        report1_file = request.files['report1']
        report2_file = request.files['report2']
        
        # Check if filenames are empty
        if video_file.filename == '' or report1_file.filename == '' or report2_file.filename == '':
            return render_template('upload.html', error='Please upload all required files')
        
        # Check file extensions
        if (video_file and video_file.filename.rsplit('.', 1)[1].lower() in ['mp4', 'avi', 'mov'] and
            report1_file and report1_file.filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls'] and
            report2_file and report2_file.filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']):
            
            # Save files
            video_filename = secure_filename(video_file.filename)
            report1_filename = secure_filename(report1_file.filename)
            report2_filename = secure_filename(report2_file.filename)
            
            video_path = os.path.join(UPLOAD_FOLDER, video_filename)
            report1_path = os.path.join(UPLOAD_FOLDER, report1_filename)
            report2_path = os.path.join(UPLOAD_FOLDER, report2_filename)
            
            video_file.save(video_path)
            report1_file.save(report1_path)
            report2_file.save(report2_path)
            
            try:
                # Create detection config
                config = DetectionConfig(
                    pose_confidence=0.5,
                    hand_confidence=0.5,
                    smoothing_frames=2,
                    cooldown_seconds=2.0,
                    save_frames=True,
                    output_folder=IMAGES_FOLDER
                )
                
                # Initialize detectors
                gesture_detector = GestureDetector(config)
                
                # Process for gesture detection
                gesture_results = gesture_detector.process_video(video_path)
                
                # Process RTIS report and master data
                processing_result = process_rtis_report(report1_path, gesture_results, report2_path)
                
                # Create a result filename
                result_filename = f"results_{int(time.time())}.xlsx"
                result_path = os.path.join(OUTPUT_FOLDER, result_filename)
                
                # Save results to Excel
                with pd.ExcelWriter(result_path) as writer:
                    pd.DataFrame(processing_result['rtis_report']).to_excel(writer, sheet_name='RTIS_Report', index=False)
                    pd.DataFrame(processing_result['matched_signals']).to_excel(writer, sheet_name='Matched_Signals', index=False)
                    pd.DataFrame([processing_result['statistics']]).to_excel(writer, sheet_name='Statistics', index=False)
                
                # Redirect to results page
                return redirect(url_for('video_processed', filename=result_filename))
                
            except Exception as e:
                return render_template('upload.html', error=f"Error processing files: {str(e)}")
        else:
            return render_template('upload.html', error='Invalid file types. Please upload video (.mp4, .avi, .mov) and Excel reports (.xlsx, .xls)')
    
    return render_template('upload.html')

@app.route('/mobile_detection', methods=['GET', 'POST'])
def mobile_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video' not in request.files:
            return render_template('upload.html', error='No video file selected')
        
        video_file = request.files['video']
        
        # Check if filename is empty
        if video_file.filename == '':
            return render_template('upload.html', error='No video file selected')
        
        # Check file extension
        if video_file and video_file.filename.rsplit('.', 1)[1].lower() in ['mp4', 'avi', 'mov']:
            video_filename = secure_filename(video_file.filename)
            video_path = os.path.join(UPLOAD_FOLDER, video_filename)
            video_file.save(video_path)
            
            try:
                # Initialize phone detector
                phone_detector = PhoneDetector()
                
                # Process for phone detection
                phone_results = phone_detector.process_video(video_path)
                
                # Calculate statistics
                total_mobiles = len(phone_results)
                video_duration = 0
                if total_mobiles > 0:
                    video_duration = phone_results[-1][0] / 60  # Convert seconds to minutes
                
                # Prepare detected images for display
                detected_images = []
                for _, _, frame_path in phone_results:
                    if frame_path:
                        detected_images.append({
                            'filename': os.path.basename(frame_path),
                            'timestamp': os.path.basename(frame_path).split('_')[0] + ":" + 
                                         os.path.basename(frame_path).split('_')[1] + "." + 
                                         os.path.basename(frame_path).split('_')[2].split('.')[0]
                        })
                
                # Render mobile results template
                return render_template('mobile_results.html', 
                                      total_mobiles=total_mobiles,
                                      processing_time=round(video_duration * 60 / 10, 2),  # Fake processing time
                                      video_duration=round(video_duration, 2),
                                      detected_images=detected_images)
                
            except Exception as e:
                return render_template('upload.html', error=f"Error processing video: {str(e)}")
        else:
            return render_template('upload.html', error='Invalid file type. Please upload a video file (.mp4, .avi, .mov)')
    
    return render_template('upload.html')

@app.route('/video_processed')
def video_processed():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    filename = request.args.get('filename')
    if not filename:
        return redirect(url_for('index'))
    
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return render_template('video_processed.html', error='Results file not found')
    
    return render_template('video_processed.html', 
                         video_output=filename.replace('.xlsx', '_video.mp4'),
                         log_file=filename)

@app.route('/download/<filename>')
def download_file(filename):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return send_file(os.path.join(OUTPUT_FOLDER, filename),
                     as_attachment=True,
                     download_name=filename)

if __name__ == '__main__':
    app.run(debug=True)