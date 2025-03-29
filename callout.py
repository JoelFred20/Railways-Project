import os
import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from mediapipe import solutions as mp_solutions

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

        # Buffering for raised hand detection for each potential person
        self.raised_hand_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]

        # Last detection time for each person
        self.last_detection_times = [-float('inf')] * config.max_num_people

        # Create output folder if saving frames is enabled
        if config.save_frames and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
            print(f"Created output folder: {config.output_folder}")

    def format_time(self, total_seconds: float) -> str:
        """
        Format time with hours, minutes, and seconds.
        
        Args:
            total_seconds (float): Total seconds to format
        
        Returns:
            str: Formatted time string in HH:MM:SS format
        """
        # Calculate hours, minutes, and seconds
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        # Format the time string
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def is_hand_raised_and_extended(self, hand_landmarks, pose_landmarks) -> bool:
        """
        Checks if the hand is raised and extended, with specific criteria:
        1. Hand is above a certain height threshold
        2. Hand is fully extended (stretched out)
        3. Confirmed by pose landmarks
        """
        if not hand_landmarks or not pose_landmarks:
            return False

        # Get key landmarks
        wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]

        # Get shoulder and nose landmarks for reference
        try:
            left_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.NOSE]
        except:
            return False

        # Check hand height - must be above shoulder level
        hand_is_high = wrist.y < min(left_shoulder.y, right_shoulder.y)

        # Check hand extension - all finger tips should be relatively far from wrist
        # Create vectors from wrist to each finger tip
        thumb_vector = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
        index_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        pinky_vector = np.array([pinky_tip.x - wrist.x, pinky_tip.y - wrist.y])

        # Calculate vector lengths
        thumb_length = np.linalg.norm(thumb_vector)
        index_length = np.linalg.norm(index_vector)
        pinky_length = np.linalg.norm(pinky_vector)

        # Hand is considered extended if finger tip vectors are significantly long
        hand_is_extended = (
            thumb_length > 0.1 and 
            index_length > 0.1 and 
            pinky_length > 0.1
        )

        # Optional: Check angle of hand relative to body
        # This helps distinguish between truly raised hands and other positions
        # Compute angle between hand vector and vertical axis
        hand_main_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        vertical_vector = np.array([0, -1])  # Pointing upwards

        # Normalize vectors
        hand_main_vector = hand_main_vector / np.linalg.norm(hand_main_vector)
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)

        # Compute angle (in radians)
        angle = np.arccos(np.clip(np.dot(hand_main_vector, vertical_vector), -1.0, 1.0))
        
        # Hand should be relatively vertical (within 45 degrees)
        hand_is_vertical = abs(angle) < np.pi/4  # About 45 degrees

        return hand_is_high and hand_is_extended and hand_is_vertical

    def process_video(self, input_path: str) -> List[Tuple[float, str, Optional[str]]]:
        """
        Process video file for raised hand detection.
        Logs detection events and timestamps from the video.
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

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe to detect people and landmarks
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)

                # First, check pose detections
                if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                    # Try to match hands with pose
                    for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        # Check if this is a raised hand
                        if self.is_hand_raised_and_extended(hand_landmarks, pose_results.pose_landmarks):
                            # Identify person (use hand index as a simple identifier)
                            person_id = hand_idx % self.config.max_num_people

                            # Update raised hand buffer
                            self.raised_hand_buffers[person_id].pop(0)
                            self.raised_hand_buffers[person_id].append(True)

                            # Check if hand is consistently raised
                            if all(self.raised_hand_buffers[person_id]):
                                # Check cooldown
                                if (video_time_seconds - self.last_detection_times[person_id] >= self.config.cooldown_seconds):
                                    # Mark detection
                                    signal_type = "Hand Raised and Extended"
                                    
                                    # Draw landmarks for visualization
                                    annotated_frame = self.draw_landmarks(
                                        frame,
                                        [pose_results.pose_landmarks],
                                        [hand_landmarks]
                                    )

                                    # Save frame
                                    frame_path = None
                                    if self.config.save_frames:
                                        frame_path = self.save_frame(
                                            annotated_frame, 
                                            video_time_seconds, 
                                            signal_type, 
                                            person_id
                                        )

                                    # Format time using the new method
                                    formatted_time = self.format_time(video_time_seconds)

                                    # Log detection
                                    detection_log.append((
                                        video_time_seconds,
                                        f"Signal detected at {formatted_time}: {signal_type} (Person {person_id})",
                                        frame_path
                                    ))

                                    # Update last detection time
                                    self.last_detection_times[person_id] = video_time_seconds

        finally:
            cap.release()

        return detection_log

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

    def save_frame(self, frame, video_time_seconds, signal_type, person_id=None):
        """
        Save the current frame to the output folder with a timestamp, signal type, and person ID.
        """
        if not self.config.save_frames:
            return

        # Format time as hours_minutes_seconds
        hours = int(video_time_seconds // 3600)
        minutes = int((video_time_seconds % 3600) // 60)
        seconds = int(video_time_seconds % 60)

        # Create a filename based on the timestamp and signal type
        timestamp = f"{hours:02d}_{minutes:02d}_{seconds:02d}"
        signal_tag = signal_type.replace(" ", "_").lower()

        # Add person ID to filename if provided
        person_info = f"_person{person_id}" if person_id is not None else ""
        filename = f"{timestamp}_{signal_tag}{person_info}.jpg"

        # Full path to save the frame
        filepath = os.path.join(self.config.output_folder, filename)

        # Save the frame
        cv2.imwrite(filepath, frame)

        return filepath

def save_log_to_file(log: List[Tuple[float, str, Optional[str]]], output_path: str):
    """
    Save the detection log to a file.
    """
    with open(output_path, 'w') as file:
        # Sort log by timestamp
        sorted_log = sorted(log, key=lambda x: x[0])
        for timestamp, message, frame_path in sorted_log:
            if frame_path:
                file.write(f"{message} [Frame saved: {frame_path}]\n")
            else:
                file.write(f"{message}\n")

def main():
    """Main entry point"""
    # Create a timestamp for the output folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = f"detected_frames_{timestamp}"

    config = DetectionConfig(
        pose_confidence=0.5,
        hand_confidence=0.5,
        hand_tracking_confidence=0.5,
        smoothing_frames=2,       # Reduced to be more responsive
        cooldown_seconds=1.0,     # Reduced cooldown for more detections
        max_num_people=10,        # Support up to 10 people in frame
        save_frames=True,
        output_folder=output_folder
    )

    detector = GestureDetector(config)

    try:
        # Use a command line argument for the video path or a default
        import sys
        video_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Joel Fredrick/Desktop/call-out-signal/uploads/16058_AJJ-MAS_1.mp4"

        # Verify the file exists first
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print("Please check the path and try again.")
            return

        detection_log = detector.process_video(video_path)

        if detection_log:  # Only save if there are detections
            # Save detection log to a file
            log_file_path = os.path.join(output_folder, 'detection_log.txt')
            save_log_to_file(detection_log, log_file_path)

            # Print summary
            print(f"Detected {len(detection_log)} signals in the video.")
            print("Detection Log saved to:", log_file_path)
            print(f"Detected frames saved to: {output_folder}")

            # Print breakdown by signal type and person
            signal_counts = {}
            person_counts = {}

            for _, message, _ in detection_log:
                # Extract signal type from message
                signal_type = message.split(": ")[1].split(" (")[0]
                person_id = message.split("(Person ")[1].split(")")[0]

                # Count occurrences of each signal type
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

                # Count occurrences for each person
                person_counts[person_id] = person_counts.get(person_id, 0) + 1

            # Print summary of signal types
            print("\nSignal Type Breakdown:")
            for signal_type, count in signal_counts.items():
                print(f"  {signal_type}: {count}")

            # Print summary of detections per person
            print("\nDetections Per Person:")
            for person_id, count in person_counts.items():
                print(f"  Person {person_id}: {count}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()