import cv2
import os
import math
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, input_video_path, output_folder='phone_detection_output', save_frames=True):
        """Initialize the phone detector"""
        # Load YOLOv8 model for phone detection
        self.phone_detector = YOLO('C:/Users/Joel Fredrick/Downloads/Telegram Desktop/train/train/yolov8n.pt')
        
        self.input_video_path = input_video_path
        self.save_frames = save_frames
        self.output_folder = output_folder
        self.cooldown_seconds = 2.0

        # Create output folder if it doesn't exist
        if self.save_frames and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            os.makedirs(os.path.join(self.output_folder, 'frames'))
            os.makedirs(os.path.join(self.output_folder, 'phone_crops'))
            print(f"Created output folder for phone detection: {self.output_folder}")

        self.last_detection_time = -float('inf')

    def process_frame(self, frame, confidence_threshold=0.25):
        """
        Process a single frame to detect phones.
        """
        results = self.phone_detector(frame)
        annotated_frame = frame.copy()
        is_phone_detected = False

        all_detections = results[0].boxes
        people_boxes = []

        for detection in all_detections:
            class_id = int(detection.cls)
            confidence = float(detection.conf)
            if class_id == 0 and confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                people_boxes.append((x1, y1, x2, y2))

        phone_detections = []
        for detection in all_detections:
            class_id = int(detection.cls)
            confidence = float(detection.conf)
            if class_id == 67 and confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                is_valid_phone = True
                for px1, py1, px2, py2 in people_boxes:
                    overlap_x1 = max(x1, px1)
                    overlap_y1 = max(y1, py1)
                    overlap_x2 = min(x2, px2)
                    overlap_y2 = min(y2, py2)
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        phone_area = (x2 - x1) * (y2 - y1)
                        person_area = (px2 - px1) * (py2 - py1)
                        if phone_area > 0.5 * person_area:
                            is_valid_phone = False
                            break
                if is_valid_phone:
                    is_phone_detected = True
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Phone {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    phone_detections.append({'bbox': (x1, y1, x2, y2), 'confidence': confidence})

        return annotated_frame, is_phone_detected, phone_detections

    def process_video(self):
        """
        Process video file for phone detection.
        """
        if not os.path.exists(self.input_video_path):
            raise FileNotFoundError(f"Input video not found: {self.input_video_path}")

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_path = os.path.join(self.output_folder, 'detected_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        timestamp_file_path = os.path.join(self.output_folder, 'phone_usage_timestamps.txt')
        timestamp_file = open(timestamp_file_path, 'w')
        
        frame_number = 0
        total_phone_time = 0
        in_sequence = False
        sequence_start = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, is_phone_detected, phone_detections = self.process_frame(frame)
                out.write(processed_frame)
                
                if is_phone_detected and self.save_frames:
                    timestamp = frame_number / fps
                    frame_filename = os.path.join(self.output_folder, 'frames', f"phone_detected_frame_{frame_number:06d}_{timestamp:.2f}s.jpg")
                    cv2.imwrite(frame_filename, processed_frame)
                
                if is_phone_detected:
                    total_phone_time += 1
                    if not in_sequence:
                        in_sequence = True
                        sequence_start = frame_number / fps
                else:
                    if in_sequence:
                        sequence_end = frame_number / fps
                        timestamp_file.write(f"PHONE USAGE FROM {sequence_start:.2f}s TO {sequence_end:.2f}s\n")
                        in_sequence = False
                
                frame_number += 1
            
            if in_sequence:
                sequence_end = frame_number / fps
                timestamp_file.write(f"PHONE USAGE FROM {sequence_start:.2f}s TO {sequence_end:.2f}s\n")
            
            total_time_seconds = total_phone_time / fps
            timestamp_file.write(f"\nTOTAL PHONE USAGE TIME: {total_time_seconds:.2f} seconds\n")
        finally:
            cap.release()
            out.release()
            timestamp_file.close()
            cv2.destroyAllWindows()
            print(f"Processing complete. Output saved to {self.output_folder}")

if __name__ == '__main__':
    input_video_path = "C:/Users/Joel Fredrick/Downloads/Telegram Desktop/CH4_00000301469000000.mp4"
    detector = PhoneDetector(input_video_path)
    detector.process_video()
