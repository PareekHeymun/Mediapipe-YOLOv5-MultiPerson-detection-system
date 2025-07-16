import cv2
import numpy as np
from yolov5 import YOLOv5
import PoseModule as pm
import time
import torch

# Configuration dictionary
CONFIG = {
    'video_path': r"C:\Users\Heymun\Desktop\Hepyn\Computer Vision\MediapipenYolo\human_pose.mp4",
    'frame_size': None, # Reduced for better FPS
    'yolov5_confidence': 0.5,  # Confidence threshold for person detection
    'yolov5_model': 'yolov5s.pt',  # Lightweight model
    'mediapipe_model_complexity': 0,  # Lite model for CPU
    'mediapipe_detection_conf': 0.5,
    'mediapipe_tracking_conf': 0.5,
    'max_detections': 2,  # Limit to 2 detections
}

def main():
    # Initialize YOLOv5
    try:
        model = YOLOv5(CONFIG['yolov5_model'], device='cpu')
        model.model.eval()  # Set to evaluation mode for efficiency
        model.conf = CONFIG['yolov5_confidence']  # Set confidence threshold
        model.max_det = CONFIG['max_detections']  # Limit to 2 detections
    except Exception as e:
        print(f"Error loading YOLOv5: {e}")
        return

    # Initialize MediaPipe pose detector
    try:
        pose_detector = pm.poseDetector(
            model_complexity=CONFIG['mediapipe_model_complexity'],
            detectionCon=CONFIG['mediapipe_detection_conf'],
            trackCon=CONFIG['mediapipe_tracking_conf']
        )
    except Exception as e:
        print(f"Error initializing pose detector: {e}")
        return
    
    # Video capture
    cap = cv2.VideoCapture(CONFIG['video_path'])
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    # Dynamically get original video resolution
    if CONFIG['frame_size'] is None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        CONFIG['frame_size'] = (width, height)
        
    pTime = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("End of video or error reading frame.")
            break

        # Resize frame for consistency and speed
        frame = cv2.resize(frame, CONFIG['frame_size'])
        orig_frame = frame.copy()  # Keep original for visualization

        # YOLOv5 person detection
        with torch.no_grad():  # Disable gradient computation for efficiency
            results = model.predict(frame)
            detections = results.pred[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

        # Process each detected person
        for det in detections:
            if det[5] == 0:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, det[:4])
                # Add padding to ensure entire person is included
                padding = 20
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)
                
                # Crop the bounding box
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                # Run MediaPipe pose detection on cropped image
                _ = pose_detector.findPose(crop_img, draw=False)
                lmList = pose_detector.findPosition(crop_img, draw=False)
                if not lmList:
                    continue
                adjusted = [[idx, cx + x1, cy + y1] for idx, cx, cy in lmList]
                face_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
                face_pts = [(x, y) for idx, x, y in adjusted if idx in face_ids]
                if face_pts:
                    xs, ys = zip(*face_pts)
                    #dynamically adjust padding to match: face is to any body part ratio
                    pad = 20
                    fx1, fy1 = min(xs) - pad, min(ys) - pad
                    fx2, fy2 = max(xs) + pad, max(ys) + pad
                    fx1, fy1 = max(0, fx1), max(0, fy1)
                    fx2 = min(orig_frame.shape[1], fx2)
                    fy2 = min(orig_frame.shape[0], fy2)
                    cv2.rectangle(orig_frame,
                                (fx1, fy1),
                                (fx2, fy2),
                                (255, 0, 0), 2)
                    
                    #store for later cropping:
                    face_coords = (fx1, fy1, fx2, fy2);
                            
                
                for start_idx, end_idx in pose_detector.mpPose.POSE_CONNECTIONS:
                    if start_idx < len(adjusted) and end_idx < len(adjusted):
                        x1_lm, y1_lm = adjusted[start_idx][1:]
                        x2_lm, y2_lm = adjusted[end_idx][1:]
                        cv2.line(orig_frame, (x1_lm, y1_lm), (x2_lm, y2_lm), (0, 0, 255), 2)
                
                for _, cx_lm, cy_lm in adjusted:
                    cv2.circle(orig_frame, (cx_lm, cy_lm), 5, (0, 0, 255), cv2.FILLED)
                

                # Draw bounding box
                cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig_frame, f'Person {det[4]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        # Calculate and display FPS with smaller text
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(orig_frame, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

        # Display the frame
        cv2.imshow("Multi-Person Pose Detection", orig_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and face_coords is not None:
            x1, y1, x2, y2 = face_coords
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                print("Warning: Empty face crop, skipping save.")
                continue

            # Resize while maintaining aspect ratio
            target_size = (640, 480)
            h, w = face_img.shape[:2]
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_face = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Create a black canvas and center the resized face
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            x_offset = (target_size[0] - new_w) // 2
            y_offset = (target_size[1] - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_face

            fname = f"face_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()