import cv2
import pickle
import numpy as np

# Load keypoints from pkl file
def load_keypoints(pkl_file):
    with open(pkl_file, 'rb') as f:
        keypoints_data = pickle.load(f)
    return keypoints_data

# Visualize keypoints on frame
def draw_keypoints_on_frame(frame, keypoints, color=(0, 255, 0), radius=3):
    for person_keypoints in keypoints:
        for kpt in person_keypoints:
            x, y, conf = int(kpt[0]), int(kpt[1]), kpt[2]
            print(f"Keypoint: x={x}, y={y}, conf={conf}")  

            if conf > 0.5: 
                cv2.circle(frame, (x, y), radius, color, -1)
    return frame

# Visualize keypoints on video
def visualize_keypoints_on_video(video_path, pkl_file, resize_factor=0.5):
    keypoints_data = load_keypoints(pkl_file)
    cap = cv2.VideoCapture(video_path)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index < len(keypoints_data):
            keypoints = keypoints_data[frame_index]
            frame = draw_keypoints_on_frame(frame, keypoints)
        
        # Resize the frame
        frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        
        cv2.imshow('Keypoints on Video', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    
    pkl_file = '../walk_03-12-09-13-10-875_video.pkl'
    video_path = '../walk_03-12-09-13-10-875_video.mp4'

    visualize_keypoints_on_video(video_path, pkl_file, resize_factor=0.5)

