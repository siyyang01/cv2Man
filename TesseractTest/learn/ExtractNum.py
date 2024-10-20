import cv2
import os

def extract_frames_from_video(video_path, output_dir, interval=30):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1

    cap.release()
    print(f"총 {extracted_count}개의 프레임이 {output_dir}에 저장되었습니다.")

video_path = 'dantashort.mp4'
output_dir = 'learn/data'
extract_frames_from_video(video_path, output_dir, interval=30)
