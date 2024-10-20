import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
import os

'''
112541855, 109366347, 86813841, 108650379, 111339115, 111975463, 86071814, 112410528, sum=839169342
'''



pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

output_dir = "danta"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 노란색 숫자 범위 정의
def extract_yellow_regions(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 노란색 영역 추출
    yellow_areas = cv2.bitwise_and(frame, frame, mask=mask)
    
    return yellow_areas

def preprocess_image(image):
    yellow_areas = extract_yellow_regions(image)
    gray = cv2.cvtColor(yellow_areas, cv2.COLOR_BGR2GRAY) # 회색조 변환
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 노이즈 제거
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # 적응형 임계처리로 숫자 영역 강조하기
    
    return thresh



# 노란색 숫자 추출
def extract_numbers(image, frame_number):
    processed_image = preprocess_image(image)

    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    numbers = [int(num) for num in text.split() if num.isdigit()] #숫자 추출
    
    if numbers:
        detected_text = ", ".join(map(str, numbers))
        cv2.putText(image, detected_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        output_filename = os.path.join(output_dir, f"frame_{frame_number}_damage_{detected_text}.png")
        cv2.imwrite(output_filename, image)
    
    return sum(numbers)

def process_video(video_path, resize_factor=0.5, save_regions=True):
    total_damage = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            
            # Resize the frame to a smaller size
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            
            total_damage += extract_numbers(frame, frame_number)
            pbar.update(1)
    
    cap.release()
    return total_damage



# __main__
video_path = 'dantashort.mp4'
total_damage = process_video(video_path, resize_factor=0.5, save_regions=True)
print(f'Total yellow damage: {total_damage}')