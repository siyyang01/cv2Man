import cv2
import os
import numpy as np
import pandas as pd
import pytesseract
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Tesseract 실행 파일 경로 설정
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

frame_dir = "learn/data"
output_dir = "learn/predic"
model_path = "digit_recognition_model.h5"

'''
# 1. 비디오에서 프레임 추출
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
'''


# 2. CSV 파일에서 데이터 로드
def load_data_from_csv(image_dir, csv_file):
    labels_df = pd.read_csv(csv_file, encoding='utf-8')

    images = []
    labels = []

    max_value = labels_df['label'].max()  # 레이블의 최대값으로 정규화
    for _, row in labels_df.iterrows():
        if pd.isna(row['filename']) or pd.isna(row['label']):
            continue

        img_path = os.path.join(image_dir, row['filename'])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (64, 64))  # 이미지 크기 통일
            images.append(image)
            labels.append(float(row['label']) / max_value)  # 정규화된 레이블 저장

    images = np.array(images).reshape(-1, 64, 64, 1) / 255.0
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.2, random_state=42), max_value

# 3. 모델 설계 및 학습
def create_regression_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # 회귀를 위한 출력층
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# 4. 저장된 프레임에서 숫자 예측 및 합산
def predict_digits_from_frames(frame_dir, model, max_value):
    total_sum = 0
    for frame_file in sorted(os.listdir(frame_dir)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frame_dir, frame_file)
            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, (64, 64)) / 255.0
            image = image.reshape(1, 64, 64, 1)

            prediction = model.predict(image)
            total_sum += prediction[0][0] * max_value  # 역정규화

    return total_sum

# 5. 모든 단계를 통합하여 실행
def main(video_path, csv_file):
    # extract_frames_from_video(video_path, frame_dir, interval=30)

    # CSV 파일에서 데이터 로드
    (X_train, X_val, y_train, y_val), max_value = load_data_from_csv(frame_dir, csv_file)

    # 회귀 모델 생성 및 학습
    model = create_regression_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # 모델 저장
    model.save(model_path)

    # 학습된 모델 로드 및 숫자 예측
    model = load_model(model_path)
    predicted_total_sum = predict_digits_from_frames(frame_dir, model, max_value)
    print(f"Predicted Total Sum: {predicted_total_sum}")

    # 정답값과 비교
    expected_sum = 839169342  # 정답 파일에 명시된 합
    print(f"Difference from expected sum: {abs(predicted_total_sum - expected_sum)}")

# 실행
video_path = 'dantashort.mp4'
csv_file = 'learn/data/answer.csv'
main(video_path, csv_file)
