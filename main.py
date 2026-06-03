import cv2
import mediapipe as mp
import joblib
import numpy as np
my_model = joblib.load("domain_model.pkl")

Base_options = mp.tasks.BaseOptions
Hand_detector =mp.tasks.vision.HandLandmarker
Hand_detector_options = mp.tasks.vision.HandLandmarkerOptions
options = Hand_detector_options(Base_options(model_asset_path="hand_landmarker.task"), num_hands=2)

detector = Hand_detector.create_from_options(options)
print(f"loaded. connecting to default camera")

cap = cv2.VideoCapture(0)

while True:
    result, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_corrected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_ready_frame = mp.Image(image_format= mp.ImageFormat.SRGB, data=rgb_corrected)
    detection = detector.detect(mp_ready_frame)
    if detection.hand_landmarks:
        data = []
        wrist_y = detection.hand_landmarks[0][0].y
        wrist_x = detection.hand_landmarks[0][0].x
        for landmark in detection.hand_landmarks[0]:
            data.append(landmark.x - wrist_x)
            data.append(landmark.y - wrist_y)
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 6, (0, 30, 0), -1)
        model_prediction= my_model.predict([data])[0]
        print(model_prediction)
    cv2.imshow("P*lantir drone (press space to close)", frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

