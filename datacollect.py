import cv2
import mediapipe as mp

Base_options = mp.tasks.BaseOptions
Hand_detector =mp.tasks.vision.HandLandmarker
Hand_detector_options = mp.tasks.vision.HandLandmarkerOptions
options = Hand_detector_options(Base_options(model_asset_path="hand_landmarker.task"), num_hands=2)

detector = Hand_detector.create_from_options(options)
print(f"loaded. connecting to default camera")

cap = cv2.VideoCapture(0)

recording = False
sign_name = "infinite void"
target_file = f"{sign_name}.csv"

while True:
    result, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_corrected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_ready_frame = mp.Image(image_format= mp.ImageFormat.SRGB, data=rgb_corrected)
    detection = detector.detect(mp_ready_frame)
    if detection.hand_landmarks:
        data = []
        for hand in detection.hand_landmarks:
            for landmark in hand:
                data.append(landmark.x)
                data.append(landmark.y)
        while len(data) < 84:
            row.append(0.0)
        
