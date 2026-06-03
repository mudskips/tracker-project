import cv2
import mediapipe as mp
import csv

Base_options = mp.tasks.BaseOptions
Hand_detector =mp.tasks.vision.HandLandmarker
Hand_detector_options = mp.tasks.vision.HandLandmarkerOptions
options = Hand_detector_options(Base_options(model_asset_path="hand_landmarker.task"), num_hands=2)

detector = Hand_detector.create_from_options(options)
print(f"loaded. connecting to default camera")

cap = cv2.VideoCapture(0)

recording = False
sign_name = "neutral"
target_file = f"{sign_name}.csv"

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
        while len(data) < 42:
            data.append(0.0)
        if recording:
            with open(file = target_file, mode = "a", newline = "") as df:
                writer = csv.writer(df)
                writer.writerow([sign_name]+ data)
    if recording:
        status = "o recording o"
        color = (0, 0, 255)
    else:
        status = f"press r to record"
        color = (255, 255, 255)

    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    cv2.imshow("data collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        recording = not recording
    if key == ord(" "):
        break


        
