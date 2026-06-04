import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

my_model = joblib.load("domain_model.pkl")
Base_options = mp.tasks.BaseOptions
Hand_detector =mp.tasks.vision.HandLandmarker
Hand_detector_options = mp.tasks.vision.HandLandmarkerOptions
options = Hand_detector_options(Base_options(model_asset_path="hand_landmarker.task"), num_hands=2)

segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
    mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="selfie_segmenter_landscape.tflite"),
        output_category_mask=True
    )
)

detector = Hand_detector.create_from_options(options)
print(f"loaded. connecting to default camera")
cap = cv2.VideoCapture(0)
cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("preloading background video...")
invoid_vid = cv2.VideoCapture("invoid.mp4")
bg_fps = invoid_vid.get(cv2.CAP_PROP_FPS)
bg_frames = []
while True:
    ret, f = invoid_vid.read()
    if not ret:
        break
    bg_frames.append(cv2.resize(f, (cap_w, cap_h)))
invoid_vid.release()
bg_frames = np.array(bg_frames)
bg_frame_count = len(bg_frames)
bg_frame_idx = 0.0
print(f"loaded {bg_frame_count} background frames")
counter = 0
threshold = 15
triggered = False
last_time = time.time()

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
        if model_prediction == "infinite void":
            counter += 1
        if counter > threshold and not triggered:
            triggered = True
    now = time.time()
    elapsed = now - last_time
    last_time = now

    if triggered:
        bg_frame_idx = (bg_frame_idx + elapsed * bg_fps) % bg_frame_count
        vid_frame = bg_frames[int(bg_frame_idx)]
        segmentation_result = segmenter.segment(mp_ready_frame)
        mask = segmentation_result.category_mask.numpy_view()
        composite = np.where(mask == 0, frame, vid_frame)
        cv2.imshow("Infinite Void", composite)
    else:
        cv2.imshow("Infinite Void", frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break