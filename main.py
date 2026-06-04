import cv2
import mediapipe as mp
import joblib
import numpy as np
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

invoid_vid = cv2.VideoCapture("invoid.mp4")

detector = Hand_detector.create_from_options(options)
print(f"loaded. connecting to default camera")

cap = cv2.VideoCapture(0)
counter = 0
threshold = 30
triggered = False

while True:
    result, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_corrected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_ready_frame = mp.Image(image_format= mp.ImageFormat.SRGB, data=rgb_corrected)
    detection = detector.detect(mp_ready_frame)
    segmentation_result = segmenter.segment(mp_ready_frame)
    mask = segmentation_result.category_mask.numpy_view()
    print("mask shape:", mask.shape)
    cv2.imshow("mask", mask)
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
    if triggered:
        ret_vid, vid_frame = invoid_vid.read()
        if not ret_vid:
            invoid_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_vid, vid_frame = invoid_vid.read()
        vid_frame = cv2.resize(vid_frame, (frame.shape[1], frame.shape[0]))
        person_mask = (mask == 0).astype(np.uint8)
        bg_mask = (mask != 0).astype(np.uint8)
        person_mask_3d = np.dstack([person_mask, person_mask, person_mask])
        bg_mask_3d = np.dstack([bg_mask, bg_mask, bg_mask])
        you = frame * person_mask_3d
        back = vid_frame * bg_mask_3d
        composite = you + back
        cv2.imshow("Infinite Void", composite)
    else:
        cv2.imshow("Infinite Void Playground", frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break