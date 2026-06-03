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
