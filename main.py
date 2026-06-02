import cv2
import mediapipe as mp

Base_options = mp.tasks.BaseOptions
Hand_detector =mp.tasks.vision.HandLandmarker
Hand_detector_options = mp.tasks.vision.HandLandmarkerOptions
options = Hand_detector_options(Base_options(model_asset_path="hand_landmarker.task"))

detector = Hand_detector.create_from_options(options)
print(f"model loaded successfully! nice.")