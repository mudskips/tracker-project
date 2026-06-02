import mediapipe as mp
import cv2

hands = mp.solutions.hands
hands = hands.Hands()
draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)