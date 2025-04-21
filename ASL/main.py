import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Try different camera indices
for i in range(3):  # Try first 3 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Successfully opened camera {i}")
        break
    else:
        print(f"Failed to open camera {i}")
        cap.release()

if not cap.isOpened():
    print("Error: Could not open any camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
        
    # Check if frame is valid
    if frame is None or frame.size == 0:
        print("Error: Invalid frame")
        break
        
    cv2.imshow('ASL', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


