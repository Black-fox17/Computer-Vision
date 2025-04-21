import cv2
import numpy as np
# from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)   
offset = 20
imgsize = 300
# Try different camera indices
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box coordinates
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            
            for landmark in handLms.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Add padding to the rectangle
            x_min = max(0, x_min - offset)
            y_min = max(0, y_min - offset)
            x_max = min(frame.shape[1], x_max + offset)
            y_max = min(frame.shape[0], y_max + offset)
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            cv2.imshow('Cropped Frame', cropped_frame)
            newImage = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            newImage[offset:offset+cropped_frame.shape[0], offset:offset+cropped_frame.shape[1]] = cropped_frame
            cv2.imshow('New Image', newImage)
            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if results.multi_hand_landmarks:
                cv2.putText(frame, f"Number of hands: {len(results.multi_hand_landmarks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


