import cv2
import numpy as np
# from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
from pipeline import model_pipeline

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# def infer_character(image: np.ndarray) -> str:
#     # Assume model expects (1, 3, 224, 224)
#     # You may need to resize and normalize image here
#     # Return a dummy output for now
#     return "A"
# detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)   
offset = 20
imgsize = 300
# Try different camera indices
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    new_frame = frame.copy()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    predicted_char = ""

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            
            for landmark in handLms.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            x_min = max(0, x_min - offset)
            y_min = max(0, y_min - offset)
            x_max = min(frame.shape[1], x_max + offset)
            y_max = min(frame.shape[0], y_max + offset)
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            cropped_new_frame = new_frame[y_min:y_max, x_min:x_max]
            # cv2.imshow('Cropped Frame', cropped_new_frame)
            # Resize to model input size
            try:
                input_image = cv2.resize(cropped_frame, (imgsize, imgsize))
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                predicted_char = model_pipeline.inference(input_image)
            except Exception as e:
                print(f"Error during prediction: {e}")
                predicted_char = ""

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display prediction
    if predicted_char:
        cv2.putText(frame, f"Predicted: {predicted_char}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 3)

    cv2.imshow('ASL Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
