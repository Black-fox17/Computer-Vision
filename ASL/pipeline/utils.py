import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(img_path):
    frame = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize Hands with static image mode
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        # Process the image and get the result
        results = hands.process(image_rgb)
    
        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on original image
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    )
    return frame
