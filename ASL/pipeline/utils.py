import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(image):
    # If input is a string, treat it as a file path
    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            raise ValueError(f"Could not read image from path: {image}")
    # If input is already an image array
    elif isinstance(image, np.ndarray):
        frame = image.copy()
    else:
        raise ValueError("Input must be either a file path or a numpy array")

    # Resize image to 200x200
    frame = cv2.resize(frame, (200, 200))
    
    # Ensure image has 3 channels
    if len(frame.shape) == 2:  # If grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

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
def num_to_char(num):
    vocab = ['', 'N', 'R', 'space', 'B', 'I', 'del', 'F', 'H', 'E', 'U', 'M', 'X', 'K', 'Q', 'Y', 'S', 'G', 'A', 'O', 'T', 'V', 'Z', 'C', 'P', 'L', 'W', 'D', 'nothing', 'J']
    return vocab[num]

print(num_to_char(18))