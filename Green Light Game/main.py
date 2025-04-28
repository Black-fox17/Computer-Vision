import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Traffic light states
GREEN = 0
RED = 1
current_state = GREEN
state_start_time = time.time()
STATE_DURATION = 5  # seconds for each state

# Movement detection parameters
MOVEMENT_THRESHOLD = 0.1  # threshold for detecting movement
previous_landmarks = None

def calculate_movement(current_landmarks, previous_landmarks):
    if previous_landmarks is None:
        return 0
    
    total_movement = 0
    for i in range(len(current_landmarks.landmark)):
        current = current_landmarks.landmark[i]
        previous = previous_landmarks.landmark[i]
        
        # Calculate Euclidean distance between current and previous positions
        movement = np.sqrt((current.x - previous.x)**2 + 
                         (current.y - previous.y)**2 + 
                         (current.z - previous.z)**2)
        total_movement += movement
    
    return total_movement / len(current_landmarks.landmark)

def draw_traffic_light(img, state):
    # Draw traffic light circle
    center = (50, 50)
    radius = 20
    if state == GREEN:
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red
    
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius, (255, 255, 255), 2)  # White border

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    img = cv2.flip(img, 1)
    
    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect pose
    results = pose.process(img_rgb)
    
    # Check if state should change
    current_time = time.time()
    if current_time - state_start_time >= STATE_DURATION:
        current_state = RED if current_state == GREEN else GREEN
        state_start_time = current_time
    
    # Draw pose landmarks and handle movement detection
    if results.pose_landmarks:
        # Calculate movement
        movement = calculate_movement(results.pose_landmarks, previous_landmarks)
        previous_landmarks = results.pose_landmarks
        
        # Draw landmarks with appropriate color based on state and movement
        if current_state == GREEN:
            # Green light - allow movement
            landmark_color = (245, 117, 66)  # Orange
            connection_color = (245, 66, 230)  # Pink
        else:
            # Red light - check for movement
            if movement > MOVEMENT_THRESHOLD:
                landmark_color = (0, 0, 255)  # Red - movement detected
                connection_color = (0, 0, 255)
            else:
                landmark_color = (0, 255, 0)  # Green - good posture
                connection_color = (0, 255, 0)
        
        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=connection_color, thickness=2, circle_radius=2)
        )
        
        # Display movement value
        cv2.putText(img, f"Movement: {movement:.3f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw traffic light
    draw_traffic_light(img, current_state)
    
    # Display state text
    state_text = "GREEN - Move Freely" if current_state == GREEN else "RED - Stay Still"
    cv2.putText(img, state_text, (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the image
    cv2.imshow("Pose Detection", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
