import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Detection configuration: 70% confidence to avoid false positives
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Video capture
cap = cv2.VideoCapture(0)

def calculate_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)

print("System starting... Press 'q' to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert color from BGR (OpenCV) to RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect hands
    results = hands.process(image)

    # Convert back to BGR for display purposes
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand "skeletons"
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get key point coordinates (Landmarks)
            # 4 = Thumb Tip, 8 = Index Finger Tip
            h, w, c = image.shape
            x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

            # Calculate the center of the line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Graphical
            cv2.circle(image, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Thumb
            cv2.circle(image, (x2, y2), 10, (255, 0, 255), cv2.FILLED) # Index
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)      # Connection line
            cv2.circle(image, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # Center

            # Mathematics
            # Calculate line length (Euclidean Distance)
            length = calculate_distance(x1, y1, x2, y2)

            # Range mapping (Linear Interpolation)
            # Hand range (approx): 30 to 250 pixels -> Volume Range: 0 to 100%
            vol_percentage = np.interp(length, [30, 250], [0, 100])
            
            # Visual feedback (Volume bar)
            cv2.rectangle(image, (50, 150), (85, 400), (0, 255, 0), 3)
            vol_bar = np.interp(length, [30, 250], [400, 150]) # Mapping for the graphical bar
            cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            
            cv2.putText(image, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 3)

    cv2.imshow('Smart Engineering - Gesture Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
