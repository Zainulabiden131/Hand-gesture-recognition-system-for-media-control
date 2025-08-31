import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Definitions
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    wrist = landmarks[0]

    # Gesture: Swipe Right (Channel Forward)
    if index_tip.x > wrist.x + 0.3:
        return "forward"

    # Gesture: Swipe Left (Channel Backward)
    elif index_tip.x < wrist.x - 0.3:
        return "backward"

    return None

# Video Capture
cap = cv2.VideoCapture(0)

print("Starting Real-Time Hand Gesture Recognition for Media Control...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture using landmarks
            gesture = detect_gesture(hand_landmarks.landmark)
            if gesture == "forward":
                pyautogui.press("right")  # Simulate channel forward key press
                cv2.putText(frame, "Channel Forward", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif gesture == "backward":
                pyautogui.press("left")  # Simulate channel backward key press
                cv2.putText(frame, "Channel Backward", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
