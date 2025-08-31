import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize MediaPipe Hands and Pycaw for audio control
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Setup pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Flag to track mute status
is_muted = False

# Functions to control volume
def increase_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = min(current_volume + 0.1, 1.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print("Volume Increased")

def decrease_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    new_volume = max(current_volume - 0.1, 0.0)
    volume.SetMasterVolumeLevelScalar(new_volume, None)
    print("Volume Decreased")

def toggle_mute():
    global is_muted
    is_muted = not is_muted
    volume.SetMute(is_muted, None)
    print("Mute Toggled")

# Main loop for detecting gestures
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check if any hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Convert landmark positions to 2D coordinates
                thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
                index_tip_x, index_tip_y = index_tip.x, index_tip.y
                middle_tip_x, middle_tip_y = middle_tip.x, middle_tip.y
                pinky_tip_x, pinky_tip_y = pinky_tip.x, pinky_tip.y

                # Detect "Mute" gesture (thumb and index finger tips touching)
                distance_thumb_index = np.sqrt((index_tip_x - thumb_tip_x)**2 + (index_tip_y - thumb_tip_y)**2)
                if distance_thumb_index < 0.05:  # Adjust threshold as needed
                    toggle_mute()

                # Detect "Increase Volume" gesture (open hand with fingers stretched)
                distance_thumb_pinky = np.sqrt((pinky_tip_x - thumb_tip_x)**2 + (pinky_tip_y - thumb_tip_y)**2)
                if distance_thumb_pinky > 0.3:  # Open hand threshold
                    increase_volume()

                # Detect "Decrease Volume" gesture (e.g., L shape with thumb and index fingers)
                distance_index_middle = np.sqrt((index_tip_x - middle_tip_x)**2 + (index_tip_y - middle_tip_y)**2)
                if distance_index_middle < 0.05 and distance_thumb_index > 0.2:  # L shape threshold
                    decrease_volume()

        # Display the video feed
        cv2.imshow("Hand Gesture Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
