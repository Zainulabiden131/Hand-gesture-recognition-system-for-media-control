import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import numpy as np
import threading
import time

# Initialize MediaPipe Hands and Pycaw for audio control
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Setup pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Initialize webcam with reduced resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame skipping
frame_skip = 2  # Process every 2nd frame
frame_count = 0

# Gesture detection variables
is_muted = False
prev_y = None
prev_x = None
last_gesture_time = time.time()
gesture_cooldown = 1.0  # 1-second cooldown

# Function to measure distance between two landmarks
def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Volume and media control functions
def increase_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(min(current_volume + 0.1, 1.0), None)
    print("Volume Increased")

def decrease_volume():
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume.SetMasterVolumeLevelScalar(max(current_volume - 0.1, 0.0), None)
    print("Volume Decreased")

def toggle_mute():
    global is_muted
    is_muted = not is_muted
    volume.SetMute(is_muted, None)
    print("Mute Toggled")

def switch_audio_channel():
    print("Audio Channel Switched")

# Prevent rapid gesture triggers
def perform_gesture_action(gesture_function):
    global last_gesture_time
    if time.time() - last_gesture_time > gesture_cooldown:
        gesture_function()
        last_gesture_time = time.time()

# Thread function for gesture recognition
def process_frame(frame):
    global prev_y, prev_x

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Detect mute gesture (thumb touching middle finger)
            if get_distance(thumb_tip, middle_tip) < 0.05:
                perform_gesture_action(toggle_mute)

            # Detect hand movement for volume control
            if prev_y is not None:
                if wrist.y < prev_y - 0.05:
                    perform_gesture_action(increase_volume)
                elif wrist.y > prev_y + 0.05:
                    perform_gesture_action(decrease_volume)
            prev_y = wrist.y

            # Detect hand movement for channel switching
            if prev_x is not None and abs(wrist.x - prev_x) > 0.1:
                perform_gesture_action(switch_audio_channel)
            prev_x = wrist.x

# Main loop for capturing video and running detection
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to save processing time

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction

    # Run gesture detection in a separate thread
    thread = threading.Thread(target=process_frame, args=(frame,))
    thread.start()

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
