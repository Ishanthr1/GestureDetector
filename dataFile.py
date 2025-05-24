import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define your gesture label here
GESTURE_LABEL = "STOP"  # Change this for each gesture you collect

# Output CSV
CSV_FILE = "gesture_data.csv"

def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

# Start webcam and collect data
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

data = []

print("Starting capture. Press 's' to save frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to make it mirror-like
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                row = extract_landmarks(hand_landmarks)
                row.append(GESTURE_LABEL)
                data.append(row)
                print(f"Saved frame with label: {GESTURE_LABEL}")

    cv2.imshow('Hand Landmark Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
if data:
    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, mode='a', header=not pd.read_csv(CSV_FILE).empty if pd.io.common.file_exists(CSV_FILE) else True, index=False)
    print(f"Saved {len(data)} samples to {CSV_FILE}")
