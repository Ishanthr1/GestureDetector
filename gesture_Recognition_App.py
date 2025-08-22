import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import time

# Load trained model
clf = joblib.load("gesture_model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Function to extract hand landmarks
def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

# Start video capture
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    prediction = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and classify landmarks
            landmarks = extract_landmarks(hand_landmarks)
            if len(landmarks) == 63:
                try:
                    df_input = pd.DataFrame([landmarks], columns=clf.feature_names_in_)
                    prediction = clf.predict(df_input)[0]
                except Exception as e:
                    prediction = f"Error: {e}"

            # Get bounding box
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw box and label
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)
            cv2.putText(frame, f"{prediction}", (x_min, y_min - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Overlay FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show window
    cv2.imshow('ASL Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
