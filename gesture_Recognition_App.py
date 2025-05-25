import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

clf = joblib.load("gesture_model.pkl")

def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

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
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = extract_landmarks(hand_landmarks)
            if len(landmarks) == 63:
                try:
                    df_input = pd.DataFrame([landmarks], columns=clf.feature_names_in_)
                    prediction = clf.predict(df_input)[0]
                except Exception as e:
                    prediction = f"Error: {e}"

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Display result
    cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Real-Time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()