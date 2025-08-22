import os
import cv2
import mediapipe as mp
import pandas as pd

# Paths
IMAGE_FOLDER = "asl_alphabet_train"
OUTPUT_CSV = "gesture_data_from_images.csv"

mp_hands = mp.solutions.hands

def extract_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

all_data = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for label in os.listdir(IMAGE_FOLDER):
        label_path = os.path.join(IMAGE_FOLDER, label)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(label_path, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load image: {img_path}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                landmarks = extract_landmarks(results.multi_hand_landmarks[0])
                landmarks.append(label)
                all_data.append(landmarks)
            else:
                print(f"No hands detected in: {img_path}")

# Save to CSV
if all_data:
    num_features = len(all_data[0]) - 1
    columns = [f"x{i}" for i in range(num_features)] + ["label"]
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(all_data)} samples to {OUTPUT_CSV}")
else:
    print("No valid hand landmarks extracted.")
