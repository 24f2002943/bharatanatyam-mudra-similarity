import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import urllib.request

# ── Download the hand landmark model if not present ──────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ── Build HandLandmarker ──────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(lm_list):
    """
    lm_list: list of 21 [x, y, z] values
    Returns a flat numpy array of shape (63,) after:
      1. Translating wrist to (0,0,0)
      2. Scaling by distance wrist→middle-finger-MCP (landmark 9)
    """
    lm = np.array(lm_list, dtype=np.float32)   # (21, 3)
    lm -= lm[0]                                  # translate wrist to origin
    scale = np.linalg.norm(lm[9])
    if scale > 0:
        lm /= scale
    return lm.flatten()                          # (63,)


def extract_from_dataset(dataset_path, output_csv, output_json, subset=None):
    data = []
    references = {}

    all_categories = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if subset:
        categories = [c for c in all_categories if any(s.lower() in c.lower() for s in subset)]
    else:
        categories = all_categories

    print(f"Found {len(categories)} mudra categories to process.")

    for category in tqdm(categories, desc="Processing categories"):
        cat_path = os.path.join(dataset_path, category)
        images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        cat_landmarks = []

        for img_name in images:
            img_path = os.path.join(cat_path, img_name)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue

            # MediaPipe Tasks API expects RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                lm_list = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                normalized = normalize_landmarks(lm_list)
                data.append([category] + list(normalized))
                cat_landmarks.append(normalized)

        # Store mean landmarks as "ideal" reference for this mudra
        if cat_landmarks:
            mean_lm = np.mean(cat_landmarks, axis=0).tolist()
            references[category] = {
                "ideal_landmarks": mean_lm,
                "description": f"Standard {category} mudra configuration."
            }

    # Save training CSV
    columns = ['label'] + [f'lm_{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)

    # Save references JSON
    with open(output_json, 'w') as f:
        json.dump(references, f, indent=4)

    print(f"\nExtraction complete!")
    print(f"Saved {len(df)} samples across {df['label'].nunique()} classes → {output_csv}")
    print(f"Saved references → {output_json}")


if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\Tanaya\Downloads\Bharatanatyam-Mudra-Dataset-master\Bharatanatyam-Mudra-Dataset-master"
    OUTPUT_CSV   = r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\mudra_landmarks.csv"
    OUTPUT_JSON  = r"c:\Users\Tanaya\Desktop\Bharatanatyam Mudra Similarity & Feedback System\data\references.json"

    # Process ALL mudras in the dataset (no subset filter)
    extract_from_dataset(DATASET_PATH, OUTPUT_CSV, OUTPUT_JSON, subset=None)
