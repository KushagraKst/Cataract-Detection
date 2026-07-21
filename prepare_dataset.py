import numpy as np
import pandas as pd
import cv2
import os
import shutil

print("Loading dataset...")

# ✅ Load dataset FIRST
df = pd.read_parquet("cataract-training-dataset.parquet")

# ✅ Debug info
print("Dataset size:", len(df))
print("Columns:", df.columns)

# 🔥 Safe sampling
LIMIT = 1000

if len(df) == 0:
    print("❌ Dataset is empty! Check file path.")
    exit()

elif len(df) <= LIMIT:
    print(f"Using full dataset: {len(df)} rows")
else:
    print(f"Sampling {LIMIT} rows...")
    df = df.sample(n=LIMIT, random_state=42)

base_path = "dataset"

# 🔥 Reset dataset folder
if os.path.exists(base_path):
    shutil.rmtree(base_path)

print("Creating folders...")

for label in df['label'].unique():
    os.makedirs(os.path.join(base_path, label), exist_ok=True)

def vector_to_image(vec):
    try:
        arr = np.array(vec.split(','), dtype=np.uint8)

        if len(arr) != 512 * 512 * 3:
            return None

        img = arr.reshape(512, 512, 3)

        # Convert BGR → RGB
        img = img[:, :, ::-1]

        # Resize
        img = cv2.resize(img, (224, 224))

        return img

    except:
        return None

print("Saving images...")

count = 0

for i, row in df.iterrows():
    img = vector_to_image(row['image_vector'])

    if img is not None:
        label = row['label']
        path = os.path.join(base_path, label, f"img_{i}.png")

        cv2.imwrite(path, img)
        count += 1

        if count % 100 == 0:
            print(f"Saved {count} images...")

print(f"✅ Dataset ready! Total images: {count}")