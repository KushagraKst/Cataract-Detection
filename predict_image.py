import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

print("Loading model...")

model = load_model("cataract_model.h5")
class_indices = pickle.load(open("class_indices.pkl", "rb"))

labels = {v: k for k, v in class_indices.items()}

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Image not found!")
        print("👉 Use path like: dataset/label/img.png")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    pred = model.predict(img)
    print("Raw prediction:", pred)

    return labels[np.argmax(pred)]

image_path = input("Enter image path: ")

result = predict_image(image_path)

print("Prediction:", result)