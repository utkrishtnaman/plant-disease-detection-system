import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.keras")

# Load class names
with open("model/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

IMAGE_SIZE = (224, 224)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(image: Image.Image):
    img = preprocess_image(image)
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    return {
        "disease": class_names[idx],
        "confidence": float(preds[idx])
    }
