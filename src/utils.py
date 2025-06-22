import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 150
LABELS = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']

def load_keras_model(path):
    """
    Load and return a Keras model from the given path.
    """
    return load_model(path)

def preprocess_image(image_bytes):
    """
    Convert uploaded image bytes to a resized normalized tensor.
    """
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_prediction(model, image_tensor):
    """
    Generate prediction and return label and confidence.
    """
    prediction = model.predict(image_tensor)[0]
    predicted_index = np.argmax(prediction)
    label = LABELS[predicted_index]
    confidence = prediction[predicted_index] * 100
    return label, confidence
