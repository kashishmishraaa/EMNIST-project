import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from src.config import MODEL_PATH, IMG_SIZE
from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_image(img_path):
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize if model was trained with normalized data
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 28, 28, 1)
    return img_array

def predict_image(image_path):
    model = load_model('models/model4.h5')
    image_input = preprocess_image(image_path)
    prediction = model.predict(image_input)
    return np.argmax(prediction)
