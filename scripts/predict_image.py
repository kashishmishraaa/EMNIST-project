import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_image

if __name__ == "__main__":
    image_path = "test_image/test_1.jpg"
    predicted = predict_image(image_path)
    print(f"Predicted Label: {predicted}")
