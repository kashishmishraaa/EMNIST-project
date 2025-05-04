import numpy as np
from src.data_loader import load_emnist_dataset
from src.model import build_emnist_model
from src.predict import preprocess_image

def test_data_loading():
    x_train, y_train, x_test, y_test = load_emnist_dataset()
    assert x_train.shape[1:] == (64, 64, 1), "Incorrect train image shape"
    assert y_train.shape[1] == 47, "Incorrect number of classes"

def test_model_building():
    model = build_emnist_model()
    assert model.input_shape == (None, 64, 64, 1), "Model input shape mismatch"
    assert model.output_shape == (None, 47), "Model output shape mismatch"

def test_image_preprocessing():
    try:
        image_path = "data/sample_digit.png"  # Ensure this test image exists
        img = preprocess_image(image_path)
        assert img.shape == (1, 64, 64, 1), "Preprocessed image shape incorrect"
    except FileNotFoundError:
        print("⚠️ Test image not found: Skipping image preprocess test.")
