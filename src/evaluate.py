from tensorflow.keras.models import load_model
from src.data_loader import load_emnist_dataset
from src.config import MODEL_PATH

def evaluate_model():
    _, _, x_test, y_test = load_emnist_dataset()
    model = load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
