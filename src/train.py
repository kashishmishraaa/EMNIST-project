from tensorflow.keras.callbacks import EarlyStopping
from src.data_loader import load_emnist_dataset
from src.model import build_emnist_model
from src.config import MODEL_PATH, EPOCHS, BATCH_SIZE

def train_model():
    x_train, y_train, x_test, y_test = load_emnist_dataset()
    model = build_emnist_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [EarlyStopping(patience=3, restore_best_weights=True)]
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    model.save(MODEL_PATH)
    return model, history
