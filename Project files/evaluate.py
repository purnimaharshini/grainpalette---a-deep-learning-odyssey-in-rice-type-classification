from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def evaluate_model(model=None):
    if model is None:
        model = load_model("model/rice_model.h5")

    # Get class names from train directory to ensure same order
    train_dir = 'dataset/train/'
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'dataset/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        classes=class_names  # Ensures same class order as training
    )

    print("Starting evaluation...")
    loss, accuracy = model.evaluate(test_generator)
    print("Evaluation finished.")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")