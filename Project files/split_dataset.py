import os
import shutil
import random

SOURCE_DIR = 'dataset'
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
TEST_DIR = os.path.join(SOURCE_DIR, 'test')
SPLIT_RATIO = 0.8  # 80% train, 20% test

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path) or class_name in ['train', 'test']:
        continue

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in test_images:
        shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))

print("Dataset split complete.")