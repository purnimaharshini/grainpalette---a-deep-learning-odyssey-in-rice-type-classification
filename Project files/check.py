import os

train_dir = 'dataset/train'
test_dir = 'dataset/test'

train_classes = set(os.listdir(train_dir))
test_classes = set(os.listdir(test_dir))

print("Train classes:", train_classes)
print("Test classes:", test_classes)
print("Classes only in train:", train_classes - test_classes)
print("Classes only in test:", test_classes - train_classes)