import os
import shutil

src = 'dataset1'
dst = 'dataset'

for class_name in os.listdir(src):
    class_src = os.path.join(src, class_name)
    class_dst = os.path.join(dst, class_name)
    if os.path.isdir(class_src):
        shutil.move(class_src, class_dst)
print("All class folders moved from dataset1 to dataset.")