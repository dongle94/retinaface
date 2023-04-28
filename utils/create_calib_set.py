import os
import shutil

cur_path = os.path.dirname(os.path.abspath(__file__))

trainset_path = os.path.join(cur_path, '../data/retinaface/train/images')
calibset_path = os.path.join(cur_path, '../data/retinaface/calib/images')

if not os.path.exists(calibset_path):
    os.makedirs(calibset_path)

for image_dir in os.listdir(trainset_path):
    train_image_dir = os.path.join(trainset_path, image_dir)
    for img in os.listdir(train_image_dir):
        shutil.copyfile(os.path.join(train_image_dir, img), calibset_path + '/' + img)