import os
import random
import shutil

# 设置数据集目录
data_dir = "/home/ps/Public/HuiWei/Project/Benchmark/data/all_data/all/"
image_dir = os.path.join(data_dir, "images")
label_dir = os.path.join(data_dir, "labels")

# 设置训练集和验证集目录
dataset_dir = "/home/ps/Public/HuiWei/Project/Benchmark/data/ScalePerson/"
train_dir_images = os.path.join(dataset_dir, "images/train")
val_dir_images = os.path.join(dataset_dir, "images/val")
train_dir_labels = os.path.join(dataset_dir, "labels/train")
val_dir_labels = os.path.join(dataset_dir, "labels/val")

# 创建训练集和验证集目录
os.makedirs(train_dir_images, exist_ok=True)
os.makedirs(val_dir_images, exist_ok=True)
os.makedirs(train_dir_labels, exist_ok=True)
os.makedirs(val_dir_labels, exist_ok=True)

# 获取图像文件列表
image_files = os.listdir(image_dir)
# 随机打乱文件列表
random.shuffle(image_files)

# 计算验证集的数量
val_size = int(0.2 * len(image_files))

# 将数据集划分为训练集和验证集
train_files = image_files[val_size:]
val_files = image_files[:val_size]

# 将图像文件和对应的标注文件复制到相应的目录
for file in train_files:
    image_path = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))  # 假设标注文件以.txt格式存储
    shutil.copy(image_path, os.path.join(train_dir_images, file))
    shutil.copy(label_path, os.path.join(train_dir_labels, file.replace(".jpg", ".txt")))

for file in val_files:
    image_path = os.path.join(image_dir, file)
    label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))  # 假设标注文件以.txt格式存储
    shutil.copy(image_path, os.path.join(val_dir_images, file))
    shutil.copy(label_path, os.path.join(val_dir_labels, file.replace(".jpg", ".txt")))

print("数据集划分完成！")
