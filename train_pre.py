import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import random
import shutil
from pathlib import Path


# 设置随机种子
random.seed(42)
np.random.seed(42)


# 手动实现数据划分函数
def train_test_split(files, test_size=0.2, random_state=None, shuffle=True):
    if random_state is not None:
        random.seed(random_state)
    if shuffle:
        random.shuffle(files)
    split_idx = int(len(files) * (1 - test_size))
    return files[:split_idx], files[split_idx:]


# 配置路径 - 在这里调整您的数据集路径
# ================================ 修改这里 ================================
BASE_DATA_PATH = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG"  # 您的原始数据集路径
IMAGES_DIR = os.path.join(BASE_DATA_PATH, "images")  # 原始图像文件夹
BBOX_DIR = os.path.join(BASE_DATA_PATH, "bbox")      # 标注文件夹


WORKING_DIR = "/kaggle/working"  # 工作目录
DATASET_DIR = os.path.join(WORKING_DIR, "dataset")  # 处理后的数据集目录
# =========================================================================


# 检查图像文件是否有效
def is_valid_image(img_path):
    try:
        # 尝试打开图像
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        # 检查图像尺寸是否有效
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False
            
        return True
    except Exception as e:
        print(f"检查图像 {img_path} 时出错: {e}")
        return False


# 创建目录结构
def create_directory_structure():
    os.makedirs(os.path.join(DATASET_DIR, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, 'labels/val'), exist_ok=True)


# 转换CSV标注为YOLO格式
def convert_csv_to_yolo(csv_path, img_path, output_label_path):
    try:
        # 读取图像获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return False
        
        img_height, img_width = img.shape[:2]
        
        # 手动解析CSV文件
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # 跳过标题行，处理数据行
        if len(lines) < 2:
            print(f"CSV文件内容不足: {csv_path}")
            return False
        
        # 创建YOLO格式的标签文件
        with open(output_label_path, 'w') as f_out:
            # 处理所有数据行（从第二行开始）
            for i in range(1, len(lines)):
                data_line = lines[i].strip()
                if not data_line:  # 跳过空行
                    continue
                    
                parts = data_line.split(',')
                if len(parts) != 5:
                    print(f"CSV格式错误，期望5列，实际{len(parts)}列: {csv_path}")
                    continue
                
                class_name, xmin, ymin, xmax, ymax = parts
                
                # 转换坐标值为整数
                try:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                except ValueError as e:
                    print(f"坐标值转换错误 {csv_path}: {e}")
                    continue
                
                # 转换为YOLO格式 (归一化的中心坐标和宽高)
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # 确保坐标在有效范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # 类别ID (这里假设只有息肉一个类别，所以为0)
                class_id = 0
                
                f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"处理CSV文件时出错 {csv_path}: {e}")
        return False


# 检查数据集结构
def check_dataset_structure():
    """检查数据集目录结构"""
    print("检查数据集结构...")
    
    # 检查images目录
    if not os.path.exists(IMAGES_DIR):
        print(f"错误: 图像目录不存在: {IMAGES_DIR}")
        return False
    
    # 检查bbox目录
    if not os.path.exists(BBOX_DIR):
        print(f"错误: 标注目录不存在: {BBOX_DIR}")
        return False
    
    # 列出一些文件示例
    image_files = glob.glob(os.path.join(IMAGES_DIR, '*'))
    csv_files = glob.glob(os.path.join(BBOX_DIR, '*.csv'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"找到 {len(csv_files)} 个CSV标注文件")
    
    # 显示前几个文件
    if image_files:
        print("图像文件示例:")
        for img in image_files[:3]:
            print(f"  {img}")
    
    if csv_files:
        print("CSV文件示例:")
        for csv in csv_files[:3]:
            print(f"  {csv}")
    
    return len(image_files) > 0 and len(csv_files) > 0


# 准备数据集
def prepare_dataset():
    # 先检查数据集结构
    if not check_dataset_structure():
        print("数据集结构检查失败，请检查路径和文件")
        return 0, 0
    
    create_directory_structure()
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print("错误: 没有找到任何图像文件")
        return 0, 0
    
    # 过滤无效图像
    valid_image_files = []
    for img_path in image_files:
        if is_valid_image(img_path):
            valid_image_files.append(img_path)
        else:
            print(f"跳过无效图像: {img_path}")
    
    print(f"有效图像文件: {len(valid_image_files)}/{len(image_files)}")
    
    if len(valid_image_files) == 0:
        print("错误: 没有找到有效图像文件")
        return 0, 0
    
    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        valid_image_files, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"训练集样本数: {len(train_files)}")
    print(f"验证集样本数: {len(val_files)}")
    
    # 处理训练集
    train_success = 0
    for img_path in train_files:
        try:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            csv_path = os.path.join(BBOX_DIR, f"{filename}.csv")
            
            if os.path.exists(csv_path):
                # 复制图像到训练目录
                dest_img_path = os.path.join(DATASET_DIR, 'images/train', os.path.basename(img_path))
                
                # 确保目录存在
                os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
                
                # 使用shutil.copy2复制文件，并检查是否成功
                shutil.copy2(img_path, dest_img_path)
                
                # 验证文件是否正确复制
                if not os.path.exists(dest_img_path) or os.path.getsize(dest_img_path) == 0:
                    print(f"警告: 图像复制失败 {img_path} -> {dest_img_path}")
                    continue
                
                # 转换标签
                label_path = os.path.join(DATASET_DIR, 'labels/train', f"{filename}.txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                
                if convert_csv_to_yolo(csv_path, img_path, label_path):
                    train_success += 1
                else:
                    print(f"转换失败: {csv_path}")
            else:
                print(f"警告: 找不到对应的CSV文件: {csv_path}")
        except Exception as e:
            print(f"处理训练图像 {img_path} 时出错: {e}")
    
    # 处理验证集
    val_success = 0
    for img_path in val_files:
        try:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            csv_path = os.path.join(BBOX_DIR, f"{filename}.csv")
            
            if os.path.exists(csv_path):
                # 复制图像到验证目录
                dest_img_path = os.path.join(DATASET_DIR, 'images/val', os.path.basename(img_path))
                
                # 确保目录存在
                os.makedirs(os.path.dirname(dest_img_path), exist_ok=True)
                
                # 复制文件
                shutil.copy2(img_path, dest_img_path)
                
                # 验证文件是否正确复制
                if not os.path.exists(dest_img_path) or os.path.getsize(dest_img_path) == 0:
                    print(f"警告: 图像复制失败 {img_path} -> {dest_img_path}")
                    continue
                
                # 转换标签
                label_path = os.path.join(DATASET_DIR, 'labels/val', f"{filename}.txt")
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                
                if convert_csv_to_yolo(csv_path, img_path, label_path):
                    val_success += 1
                else:
                    print(f"转换失败: {csv_path}")
            else:
                print(f"警告: 找不到对应的CSV文件: {csv_path}")
        except Exception as e:
            print(f"处理验证图像 {img_path} 时出错: {e}")
    
    print(f"成功处理训练样本: {train_success}/{len(train_files)}")
    print(f"成功处理验证样本: {val_success}/{len(val_files)}")
    
    # 创建data.yaml文件，使用绝对路径
    absolute_path = os.path.abspath(DATASET_DIR)
    yaml_content = f"""path: {absolute_path}
train: images/train
val: images/val


nc: 1
names: ['polyp']
"""
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"创建YAML配置文件: {yaml_path}")
    print(f"YAML中使用的数据集路径: {absolute_path}")
    
    # 验证数据集目录中的文件
    train_images_count = len(glob.glob(os.path.join(DATASET_DIR, 'images/train/*')))
    val_images_count = len(glob.glob(os.path.join(DATASET_DIR, 'images/val/*')))
    train_labels_count = len(glob.glob(os.path.join(DATASET_DIR, 'labels/train/*')))
    val_labels_count = len(glob.glob(os.path.join(DATASET_DIR, 'labels/val/*')))
    
    print(f"数据集目录验证:")
    print(f"  训练图像: {train_images_count}")
    print(f"  验证图像: {val_images_count}")
    print(f"  训练标签: {train_labels_count}")
    print(f"  验证标签: {val_labels_count}")
    
    # 检查是否有图像和标签不匹配的情况
    if train_images_count != train_labels_count:
        print(f"警告: 训练图像和标签数量不匹配 ({train_images_count} vs {train_labels_count})")
    if val_images_count != val_labels_count:
        print(f"警告: 验证图像和标签数量不匹配 ({val_images_count} vs {val_labels_count})")
    
    return train_success, val_success


# 可视化预测结果
def visualize_predictions(model, val_images_dir, val_labels_dir, num_samples=5):
    # 获取验证集图像文件
    val_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        val_images.extend(glob.glob(os.path.join(val_images_dir, ext)))
    
    if not val_images:
        print(f"在 {val_images_dir} 中没有找到验证图像")
        return
    
    # 随机选择样本
    selected_images = random.sample(val_images, min(num_samples, len(val_images)))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, img_path in enumerate(selected_images):
        # 检查图像是否存在且有效
        if not os.path.exists(img_path) or not is_valid_image(img_path):
            print(f"跳过无效图像: {img_path}")
            continue
            
        # 获取对应的标签文件
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_labels_dir, f"{filename}.txt")
        
        # 读取图像
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # 绘制真实标签
        ax1 = axes[0, i]
        ax1.imshow(img_rgb)
        ax1.set_title(f'真实标签: {filename}')
        ax1.axis('off')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # 转换回原始坐标
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    # 绘制边界框
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, color='green', linewidth=2)
                    ax1.add_patch(rect)
        
        # 模型预测
        ax2 = axes[1, i]
        ax2.imshow(img_rgb)
        ax2.set_title(f'预测结果: {filename}')
        ax2.axis('off')
        
        results = model.predict(img_path, conf=0.25)
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color='red', linewidth=2)
                ax2.add_patch(rect)
                ax2.text(x1, y1-10, f'polyp: {conf:.2f}', 
                        bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.7),
                        color='white', fontsize=8)
    
    plt.tight_layout()
    plt.show()


# 主训练函数
def main():
    print("开始准备数据集...")
    train_count, val_count = prepare_dataset()
    
    print(f"数据集准备完成!")
    print(f"训练样本: {train_count}")
    print(f"验证样本: {val_count}")
    
    if train_count == 0 or val_count == 0:
        print("错误: 没有成功处理任何样本，请检查数据路径")
        return
    
    # 加载预训练模型
    print("加载YOLOv8模型...")
    model = YOLO('yolov8n.pt')
    
    # 训练参数配置
    training_args = {
        'data': os.path.join(DATASET_DIR, 'data.yaml'),
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'patience': 15,  # 早停机制
        'save': True,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'close_mosaic': 10,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'device': 0,  # 使用GPU
        'workers': 4,
        'project': os.path.join(WORKING_DIR, 'polyp_detection'),
        'name': 'exp',
        'verbose': True,
        'amp': False  # 关闭AMP训练
    }
    
    print("开始训练...")
    print(f"使用的数据配置: {training_args['data']}")
    
    try:
        # 开始训练
        results = model.train(** training_args)
        print("训练完成!")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 打印更多调试信息
        print("数据配置文件内容:")
        with open(training_args['data'], 'r') as f:
            print(f.read())
        return
    
    # 加载最佳模型
    best_model_path = os.path.join(WORKING_DIR, 'polyp_detection/exp/weights/best.pt')
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        print(f"加载最佳模型: {best_model_path}")
    else:
        best_model = model
        print("使用最终模型进行预测")
    
    # 可视化预测结果
    print("进行验证集预测可视化...")
    visualize_predictions(
        best_model, 
        os.path.join(DATASET_DIR, 'images/val'), 
        os.path.join(DATASET_DIR, 'labels/val'), 
        num_samples=5
    )
    
    # 评估模型
    print("评估模型性能...")
    try:
        metrics = best_model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"精确率: {metrics.box.precision:.4f}")
        print(f"召回率: {metrics.box.recall:.4f}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")


if __name__ == "__main__":
    main()
