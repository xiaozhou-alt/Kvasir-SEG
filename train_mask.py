import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import requests
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findSystemFonts
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import cv2

# 设置中文字体（保持不变）
def setup_chinese_font():
    """设置中文字体，确保中文正常显示"""
    font_options = [
        {
            "name": "SimHei",
            "system_paths": [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/simhei/SimHei.ttf",
                "/Library/Fonts/SimHei.ttf",
                "C:/Windows/Fonts/simhei.ttf"
            ],
            "urls": [
                "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf",
                "https://gitee.com/chenguanzhou/simhei/raw/master/SimHei.ttf"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "SimHei.ttf")
        },
        {
            "name": "Noto Sans CJK SC",
            "system_paths": [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            ],
            "urls": [
                "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/NotoSansCJKsc-Regular.otf"
            ],
            "local_path": os.path.join("/kaggle/working/fonts", "NotoSansCJKsc-Regular.otf")
        }
    ]
    
    system_fonts = findSystemFonts()
    
    for font in font_options:
        font_path = None
        
        for path in font["system_paths"]:
            if os.path.exists(path):
                font_path = path
                break
        
        if not font_path:
            for path in system_fonts:
                try:
                    if font["name"].lower() in path.lower():
                        font_path = path
                        break
                except:
                    continue
        
        if font_path:
            try:
                font_prop = FontProperties(fname=font_path)
                plt.text(0, 0, "测试", fontproperties=font_prop)
                plt.close()
                
                print(f"成功加载中文字体: {font['name']} ({font_path})")
                
                plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                plt.rcParams['axes.unicode_minus'] = False
                return font_prop
            except:
                print(f"字体 {font['name']} 存在但无法使用，尝试下一种...")
                continue
        
        if 'urls' in font and font["urls"]:
            try:
                font_dir = os.path.dirname(font["local_path"])
                Path(font_dir).mkdir(parents=True, exist_ok=True)
                
                downloaded = False
                for url in font["urls"]:
                    for attempt in range(3):
                        try:
                            print(f"系统中未找到{font['name']}字体，正在从 {url} 下载（尝试 {attempt+1}/3）...")
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            response = requests.get(url, headers=headers, timeout=60)
                            response.raise_for_status()
                            
                            with open(font["local_path"], "wb") as f:
                                f.write(response.content)
                            
                            if os.path.getsize(font["local_path"]) < 1024 * 100:
                                raise Exception("下载的字体文件不完整")
                                
                            downloaded = True
                            break
                        except Exception as e:
                            print(f"从 {url} 下载{font['name']}失败（尝试 {attempt+1}/3）: {str(e)}")
                            if attempt == 2:
                                continue
                            time.sleep(2)
                    if downloaded:
                        break
                
                if not downloaded:
                    raise Exception("所有下载链接都失败了")
                
                font_prop = FontProperties(fname=font["local_path"])
                print(f"成功下载并加载{font['name']}字体: {font['local_path']}")
                
                plt.rcParams["font.family"] = [font["name"], "sans-serif"]
                plt.rcParams["font.sans-serif"] = [font["name"], "sans-serif"]
                plt.rcParams['axes.unicode_minus'] = False
                return font_prop
            except Exception as e:
                print(f"加载或下载{font['name']}字体失败: {str(e)}")
                continue
    
    try:
        print("尝试使用Kaggle预装的Noto字体...")
        noto_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if os.path.exists(noto_font_path):
            font_prop = FontProperties(fname=noto_font_path)
            plt.rcParams["font.family"] = ["Noto Sans CJK SC", "sans-serif"]
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "sans-serif"]
            plt.rcParams['axes.unicode_minus'] = False
            print("成功使用Noto Sans CJK SC字体")
            return font_prop
    except:
        pass
    
    print("警告: 所有中文字体加载失败，中文可能无法正常显示")
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams['axes.unicode_minus'] = False
    return FontProperties()


# 测试中文字体显示（保持不变）
def test_chinese_font_display(chinese_font):
    plt.figure(figsize=(8, 4))
    plt.title("中文字体测试 - 胃息肉分割系统", fontproperties=chinese_font)
    plt.xlabel("X轴标签（中文测试）", fontproperties=chinese_font)
    plt.ylabel("Y轴标签（中文测试）", fontproperties=chinese_font)
    plt.text(0.5, 0.5, "测试文字：胃息肉分割 ABC123", 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=12, fontproperties=chinese_font)
    plt.plot([1, 2, 3], [4, 5, 6], label="示例曲线")
    
    plt.legend(
        title="图例（中文）", 
        prop=chinese_font,
        title_fontproperties=chinese_font
    )
    
    test_img_path = os.path.join('/kaggle/working/', 'chinese_font_test.png')
    plt.savefig(test_img_path, dpi=300)
    print(f"中文字体测试图像已保存至: {test_img_path}")
    plt.close()
    
    return test_img_path

# 评估指标计算函数（保持不变）
def calculate_metrics(pred, target, threshold=0.5):
    """
    计算分割任务的评估指标
    pred: 模型预测输出 (经过sigmoid)
    target: 真实标签
    threshold: 二值化阈值
    """
    # 将预测值二值化
    pred = (pred > threshold).float()
    
    # 计算TP, TN, FP, FN
    TP = (pred * target).sum()
    TN = ((1 - pred) * (1 - target)).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()
    
    # 计算精确率
    precision = TP / (TP + FP + 1e-8)  # 加小值避免除零
    
    # 计算召回率
    recall = TP / (TP + FN + 1e-8)
    
    # 计算Dice系数
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    
    # 计算IoU (交并比)
    iou = TP / (TP + FP + FN + 1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'dice': dice.item(),
        'iou': iou.item()
    }

# 自定义Dice损失函数（保持不变）
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, input, target):
        input = torch.sigmoid(input)
        intersection = (input * target).sum()
        return 1 - (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)

# 混合损失函数（交叉熵+Dice）（保持不变）
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        dice = self.dice_loss(input, target)
        return self.bce_weight * bce + self.dice_weight * dice

# 增强的数据增强策略
class RandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_range=(0.5, 2.0)):
        self.p = p
        self.radius_range = radius_range
        
    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

# 修复的弹性变换类 - 解决维度不匹配问题
class RandomElasticTransform(object):
    def __init__(self, p=0.4, alpha=120, sigma=15):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, img):
        if random.random() < self.p:
            img = np.array(img)
            # 获取图像尺寸（高度、宽度），忽略通道数
            h, w = img.shape[:2]
            
            # 创建2D位移场（与图像尺寸匹配）
            dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), 
                                 (0, 0), self.sigma) * self.alpha
            dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), 
                                 (0, 0), self.sigma) * self.alpha
            
            # 创建坐标网格
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            # 对每个通道应用相同的变换
            if len(img.shape) == 3:  # 彩色图像
                transformed = np.zeros_like(img)
                for c in range(img.shape[2]):
                    transformed[:, :, c] = cv2.remap(
                        img[:, :, c], map_x, map_y, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT
                    )
                return Image.fromarray(transformed)
            else:  # 灰度图像
                img = cv2.remap(
                    img, map_x, map_y, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_REFLECT
                )
                return Image.fromarray(img)
        return img

class RandomZoom(object):
    def __init__(self, p=0.5, zoom_range=(0.8, 1.2)):
        self.p = p
        self.zoom_range = zoom_range
        
    def __call__(self, img):
        if random.random() < self.p:
            zoom = random.uniform(*self.zoom_range)
            w, h = img.size
            new_w, new_h = int(w * zoom), int(h * zoom)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            # 如果缩小了，随机裁剪回原尺寸
            if zoom < 1.0:
                x = random.randint(0, new_w - w) if new_w > w else 0
                y = random.randint(0, new_h - h) if new_h > h else 0
                img = img.crop((x, y, x + w, y + h))
            # 如果放大了，中心裁剪
            elif zoom > 1.0:
                x = (new_w - w) // 2
                y = (new_h - h) // 2
                img = img.crop((x, y, x + w, y + h))
        return img

# 设置随机种子（保持不变）
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 检查GPU是否可用（保持不变）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据集类定义（保持不变）
class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
        # 数据重复因子，小样本时增加训练次数
        self.repeat = 2 if is_train and len(image_paths) < 1500 else 1
        
    def __len__(self):
        return len(self.image_paths) * self.repeat
    
    def __getitem__(self, idx):
        # 处理重复索引
        idx = idx % len(self.image_paths)
        
        # 加载图像和掩码
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        
        # 更多数据增强
        if self.is_train:
            # 随机对比度增强
            if random.random() < 0.3:
                factor = random.uniform(0.7, 1.3)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            # 随机亮度增强
            if random.random() < 0.3:
                factor = random.uniform(0.7, 1.3)
                image = ImageEnhance.Brightness(image).enhance(factor)
        
        # 应用变换
        if self.transform:
            image = self.transform['train'](image)
            mask = self.transform['mask'](mask)
            
        # 将掩码二值化
        mask = (mask > 0.5).float()
        
        return image, mask

# 注意力模块（保持不变）
class AttentionGate(nn.Module):
    def __init__(self, gate_channels, input_channels, inter_channels=None):
        super(AttentionGate, self).__init__()
        
        if inter_channels is None:
            inter_channels = input_channels // 2
            
        self.gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.transform = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        # 上采样层，用于匹配尺寸
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, g):
        # x: 编码器特征图 (输入特征)
        # g: 解码器特征图 (门控信号)
        
        g_conv = self.gate(g)
        x_conv = self.transform(x)
        
        # 确保g_conv和x_conv尺寸匹配
        if g_conv.size()[2:] != x_conv.size()[2:]:
            # 根据需要进行上采样或下采样以匹配尺寸
            if g_conv.size()[2] < x_conv.size()[2]:
                g_conv = self.upsample(g_conv)
            else:
                x_conv = nn.functional.adaptive_max_pool2d(x_conv, g_conv.size()[2:])
        
        # 相加并激活
        psi = self.relu(g_conv + x_conv)
        psi = self.psi(psi)
        
        # 应用注意力权重
        return x * psi

# U-Net模型组件（保持不变）
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True, dropout=0.2):
        super(Up, self).__init__()
        
        self.attention = AttentionGate(
            gate_channels=in_channels // 2 if not bilinear else in_channels // 2,
            input_channels=skip_channels
        )
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
            
    def forward(self, x1, x2):
        x2 = self.attention(x2, x1)
        x1 = self.up(x1)
        
        # 输入尺寸调整
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

# 轻量化AttUNet（保持不变）
class AttUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, dropout=0.2):
        super(AttUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 减少通道数，减轻模型复杂度
        self.inc = DoubleConv(n_channels, 32, dropout)
        self.down1 = Down(32, 64, dropout)
        self.down2 = Down(64, 128, dropout)
        self.down3 = Down(128, 256, dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor, dropout)
        
        # 相应调整上采样通道
        self.up1 = Up(512, 256 // factor, skip_channels=256, bilinear=bilinear, dropout=dropout)
        self.up2 = Up(256, 128 // factor, skip_channels=128, bilinear=bilinear, dropout=dropout)
        self.up3 = Up(128, 64 // factor, skip_channels=64, bilinear=bilinear, dropout=dropout)
        self.up4 = Up(64, 32, skip_channels=32, bilinear=bilinear, dropout=dropout)
        
        self.outc = OutConv(32, n_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 后处理函数（保持不变）
def post_process_mask(mask, min_area=50):
    mask_np = mask.numpy().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    # 保留面积大于阈值的区域
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_np[labels == i] = 0
            
    return torch.from_numpy(mask_np).float()

# 改进的训练函数（保持不变）
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, patience=25):
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'dice': [],
        'iou': []
    }
    
    scaler = GradScaler()
    best_dice = -float('inf')
    counter = 0
    
    # 学习率预热
    warmup_epochs = 3
    warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs * len(train_loader)
    )
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for images, masks in train_loop:
            train_loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 应用学习率预热
            if epoch < warmup_epochs:
                warmup_lr_scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        train_loss_avg = train_loss / len(train_loader.dataset)
        metrics_history['train_loss'].append(train_loss_avg)
        
        # 验证
        model.eval()
        val_loss = 0.0
        total_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'dice': 0.0,
            'iou': 0.0
        }
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                pred_masks = torch.sigmoid(outputs)
                batch_metrics = calculate_metrics(pred_masks, masks)
                
                for key in total_metrics:
                    total_metrics[key] += batch_metrics[key] * images.size(0)
        
        val_loss_avg = val_loss / len(val_loader.dataset)
        for key in total_metrics:
            total_metrics[key] /= len(val_loader.dataset)
        
        # 学习率调度（在预热后）
        if epoch >= warmup_epochs:
            scheduler.step(val_loss_avg)
        
        # 记录指标
        metrics_history['val_loss'].append(val_loss_avg)
        metrics_history['precision'].append(total_metrics['precision'])
        metrics_history['recall'].append(total_metrics['recall'])
        metrics_history['dice'].append(total_metrics['dice'])
        metrics_history['iou'].append(total_metrics['iou'])
        
        # 打印本轮指标
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'训练损失: {train_loss_avg:.4f} | 验证损失: {val_loss_avg:.4f}')
        print(f'精确率: {total_metrics["precision"]:.4f} | 召回率: {total_metrics["recall"]:.4f}')
        print(f'Dice系数: {total_metrics["dice"]:.4f} | IoU: {total_metrics["iou"]:.4f}\n')
        
        # 早停机制改进：允许微小波动
        current_dice = total_metrics['dice']
        if current_dice > best_dice + 1e-4:
            best_dice = current_dice
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print(f"保存最佳模型 (Dice: {best_dice:.4f})")
            counter = 0
        else:
            counter += 1
            print(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                print(f"早停在第 {epoch+1} 轮")
                break
    
    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    return model, metrics_history

# 模型集成预测（保持不变）
def ensemble_predict(models, image, thresholds=[0.5, 0.55, 0.6]):
    with torch.no_grad():
        preds = []
        for i, model in enumerate(models):
            model.eval()
            pred = torch.sigmoid(model(image))
            pred = (pred > thresholds[i % len(thresholds)]).float()
            preds.append(pred)
        
        # 多数投票
        ensemble_pred = torch.mean(torch.stack(preds), dim=0)
        return (ensemble_pred > 0.5).float()

# 显示预测结果（保持不变）
def show_predictions(model, dataset, num_samples=5, chinese_font=None, use_postprocessing=True):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    plt.figure(figsize=(15, 3*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            output = model(image)
            pred_mask = torch.sigmoid(output).cpu().squeeze()
            pred_mask_binary = (pred_mask > 0.5).float()
            
            # 后处理
            if use_postprocessing:
                pred_mask_binary = post_process_mask(pred_mask_binary)
            
            # 计算当前样本的指标
            metrics = calculate_metrics(pred_mask_binary.unsqueeze(0), mask.unsqueeze(0))
            
            # 原始图像
            plt.subplot(num_samples, 3, i*3 + 1)
            plt.imshow(image.cpu().squeeze().permute(1, 2, 0))
            plt.title('原始图像', fontproperties=chinese_font)
            plt.axis('off')
            
            # 真实掩码
            plt.subplot(num_samples, 3, i*3 + 2)
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.title('真实掩码', fontproperties=chinese_font)
            plt.axis('off')
            
            # 预测掩码
            plt.subplot(num_samples, 3, i*3 + 3)
            plt.imshow(pred_mask_binary.squeeze(), cmap='gray')
            plt.title(f'预测掩码 (Dice: {metrics["dice"]:.3f})', fontproperties=chinese_font)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/predictions.png')
    plt.show()

# 绘制训练过程中的指标曲线（保持不变）
def plot_metrics(metrics_history, chinese_font):
    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='训练损失')
    plt.plot(metrics_history['val_loss'], label='验证损失')
    plt.title('损失曲线', fontproperties=chinese_font)
    plt.xlabel('轮次', fontproperties=chinese_font)
    plt.ylabel('损失值', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 绘制Dice和IoU曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['dice'], label='Dice系数')
    plt.plot(metrics_history['iou'], label='IoU')
    plt.title('Dice和IoU曲线', fontproperties=chinese_font)
    plt.xlabel('轮次', fontproperties=chinese_font)
    plt.ylabel('值', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 绘制精确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history['precision'], label='精确率', color='green')
    plt.title('精确率曲线', fontproperties=chinese_font)
    plt.xlabel('轮次', fontproperties=chinese_font)
    plt.ylabel('精确率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 绘制召回率曲线
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history['recall'], label='召回率', color='orange')
    plt.title('召回率曲线', fontproperties=chinese_font)
    plt.xlabel('轮次', fontproperties=chinese_font)
    plt.ylabel('召回率', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/metrics_curve.png')
    plt.show()

def main():
    # 设置中文字体
    chinese_font = setup_chinese_font()
    test_chinese_font_display(chinese_font)
    
    # 数据路径设置
    image_dir = '/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images'
    mask_dir = '/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks'
    
    # 获取所有图像和掩码路径
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*')))
    
    assert len(image_paths) == len(mask_paths), "图像和掩码数量不匹配"
    print(f"找到 {len(image_paths)} 个样本")
    
    # 划分训练集和验证集（小样本时适当减少验证集比例）
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=seed
    )
    
    # 更强的数据增强策略
    transform = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            RandomZoom(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            RandomGaussianBlur(p=0.4),
            RandomElasticTransform(p=0.4),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'mask': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    }
    
    # 创建数据集和数据加载器
    train_dataset = PolypDataset(train_img, train_mask, transform, is_train=True)
    val_dataset = PolypDataset(val_img, val_mask, transform, is_train=False)
    
    batch_size = 12  # 小样本时适当减小batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # 初始化模型、损失函数和优化器
    model = AttUNet(n_channels=3, n_classes=1, bilinear=True, dropout=0.25).to(device)
    
    # 使用混合损失函数
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.7)
    
    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6, verbose=True, min_lr=1e-6
    )
    
    # 训练模型
    print("开始训练...")
    model, metrics_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=150, patience=25
    )
    
    # 绘制指标曲线
    plot_metrics(metrics_history, chinese_font)
    
    # 显示预测结果
    print("显示预测结果...")
    show_predictions(model, val_dataset, num_samples=10, chinese_font=chinese_font, 
                    use_postprocessing=True)

if __name__ == "__main__":
    main()