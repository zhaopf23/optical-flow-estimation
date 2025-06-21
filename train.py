import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from datetime import datetime
import math
import pandas as pd

# 设置随机种子以确保可重复性
torch.manual_seed(41)
np.random.seed(41)
random.seed(41)

# 定义数据预处理
class FlowTransform:
    def __init__(self, crop_size=(384, 512), augment=True):
        self.crop_size = crop_size
        self.augment = augment
        
    def __call__(self, img1, img2, flow):
        # 转换为Tensor
        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)
        flow = torch.from_numpy(flow).float()
        
        # 归一化图像
        img1 = transforms.functional.normalize(img1, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img2 = transforms.functional.normalize(img2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        # 随机裁剪
        h, w = img1.shape[1:]
        crop_h, crop_w = self.crop_size
        if h > crop_h or w > crop_w:
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            
            img1 = img1[:, top:top+crop_h, left:left+crop_w]
            img2 = img2[:, top:top+crop_h, left:left+crop_w]
            flow = flow[:, top:top+crop_h, left:left+crop_w]
            
            # 调整光流坐标
            flow[0] = flow[0] * (crop_w / w)
            flow[1] = flow[1] * (crop_h / h)
        
        # 数据增强
        if self.augment:
            # 随机水平翻转
            if random.random() > 0.5:
                img1 = torch.flip(img1, [2])
                img2 = torch.flip(img2, [2])
                flow = torch.flip(flow, [2])
                flow[0] = -flow[0]  # 翻转后u分量取反
                
            # 随机垂直翻转
            if random.random() > 0.5:
                img1 = torch.flip(img1, [1])
                img2 = torch.flip(img2, [1])
                flow = torch.flip(flow, [1])
                flow[1] = -flow[1]  # 翻转后v分量取反
        
        return img1, img2, flow

# 定义FlyingChairs数据集类
class FlyingChairsDataset(Dataset):
    def __init__(self, data_dir, split_file, split='train', transform=None, sample_ratio=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 打印路径信息以便调试
        print(f"Data directory: {os.path.abspath(data_dir)}")
        print(f"Split file: {os.path.abspath(split_file)}")
        
        # 检查路径是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # 读取训练/验证划分文件
        with open(split_file, 'r') as f:
            lines = f.readlines()
            print(f"Total lines in split file: {len(lines)}")
            
            valid_samples = 0
            for idx, line in enumerate(lines):
                # 清理行内容
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 1表示训练集，2表示验证集
                if (split == 'train' and line == '1') or (split == 'val' and line == '2'):
                    # 样本ID从1开始，对应文件命名
                    sample_id = idx + 1
                    
                    # 检查文件是否存在
                    img1_path = os.path.join(data_dir, f'{sample_id:05d}_img1.ppm')
                    img2_path = os.path.join(data_dir, f'{sample_id:05d}_img2.ppm')
                    flow_path = os.path.join(data_dir, f'{sample_id:05d}_flow.flo')
                    
                    if os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(flow_path):
                        self.samples.append(sample_id)
                        valid_samples += 1
                    else:
                        missing_files = []
                        if not os.path.exists(img1_path): missing_files.append(img1_path)
                        if not os.path.exists(img2_path): missing_files.append(img2_path)
                        if not os.path.exists(flow_path): missing_files.append(flow_path)
                        print(f"Missing files for sample {sample_id}: {', '.join(missing_files)}")
        if sample_ratio < 1.0:
            num_samples = int(len(self.samples) * sample_ratio)
            self.samples = random.sample(self.samples, num_samples)
            print(f"Using {num_samples} samples ({sample_ratio*100}% of full dataset)")
        
        print(f"Loaded {len(self.samples)} valid samples for {split} set (from {valid_samples} potential samples)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # 读取图像对
        img1_path = os.path.join(self.data_dir, f'{sample_id:05d}_img1.ppm')
        img2_path = os.path.join(self.data_dir, f'{sample_id:05d}_img2.ppm')
        flow_path = os.path.join(self.data_dir, f'{sample_id:05d}_flow.flo')
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None:
            raise FileNotFoundError(f"Image1 not found or corrupted: {img1_path}")
        if img2 is None:
            raise FileNotFoundError(f"Image2 not found or corrupted: {img2_path}")
        
        # 读取光流文件
        flow = self.read_flo(flow_path)
        
        # 应用变换
        if self.transform:
            return self.transform(img1, img2, flow)
        else:
            # 转换为Tensor
            img1 = transforms.functional.to_tensor(img1)
            img2 = transforms.functional.to_tensor(img2)
            flow = torch.from_numpy(flow).float()
            
            # 归一化图像
            img1 = transforms.functional.normalize(img1, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            img2 = transforms.functional.normalize(img2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            return img1, img2, flow
    
    @staticmethod
    def read_flo(file_path):
        """读取.flo格式的光流文件"""
        with open(file_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise RuntimeError(f'Invalid .flo file: {file_path}')
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * h * w)
        
        # 重塑为 (2, H, W) 格式
        flow = np.resize(data, (h, w, 2)).transpose(2, 0, 1)
        return flow

# 改进的光流估计网络 - 修正通道数问题
class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        
        # 特征提取器（共享权重）
        self.conv1 = self._make_conv_block(3, 64)
        self.conv2 = self._make_conv_block(64, 128, stride=2)
        self.conv3 = self._make_conv_block(128, 256, stride=2)
        self.conv3_1 = self._make_conv_block(256, 256)
        self.conv4 = self._make_conv_block(256, 512, stride=2)
        self.conv4_1 = self._make_conv_block(512, 512)
        
        # 光流预测器 - 修正通道数
        self.deconv3 = self._make_deconv_block(1024, 256)  # 512*2 = 1024
        self.deconv2 = self._make_deconv_block(768, 128)    # 修正为768 (256*3)
        self.deconv1 = self._make_deconv_block(384, 64)     # 修正为384 (128*3)
        
        # 最终预测层 - 修正输入通道数
        self.predict_flow = nn.Conv2d(192, 2, kernel_size=3, padding=1)  # 修正为192 (64*3)
        
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _make_deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, img1, img2):
        # 提取图像1的特征
        conv1a = self.conv1(img1)
        conv2a = self.conv2(conv1a)
        conv3a = self.conv3_1(self.conv3(conv2a))
        conv4a = self.conv4_1(self.conv4(conv3a))
        
        # 提取图像2的特征
        conv1b = self.conv1(img2)
        conv2b = self.conv2(conv1b)
        conv3b = self.conv3_1(self.conv3(conv2b))
        conv4b = self.conv4_1(self.conv4(conv3b))
        
        # 合并特征
        x = torch.cat([conv4a, conv4b], dim=1)  # 512+512=1024通道
        
        # 上采样过程（带有跳跃连接） - 修正通道数
        deconv3 = self.deconv3(x)  # 1024 -> 256
        deconv3 = torch.cat([deconv3, conv3a, conv3b], dim=1)  # 256+256+256=768
        
        deconv2 = self.deconv2(deconv3)  # 768 -> 128
        deconv2 = torch.cat([deconv2, conv2a, conv2b], dim=1)  # 128+128+128=384
        
        deconv1 = self.deconv1(deconv2)  # 384 -> 64
        deconv1 = torch.cat([deconv1, conv1a, conv1b], dim=1)  # 64+64+64=192
        
        # 预测光流
        flow = self.predict_flow(deconv1)  # 192 -> 2
        return flow

# 改进的多尺度损失函数
class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        
    def forward(self, pred_flow, true_flow):
        """
        :param pred_flow: 预测光流 (batch, 2, H, W)
        :param true_flow: 真实光流 (batch, 2, H, W)
        """
        # 计算平滑L1损失
        loss = self.smooth_l1_loss(pred_flow, true_flow)
        
        # 梯度归一化项（提高训练稳定性）
        grad_loss = self.gradient_loss(pred_flow, true_flow)
        
        # 总损失
        total_loss = loss + 0.01 * grad_loss
        
        return total_loss
    
    def smooth_l1_loss(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
        return loss.mean()
    
    def gradient_loss(self, pred, target):
        # 计算预测光流的梯度
        pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        # 计算真实光流的梯度
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        # 计算梯度差异
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)
        
        return grad_diff_x.mean() + grad_diff_y.mean()

# 带重启的余弦退火学习率调度器
class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.T_cur = self.T_cur + 1
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 = self.T_0 * self.T_mult
        
        super().step(epoch)

# 学习率预热调度器（修复版本）
class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_scheduler):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.step_count = 0  # 添加步数计数器
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.step_count < self.warmup_steps:
            # 线性增加学习率
            return [base_lr * (self.step_count / self.warmup_steps) 
                    for base_lr in self.base_lrs]
        return self.base_scheduler.get_lr()
    
    def step(self, epoch=None):
        self.step_count += 1  # 每次step增加计数器
        if self.step_count <= self.warmup_steps:
            # 在预热阶段，手动设置学习率
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            # 预热结束后，使用基础调度器
            self.base_scheduler.step(epoch)
        
        # 更新last_epoch
        self.last_epoch += 1

# 可视化光流
def visualize_flow(flow):
    # 将光流转换为numpy数组
    flow_np = flow.detach().cpu().numpy()
    u = flow_np[0]
    v = flow_np[1]
    
    # 创建HSV图像
    h, w = u.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(u, v)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换为RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

# 损失可视化函数
def plot_loss_history(train_losses, val_losses, lr_history, output_dir="loss_plots"):
    """绘制训练和验证损失曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 训练损失
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 学习率变化
    plt.subplot(2, 1, 2)
    plt.plot(lr_history, label='Learning Rate', color='green')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'third_loss_history_{timestamp}.png'))
    plt.close()
    
    # 保存损失数据
    pd.DataFrame({
        'epoch': range(1, len(train_losses)+1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rate': lr_history
    }).to_csv(os.path.join(output_dir, 'loss_history.csv'), index=False)

# 训练函数
def train_model(resume_checkpoint=None, sample_ratio=1.0):  # 添加 sample_ratio 参数，默认值为 1.0（100%）
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    
    # 构建正确的路径
    data_dir = os.path.join(script_dir, 'FlyingChairs_release', 'data')
    split_file = os.path.join(script_dir, 'FlyingChairs_release', 'FlyingChairs_train_val.txt')
    
    # 创建数据集 - 添加 sample_ratio 参数
    try:
        train_transform = FlowTransform(augment=True)
        val_transform = FlowTransform(augment=False)
        
        train_dataset = FlyingChairsDataset(data_dir, split_file, split='train', 
                                           transform=train_transform, sample_ratio=sample_ratio)
        val_dataset = FlyingChairsDataset(data_dir, split_file, split='val', 
                                         transform=val_transform, sample_ratio=sample_ratio)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        return
    
    # 检查数据集是否为空
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: One or both datasets are empty. Check data paths and split file.")
        return
    
    batch_size = 16 
    # 根据GPU内存调整批量大小
    # if torch.cuda.is_available():
    #     # 尝试较大的批量大小
    #     try:
    #         # 测试GPU是否能处理更大的batch_size
    #         test_batch = torch.randn(32, 3, 384, 512).to(device)
    #         model(test_batch, test_batch)  # 前向传播测试
    #         batch_size = 32  # 如果成功，使用32
    #         print("Using larger batch_size=32")
    #     except RuntimeError as e:
    #         if 'out of memory' in str(e):
    #             print("GPU memory insufficient for batch_size=32, trying batch_size=16")
    #             try:
    #                 torch.cuda.empty_cache()
    #                 test_batch = torch.randn(16, 3, 384, 512).to(device)
    #                 model(test_batch, test_batch)
    #                 batch_size = 16
    #                 print("Using batch_size=16")
    #             except RuntimeError:
    #                 print("Using default batch_size=8")
    #                 batch_size = 8
    #         else:
    #             print("Using default batch_size=8")
    #             batch_size = 8
    # else:
    #     batch_size = 2  # CPU模式使用较小的batch_size
    
    num_workers = min(8, os.cpu_count() // 2)
    print(f"Using {num_workers} data loader workers and batch_size={batch_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 初始化模型
    model = FlowNet().to(device)

    # 损失函数
    criterion = MultiScaleLoss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 学习率调度器
    base_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler = WarmupScheduler(optimizer, warmup_steps=5, base_scheduler=base_scheduler)
    
    # 训练参数
    start_epoch = 0
    num_epochs = 50
    best_val_loss = float('inf')
    
    # 损失历史记录
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    
    # 恢复训练检查点
    if resume_checkpoint:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint:
            scheduler_state = checkpoint['scheduler_state_dict']
            if callable(scheduler_state):
                scheduler.base_scheduler.T_0 = scheduler_state().get('T_0', 10)
                scheduler.base_scheduler.T_cur = scheduler_state().get('T_cur', 0)
        
        # 加载历史记录
        train_loss_history = checkpoint.get('train_loss_history', [])
        val_loss_history = checkpoint.get('val_loss_history', [])
        lr_history = checkpoint.get('lr_history', [])
        
        # 设置起始epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 手动设置调度器的步数
        if hasattr(scheduler, 'step_count'):
            scheduler.step_count = start_epoch
        
        print(f"Resumed from epoch {start_epoch-1}, best val loss: {best_val_loss:.4f}")
    
    # 打印模型参数数量
    if start_epoch == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)
    
    # 训练循环
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (img1, img2, flow) in enumerate(train_loader):
            img1, img2, flow = img1.to(device, non_blocking=True), img2.to(device, non_blocking=True), flow.to(device, non_blocking=True)
            
            # 前向传播
            pred_flow = model(img1, img2)
            
            # 计算损失
            loss = criterion(pred_flow, flow)
            train_loss += loss.item()
            
            # 反向传播和优化
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # 每50个batch打印一次
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, flow in val_loader:
                img1, img2, flow = img1.to(device), img2.to(device), flow.to(device)
                pred_flow = model(img1, img2)
                val_loss += criterion(pred_flow, flow).item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.base_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'lr_history': lr_history,
            }, os.path.join(results_dir, 'third_best_flow_model.pth'))
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(results_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.base_scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'lr_history': lr_history,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # 可视化
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            try:
                sample_idx = random.randint(0, len(val_dataset) - 1)
                sample_img1, sample_img2, sample_flow = val_dataset[sample_idx]
                sample_img1 = sample_img1.unsqueeze(0).to(device)
                sample_img2 = sample_img2.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_flow = model(sample_img1, sample_img2)
                
                # 可视化
                true_flow_vis = visualize_flow(sample_flow)
                pred_flow_vis = visualize_flow(pred_flow[0])
                
                # 创建可视化图像
                plt.figure(figsize=(18, 6))
                
                # 原始图像1
                plt.subplot(1, 3, 1)
                plt.title("Image 1")
                img1_vis = sample_img1[0].permute(1, 2, 0).cpu().numpy()
                img1_vis = (img1_vis - img1_vis.min()) / (img1_vis.max() - img1_vis.min())
                plt.imshow(img1_vis)
                plt.axis('off')
                
                # 真实光流
                plt.subplot(1, 3, 2)
                plt.title("True Optical Flow")
                plt.imshow(true_flow_vis)
                plt.axis('off')
                
                # 预测光流
                plt.subplot(1, 3, 3)
                plt.title(f"Predicted Optical Flow (Epoch {epoch+1})")
                plt.imshow(pred_flow_vis)
                plt.axis('off')
                
                plt.savefig(os.path.join(results_dir, "visualizations", f'flow_comparison_epoch_{epoch+1}.png'))
                plt.close()
            except Exception as e:
                print(f"Error during visualization: {e}")
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'LR: {current_lr:.2e}, Time: {epoch_time:.1f}s')
        
        # 每5个epoch绘制一次损失曲线
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plot_loss_history(train_loss_history, val_loss_history, lr_history, results_dir)
    
    # 训练结束后绘制最终损失曲线
    plot_loss_history(train_loss_history, val_loss_history, lr_history, results_dir)
    
    # 保存最终模型
    final_model_path = os.path.join(results_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.base_scheduler.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'lr_history': lr_history,
    }, final_model_path)
    
    # 总训练时间
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/3600:.2f} hours!")
    print(f"All results saved to: {results_dir}")
    print(f"Final model saved to: {final_model_path}")
    
    return train_loss_history, val_loss_history

if __name__ == "__main__":
    # 示例：从第二轮开始训练，指定第一轮结束时的检查点
    train_model()
