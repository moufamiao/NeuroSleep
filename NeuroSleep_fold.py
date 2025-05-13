import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, \
    f1_score
import matplotlib.pyplot as plt
import pywt
from tqdm import tqdm
import pre_data.loadedf_shhs
import pre_data.loadedf_78
import pre_data.loadedf
from imblearn.metrics import geometric_mean_score
# 在文件开头添加sklearn的KFold导入
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

class EEGPreprocessor:
    def __init__(self, train_mean=None, train_std=None):
        self.train_mean = train_mean
        self.train_std = train_std

    def normalize(self, X, eps=1e-8):
        """基于训练集的标准化核心方法"""
        if self.train_mean is None:
            # 初始化时自动计算统计量
            self.train_mean = np.mean(X, axis=(0, 1))
            self.train_std = np.std(X, axis=(0, 1)) + eps
        return (X - self.train_mean) / self.train_std


class EEGDataset(Dataset):
    def __init__(self, X, y, preprocessor=None, augment=False):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.augment = augment  # 参数保留但不使用

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 保持数据拷贝逻辑
        signal = self.X[idx].copy()  # shape (3000,1)

        # 保持维度转换逻辑
        return (
            torch.FloatTensor(signal),  # 转换为(1,3000)
            torch.LongTensor([self.y[idx]]).squeeze()  # 标签维度压缩
        )


class AdditiveAttention(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5

        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.norm = nn.BatchNorm1d(token_dim * num_heads)  # 标准化放在 Proj 后
        self.dropout = nn.Dropout(p=0.2)  # ✅ 新增 Dropout 层

        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = F.normalize(self.to_query(x), dim=-1)  # B x N x D
        key = F.normalize(self.to_key(x), dim=-1)      # B x N x D

        query_weight = query @ self.w_g                # B x N x 1
        A = query_weight * self.scale_factor           # B x N x 1
        A = F.normalize(A, dim=1)

        G = torch.sum(A * query, dim=1)                # B x D
        G = einops.repeat(G, "b d -> b n d", n=key.shape[1])  # B x N x D

        out = self.Proj(G * key) + query               # B x N x D

        B, N, D = out.shape
        out = out.view(B * N, D)
        out = self.norm(out)                           # BatchNorm
        out = F.gelu(out)                              # GELU 激活
        out = self.dropout(out)                        # ✅ Dropout
        out = out.view(B, N, D)

        out = self.final(out)                          # Final linear 输出

        return out



class BasicResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 快捷连接（处理维度不匹配）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 核心残差连接
        return self.gelu(out)  # 相加后统一激活



class EEGHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Conv1d(32, 64, kernel_size=7),
            #DynamicConv1d(32, 64, kernel_size=7, num_experts=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3),
            nn.Dropout(0.3)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            dropout=0.6,
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(128)

        # 注意力机制（输入维度改为 128）
        self.attention = AdditiveAttention(
            in_dims=128,
            token_dim=128,
            num_heads=4
        )

        # 分类器（输入维度改为 128）
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(128, 5)  # 输出 5 类
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # CNN处理
        x = x.permute(0, 2, 1)  # [batch, channels, time]
        x = self.cnn(x)  # [batch, 64, seq]
        x = x.permute(0, 2, 1)  # [batch, seq, features]

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch, seq, 128]
        lstm_out = self.lstm_norm(lstm_out)

        # 注意力机制
        attn_out = self.attention(lstm_out)  # [batch, seq, 128]
        context = attn_out.mean(dim=1)  # [batch, 128]

        # 分类
        return self.classifier(context)


class SleepStageLoss(nn.Module):
    def __init__(self, fs=100, epoch_duration=30, lambda_phys=0.5, device='cuda'):
        super().__init__()
        self.device = device
        self.fs = fs
        self.epoch_samples = fs * epoch_duration  # 3000 = 100Hz * 30s
        self.lambda_phys = lambda_phys

        # 生理规则阈值矩阵 [num_classes=5, num_bands=5]
        self.register_buffer('min_thresholds', torch.tensor([
            [0.0, 0.0, 0.5, 0.3, 0.0],  # W期
            [0.0, 0.3, 0.0, 0.0, 0.0],  # N1期
            [0.0, 0.0, 0.0, 0.0, 0.15],  # N2期
            [0.2, 0.0, 0.0, 0.0, 0.0],  # N3期
            [0.0, 0.4, 0.2, 0.0, 0.0]  # REM期
        ], device=device))

        self.register_buffer('max_thresholds', torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],  # W期
            [1.0, 1.0, 0.5, 1.0, 1.0],  # N1期
            [1.0, 1.0, 1.0, 1.0, 1.0],  # N2期
            [1.0, 1.0, 1.0, 1.0, 1.0],  # N3期
            [1.0, 1.0, 0.3, 1.0, 1.0]  # REM期
        ], device=device))

    def compute_spectral_loss(self, eeg, true_labels):
        """基于真实标签计算频谱约束损失"""
        eeg = eeg.squeeze(-1).to(self.device)
        true_labels = true_labels.to(self.device)

        # 将 fft 操作限定在 CPU
        eeg_cpu = eeg.cpu()  # 将数据移到 CPU
        fft_coef = torch.fft.rfft(eeg_cpu, dim=1)  # 仅在 CPU 上计算 FFT
        freqs = torch.fft.rfftfreq(self.epoch_samples, 1 / self.fs, device='cpu')  # 频率计算也在 CPU 上进行

        # 计算频段能量比例（与之前相同）
        band_edges = [(0.5, 2), (4, 7), (8, 13), (13, 30), (11, 16)]
        band_energies = []
        for low, high in band_edges:
            mask = (freqs >= low) & (freqs <= high)
            energy = torch.sum(torch.abs(fft_coef[:, mask]) ** 2, dim=1)
            band_energies.append(energy)
        band_energy = torch.stack(band_energies, dim=1)
        total_energy = band_energy.sum(dim=1, keepdim=True) + 1e-8
        band_ratios = band_energy / total_energy

        # 确保 min_thresh 和 band_ratios 在相同的设备上
        min_thresh = self.min_thresholds[true_labels].to(self.device)  # 将 min_thresh 移到相同设备
        max_thresh = self.max_thresholds[true_labels].to(self.device)  # 将 max_thresh 移到相同设备

        # 将 band_ratios 移到与 min_thresh 相同的设备
        band_ratios = band_ratios.to(self.device)

        # 计算违反阈值的情况
        min_violation = torch.relu(min_thresh - band_ratios)
        max_violation = torch.relu(band_ratios - max_thresh)
        total_violation = (min_violation + max_violation).mean()

        return total_violation

    def compute_spindle_loss(self, eeg, true_labels):
        """基于真实标签计算纺锤波损失（仅在真实N2期样本计算）"""
        eeg = eeg.squeeze(-1).to(self.device).contiguous()
        true_labels = true_labels.to(self.device)
        # 仅处理真实标签为N2期的样本
        mask = (true_labels == 2)  # N2期索引为2
        if not mask.any():
            return torch.tensor(0.0, device=self.device)

        # 仅对N2期样本计算纺锤波
        eeg_n2 = eeg[mask]

        # 纺锤波检测（与之前相同）
        t = torch.linspace(0, 1, self.fs, device=self.device)
        spindle_kernel = torch.sin(2 * torch.pi * 13 * t) * torch.hamming_window(self.fs, device=self.device)

        # 转移计算到 CPU 上
        filtered = nn.functional.conv1d(
            eeg_n2.unsqueeze(1),
            spindle_kernel.view(1, 1, -1),
            padding='same'
        ).squeeze(1)

        # 将傅里叶变换移到CPU
        analytic = torch.view_as_real(torch.fft.fft(filtered.cpu(), dim=1))  # 使用CPU进行FFT
        envelope = torch.sqrt(analytic[..., 0] ** 2 + analytic[..., 1] ** 2)
        threshold = 0.5 * envelope.amax(dim=1, keepdim=True)
        duration = torch.sigmoid(10 * (envelope - threshold)).mean(dim=1)

        # 计算损失：持续时间不足时惩罚
        spindle_loss = torch.relu(0.5 - duration).mean()
        return spindle_loss

    def forward(self, pred, target, eeg):
        target = target.squeeze(-1).long().to(self.device)
        eeg = eeg.to(self.device)
        pred = pred.to(self.device)
        target = target.to(self.device)
        # 监督损失（保持不变）
        ce_loss = nn.CrossEntropyLoss()(pred, target)

        # 关键修改：生理约束基于真实标签
        spectral_loss = self.compute_spectral_loss(eeg, target)  # 传入target而非pred
        spindle_loss = self.compute_spindle_loss(eeg, target)  # 传入target而非pred_labels

        # 组合损失
        #total_loss = ce_loss + self.lambda_phys * (4 * spectral_loss + spindle_loss)
        total_loss = ce_loss + self.lambda_phys * 10 * spectral_loss
        return total_loss, ce_loss.detach(), spectral_loss.detach(), spindle_loss.detach()


class Trainer:
    def __init__(self, model, device, preprocessor):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor
        self.criterion = SleepStageLoss(fs=100, lambda_phys=0.8, device=device)

    def prepare_data(self, X, y):
        """修正后的数据预处理流程"""
        # 步骤1：先划分原始数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.05,
            stratify=y,
            #random_state=42  # 可重复性种子
        )
        # 步骤3：用去噪后的训练数据计算标准化参数
        self.preprocessor = EEGPreprocessor()
        X_train_norm = self.preprocessor.normalize(X_train)  # 计算并应用标准化

        # 步骤4：验证集使用训练参数标准化
        X_val_norm = (X_val - self.preprocessor.train_mean) / self.preprocessor.train_std

        # 步骤5：创建数据集
        train_dataset = EEGDataset(X_train_norm, y_train, self.preprocessor, augment=True)
        val_dataset = EEGDataset(X_val_norm, y_val, self.preprocessor, augment=False)

        return train_dataset, val_dataset

    def train(self, train_loader, val_loader, epochs):
        # 优化器配置
        optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 5e-3}
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # 关键修改：跟踪最佳结果
        best_acc = 0.0
        best_metrics = {}
        best_model_state = None

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for inputs, labels in progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss, ce_loss, spectral_loss, spindle_loss = self.criterion(outputs, labels, inputs)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                progress.set_postfix({'loss': loss.item()})

            # 验证阶段
            val_loss, val_acc, val_metrics = self.evaluate(val_loader)

            # 更新最佳结果
            if val_acc > best_acc:
                best_acc = val_acc
                best_metrics = {
                    'mf1': val_metrics['mf1'],
                    'k': val_metrics['k'],
                    'mgm': val_metrics['mgm']
                }
                best_model_state = self.model.state_dict().copy()

            # 学习率调整
            scheduler.step(val_loss)

            # 打印日志
            print(f'Epoch {epoch + 1}: '
                  f'Train Loss: {train_loss / len(train_loader.dataset):.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f"Acc: {val_acc:.4f} | "
                  f"mF1: {val_metrics['mf1']:.4f} | "
                  f"Kappa: {val_metrics['k']:.4f} | "
                  f"G-mean: {val_metrics['mgm']:.4f} | "
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        # 加载最佳模型状态
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_acc, best_metrics

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_preds = []  # 新增
        all_labels = []  # 新增

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss, ce_loss, spectral_loss, spindle_loss = self.criterion(outputs, labels, inputs)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels.squeeze()).sum().item()

                all_preds.extend(preds.cpu().numpy())  # 新增
                all_labels.extend(labels.cpu().numpy())  # 新增

        # 计算指标
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        mf1 = f1_score(all_labels, all_preds, average='macro')
        k = cohen_kappa_score(all_labels, all_preds)
        mgm = geometric_mean_score(all_labels, all_preds, average='macro')

        # 修改返回结构
        return avg_loss, accuracy, {
            'mf1': mf1,
            'k': k,
            'mgm': mgm,
            'preds': all_preds,  # 新增
            'labels': all_labels  # 新增
        }


# 主程序 ----------------------------------------------------------------
if __name__ == "__main__":
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    X, y = pre_data.loadedf_78.loaddata()
    X = np.transpose(X, (0, 2, 1))  # 保持原始数据形状处理

    # 配置交叉验证
    n_splits = 20
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储各折最佳评估结果
    fold_metrics = {
        'acc': [],
        'mf1': [],
        'k': [],
        'mgm': []
    }
    # 新增：存储所有预测结果和真实标签
    all_labels = []
    all_preds = []
    fold_accuracies = []  # 用于存储每折准确率

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # 划分训练集/验证集
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 初始化预处理器（每折独立）
        preprocessor = EEGPreprocessor()

        # 预处理（计算训练集统计量）
        X_train_norm = preprocessor.normalize(X_train)
        X_val_norm = (X_val - preprocessor.train_mean) / preprocessor.train_std

        # 创建数据集
        train_dataset = EEGDataset(X_train_norm, y_train, augment=True)
        val_dataset = EEGDataset(X_val_norm, y_val, augment=False)

        # 创建DataLoader
        batch_size = 512
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            num_workers=0, pin_memory=False
        )

        # 初始化新模型（确保权重重置）
        model = EEGHybridModel()
        trainer = Trainer(model, device, preprocessor)

        # 训练并获取最佳结果
        best_acc, best_metrics = trainer.train(train_loader, val_loader, epochs=55)

        # 关键新增：用最佳模型获取完整预测结果
        _, _, val_metrics = trainer.evaluate(val_loader)
        fold_preds = val_metrics['preds']  # 需要修改evaluate方法返回预测
        fold_labels = val_metrics['labels']

        # 记录结果
        fold_metrics['acc'].append(best_acc)
        fold_metrics['mf1'].append(best_metrics['mf1'])
        fold_metrics['k'].append(best_metrics['k'])
        fold_metrics['mgm'].append(best_metrics['mgm'])

        # 收集预测结果
        all_labels.extend(fold_labels)
        all_preds.extend(fold_preds)
        fold_accuracies.append(best_acc)  # 记录每折准确率

        # 新增：整体评估 -----------------------------------------------------
    print("\n=== Final Evaluation ===")
    print(f"20-Fold Cross Validation Accuracies: {fold_accuracies}")
    print(f"Mean Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"f1_score: {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")
    print(f"Macro-averaged G-mean: {geometric_mean_score(all_labels, all_preds, average='macro'):.4f}")

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 获取类别名称（假设类别从0开始）
    classes = np.arange(cm.shape[0])

    # 打印表头
    print(f"\n{'':<7} |", end="")
    for cls in classes:
        print(f"{'Predicted ' + str(cls):<9}", end="")
    print(f" | {'Total':<6} | {'Accuracy':<8}")
    print("-" * (10 + 11 * len(classes)))

    # 打印每一行
    for i, cls in enumerate(classes):
        true_count = np.sum(cm[i, :])
        accuracy = cm[i, i] / true_count if true_count > 0 else 0
        print(f"True {cls:<3} |", end="")
        for j in range(len(classes)):
            print(f"{cm[i, j]:^9}", end="")
        print(f" | {true_count:^6} | {accuracy:^8.2%}")

    # 打印总计数
    print("-" * (10 + 11 * len(classes)))
    print(f"{'Total':<7} |", end="")
    for j in range(len(classes)):
        col_count = np.sum(cm[:, j])
        print(f"{col_count:^9}", end="")
    print()

    # 原有交叉验证结果打印
    print("\n=== Cross Validation Results ===")
    for metric in fold_metrics:
        values = fold_metrics[metric]
        print(f"{metric.upper():<5} | Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")