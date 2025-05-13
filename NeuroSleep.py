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

        # ✅ BatchNorm1d 在 Linear 和 GELU 之间
        self.norm = nn.BatchNorm1d(token_dim * num_heads)

        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = F.normalize(self.to_query(x), dim=-1)  # B x N x D
        key = F.normalize(self.to_key(x), dim=-1)  # B x N x D

        query_weight = query @ self.w_g  # B x N x 1
        A = query_weight * self.scale_factor  # B x N x 1
        A = F.normalize(A, dim=1)

        G = torch.sum(A * query, dim=1)  # B x D
        G = einops.repeat(G, "b d -> b n d", n=key.shape[1])  # B x N x D

        out = self.Proj(G * key) + query  # B x N x D

        # ✅ BatchNorm1d expects (B*N, D)
        B, N, D = out.shape
        out = out.view(B * N, D)
        out = self.norm(out)  # 标准化在 Proj 和激活之间
        out = F.gelu(out)  # 激活
        out = out.view(B, N, D)

        out = self.final(out)  # 输出投影

        return out


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
            [0.0, 0.4, 0.2, 0.0, 0.0]   # REM期
        ], device=device))

        self.register_buffer('max_thresholds', torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],  # W期
            [1.0, 1.0, 0.5, 1.0, 1.0],  # N1期
            [1.0, 1.0, 1.0, 1.0, 1.0],  # N2期
            [1.0, 1.0, 1.0, 1.0, 1.0],  # N3期
            [1.0, 1.0, 0.3, 1.0, 1.0]   # REM期
        ], device=device))

    def compute_spectral_loss(self, eeg, true_labels):
        """
        对每个通道分别计算 band ratio，再对通道平均
        eeg: (B, C, T)
        """
        if eeg.shape[1] == self.epoch_samples:
            eeg = eeg.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        B, C, T = eeg.shape
        eeg = eeg.to(self.device)
        true_labels = true_labels.to(self.device)

        fft_coef = torch.fft.rfft(eeg, dim=2)
        freqs = torch.fft.rfftfreq(self.epoch_samples, 1 / self.fs, device=self.device)

        band_edges = [(0.5, 2), (4, 7), (8, 13), (13, 30), (11, 16)]
        band_energies = []
        for low, high in band_edges:
            mask = (freqs >= low) & (freqs <= high)
            energy = torch.sum(torch.abs(fft_coef[:, :, mask]) ** 2, dim=2)  # [B, C]
            band_energies.append(energy)

        band_energy = torch.stack(band_energies, dim=2)  # [B, C, 5]
        total_energy = band_energy.sum(dim=2, keepdim=True) + 1e-8
        band_ratios = band_energy / total_energy  # [B, C, 5]

        # 拉伸标签到 [B, C, 5] 方便广播
        min_thresh = self.min_thresholds[true_labels].unsqueeze(1)  # [B, 1, 5]
        max_thresh = self.max_thresholds[true_labels].unsqueeze(1)

        min_violation = torch.relu(min_thresh - band_ratios)
        max_violation = torch.relu(band_ratios - max_thresh)
        total_violation = (min_violation + max_violation).mean()  # 对 B×C×5 平均

        return total_violation

    def compute_spindle_loss(self, eeg, true_labels):
        """
        对每个通道分别计算纺锤波 loss，平均后返回
        eeg: (B, C, T)
        """
        eeg = eeg.to(self.device).contiguous()
        true_labels = true_labels.to(self.device)

        mask = (true_labels == 2)  # N2期
        if not mask.any():
            return torch.tensor(0.0, device=self.device)

        eeg_n2 = eeg[mask]  # [B_n2, C, T]
        Bn2, C, T = eeg_n2.shape

        t = torch.linspace(0, 1, self.fs, device=self.device)
        spindle_kernel = torch.sin(2 * torch.pi * 13 * t) * torch.hamming_window(self.fs, device=self.device)
        kernel = spindle_kernel.view(1, 1, -1)  # [1, 1, fs]

        # 逐通道卷积
        filtered = nn.functional.conv1d(
            eeg_n2.view(-1, 1, T),  # [Bn2*C, 1, T]
            kernel,
            padding='same'
        ).view(Bn2, C, T)

        analytic = torch.view_as_real(torch.fft.fft(filtered, dim=2))
        envelope = torch.sqrt(analytic[..., 0] ** 2 + analytic[..., 1] ** 2)
        threshold = 0.5 * envelope.amax(dim=2, keepdim=True)
        duration = torch.sigmoid(10 * (envelope - threshold)).mean(dim=2)  # [B_n2, C]

        spindle_loss = torch.relu(0.5 - duration).mean()  # 所有通道平均
        return spindle_loss

    def forward(self, pred, target, eeg):
        """
        pred: (B, num_classes)
        target: (B, 1)
        eeg: (B, C, T)
        """
        target = target.squeeze(-1).long().to(self.device)
        eeg = eeg.to(self.device)
        pred = pred.to(self.device)

        # 交叉熵监督损失
        ce_loss = nn.CrossEntropyLoss()(pred, target)

        # 生理约束损失（多通道支持）
        spectral_loss = self.compute_spectral_loss(eeg, target)
        spindle_loss = self.compute_spindle_loss(eeg, target)

        # 组合损失
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
            random_state=42  # 可重复性种子
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
        # 优化配置
        optimizer = optim.AdamW([
            {'params': self.model.cnn.parameters(), 'lr': 1e-4},
            {'params': self.model.lstm.parameters(), 'lr': 1e-3},
            {'params': self.model.classifier.parameters(), 'lr': 5e-3}
        ], weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # 训练记录
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_acc = 0.0

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

            # 学习率调整
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')

            # 记录历史
            history['train_loss'].append(train_loss / len(train_loader.dataset))
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch {epoch + 1}: '
                  f'Train Loss: {history["train_loss"][-1]:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f"Acc: {val_acc:.4f} | "
                  f"mF1: {val_metrics['mf1']:.4f} | "
                  f"Kappa: {val_metrics['k']:.4f} | "
                  f"G-mean: {val_metrics['mgm']:.4f} | "
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        return history

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss, ce_loss, spectral_loss, spindle_loss = self.criterion(outputs, labels, inputs)

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels.squeeze()).sum().item()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        # 计算损失和准确率
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        mf1 = f1_score(all_labels, all_preds, average='macro')
        k = cohen_kappa_score(all_labels, all_preds)
        mgm = geometric_mean_score(all_labels, all_preds, average='macro')

        # 保持原有返回结构 + 新增指标
        return avg_loss, accuracy, {
            'mf1': mf1,
            'k': k,
            'mgm': mgm
        }


# 主程序 ----------------------------------------------------------------
if __name__ == "__main__":
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # # 加载数据
    X, y = pre_data.loadedf.loaddata()
    X = np.transpose(X, (0, 2, 1))

    # 数据预处理
    preprocessor = EEGPreprocessor()
    trainer = Trainer(EEGHybridModel(), device, preprocessor)
    train_dataset, val_dataset = trainer.prepare_data(X, y)

    # 创建DataLoader
    batch_size = 512
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )

    # 训练模型
    history = trainer.train(train_loader, val_loader, epochs=100)

    # # 可视化结果
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_loss'], label='Val Loss')
    # plt.title('Loss Curves')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(history['val_acc'], label='Validation Accuracy')
    # plt.title('Accuracy Curve')
    # plt.legend()
    # plt.show()

    # 加载最佳模型
    best_model_path = 'best_model.pth'
    trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model from {best_model_path}")

    # 使用最佳模型评估验证集
    val_loss, val_acc, val_metrics = trainer.evaluate(val_loader)

    # 打印最佳模型的性能报告
    print("\n--- Best Model Performance Report ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Macro F1 Score: {val_metrics['mf1']:.4f}")
    print(f"Cohen's Kappa: {val_metrics['k']:.4f}")
    print(f"Geometric Mean: {val_metrics['mgm']:.4f}")

    # 生成混淆矩阵和分类报告
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trainer.model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # 分类报告
    target_names = [f"Class {i}" for i in range(len(set(all_labels)))]
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\nClassification Report:")
    print(report)