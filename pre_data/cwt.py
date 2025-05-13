import pre_data.loadedf
import numpy as np
import pywt
from scipy.signal import resample
from tqdm import tqdm
import os
import time

# 加载数据
X, y = pre_data.loadedf.loaddata()  # 假设 X.shape = (42308, 1, 3000)
n, c, t = X.shape
fs = 100  # 采样率 100 Hz

scales = np.arange(1, 65)  # 128 个尺度
wavelet = 'cmor'  # 复 Morlet 小波
batch_size = 256  # 每次处理 256 个样本
target_time_dim = 64  # 目标时间维度

# 结果存储
cwt_result_list = []

# 生成带时间戳的唯一文件名
save_path = "cwt_result"

# 带进度条的处理
with tqdm(total=n, desc="CWT Transform", unit="sample",
         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    # [...] 循环代码不变
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        batch_size_actual = batch.shape[0]

        batch_cwt = np.zeros((batch_size_actual, len(scales), target_time_dim), dtype=np.float16)

        for j in range(batch_size_actual):
            coeffs, _ = pywt.cwt(batch[j, 0, :], scales, wavelet, 1 / fs)
            batch_cwt[j] = resample(np.abs(coeffs), target_time_dim, axis=1)

        cwt_result_list.append(batch_cwt)
        pbar.update(batch_size_actual)

# 合并所有批次结果
cwt_result = np.concatenate(cwt_result_list, axis=0)

# 保存为numpy文件
np.save(save_path, cwt_result)
print(f"\n转换完成！数据已保存至: {os.path.abspath(save_path)}")
print("最终数据形状:", cwt_result.shape)