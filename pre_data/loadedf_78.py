import numpy as np
import os


def loaddata():
    # 设置文件夹路径
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建数据目录的绝对路径
    data_dir = os.path.join(script_dir, '..', 'data_edf_78_npz', 'fpzcz')
    file_names = [f for f in os.listdir(data_dir) if f.endswith('.npz')]  # 获取所有的 .npz 文件

    # 初始化列表用于存储所有的 X 和 y
    X_list = []
    y_list = []

    # 遍历文件并加载数据
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)

        # 加载 .npz 文件
        data = np.load(file_path)

        # 提取 EEG 数据和标签
        eeg_signal = data['x']  # EEG 数据，形状是 (n_epochs, n_samples_per_epoch)
        sleep_stages = data['y']  # 标签，形状是 (n_epochs,)

        # 扩展 EEG 数据的中间维度（第二个维度）
        eeg_signal_expanded = np.expand_dims(eeg_signal, axis=1)  # 形状变为 (n_epochs, 1, n_samples_per_epoch)

        X_list.append(eeg_signal_expanded)
        y_list.append(sleep_stages)

    # 将所有的数据合并成一个大数组
    X = np.concatenate(X_list, axis=0)  # 合并所有样本
    y = np.concatenate(y_list, axis=0)  # 合并所有标签

    return X, y

# X, y = loaddata()
# # 输出最终的 X 和 y 格式
# print("X shape:", X.shape)  # 输出 X 的形状
# print("y shape:", y.shape)  # 输出 y 的形状
