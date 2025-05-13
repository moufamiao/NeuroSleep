import numpy as np


def load_npy_file(file_path):
    """
    Load data from a .npy file.

    Parameters:
    - file_path: str, the path to the .npy file.

    Returns:
    - data: numpy.ndarray, the loaded data.
    """
    try:
        data = np.load(file_path)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def main():
    # 指定你的 .npy 文件路径
    file_path = r"C:\Users\moufamiao\Desktop\pycharm_data\test\data_shhs_npz\fpzcz\r_permute_shhs.npy"

    # 加载数据
    data= load_npy_file(file_path)


    if data is not None:
        print("Data shape:", data.shape)
        print("Data dtype:", data.dtype)
        print("Data size (number of elements):", data.size)
        print("Data item size (bytes per element):", data.itemsize)
        print("Total bytes consumed by the elements of this array:", data.nbytes)
        print("Number of dimensions:", data.ndim)
        print("Memory layout (C-contiguous):", data.flags['C_CONTIGUOUS'])
        print("Memory layout (F-contiguous):", data.flags['F_CONTIGUOUS'])
        print("Array strides (bytes to step in each dimension):", data.strides)
        print("First few elements:")
        print(data[:5])  # 打印前5个元素（假设是一维数组）


if __name__ == "__main__":
    main()