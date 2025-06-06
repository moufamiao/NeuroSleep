import argparse
import glob
import math
import ntpath
import os
import shutil


from datetime import datetime

import numpy as np

from mne.io import read_raw_edf

from pre_data import dhedfreader

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

# Sleep stage dictionary
stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

# Class dictionary for labeling
class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

# Annotation to label mapping
ann2label = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,  # Treat Sleep stage 4 as N3
    "Sleep stage R": REM,
    "Sleep stage ?": UNKNOWN,
    "Movement time": UNKNOWN
}

# Epoch size in seconds (default 30 seconds per epoch)
EPOCH_SEC_SIZE = 30

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\moufamiao\Desktop\sleep\data\78",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="data_edf_78_npz/fpzcz",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz",
                        help="The selected channel")
    args = parser.parse_args()

    # Output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Selected channel
    select_ch = args.select_ch

    # Read PSG and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        # 读取原始PSG数据
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info['sfreq']
        raw_ch_df = raw.to_data_frame()[select_ch]  # 不使用scaling_time参数

        # 如果需要缩放数据，可以手动缩放
        raw_ch_df = raw_ch_df / 100.0  # 假设需要将数据缩放到100

        # Get raw header
        f = open(psg_fnames[i], 'r', errors='ignore')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

        # Read annotation header and records
        f = open(ann_fnames[i], 'r', errors='ignore')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
        f.close()
        ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []    # indices to remove
        labels = []        # list of labels
        label_idx = []     # indices with labels
        # 生成标签并移除不需要的索引
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label.get(ann_str[2:-1], UNKNOWN)  # 如果找不到标签，使用 UNKNOWN
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                label_idx.append(idx)
                print("Include onset:{}, duration:{}, label:{} ({})".format(onset_sec, duration_sec, label, ann_str))
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                remove_idx.append(idx)
                print("Remove onset:{}, duration:{}, label:{} ({})".format(onset_sec, duration_sec, label, ann_str))

        # 检查 labels 是否为空
        if len(labels) > 0:
            labels = np.hstack(labels)
        else:
            raise ValueError("No valid labels found. Check your annotation processing logic.")

        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra labels if any
        if len(label_idx) > len(select_idx):
            print("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
            extra_idx = np.setdiff1d(label_idx, select_idx)
            n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
            if n_label_trims != 0:
                labels = labels[:-n_label_trims]
            print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-second epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) // (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # Select sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        # Save the processed data
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        print("\n=======================================\n")

if __name__ == "__main__":
    main()
