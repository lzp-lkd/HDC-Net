# utils.py (已更新，以满足您的需求)

import numpy as np
import torch
import torch.nn.functional as F
import itertools
import random
import os
from skimage import io
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision.utils import make_grid

# -----------------------------------------------------------------------------
# 1. 全局参数定义 (Global Parameters)
# -----------------------------------------------------------------------------

# 数据集根目录
FOLDER = ""
# 数据集特定参数
# Vaihingen 数据集
DATASET = 'Vaihingen'
train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids = ['5', '21', '15', '30']
MAIN_FOLDER = FOLDER + 'Vaihingen/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'


# # Potsdam 数据集 (如果需要，取消下面的注释)
# train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
#              '4_12', '6_8', '6_12', '6_7', '4_11']
# test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
# DATASET = 'Potsdam'
# Stride_Size = 128 # for quickly training
# MAIN_FOLDER = FOLDER + 'Potsdam/'
# DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
# DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
# LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
# ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'

# 模型和训练参数
WINDOW_SIZE = (256, 256)
Stride_Size = 32
BATCH_SIZE = 10
IN_CHANNELS = 3
CACHE = True

# 类别定义
# 【重要】将全局变量名从 LABELS 改为 LABEL_NAMES 以增加可读性
LABEL_NAMES = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABEL_NAMES)
WEIGHTS = torch.ones(N_CLASSES)

# ISPRS 标准调色板
palette = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255), 3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0),
           6: (0, 0, 0)}  # 注意：这里的调色板似乎有7个类别，但您的LABELS只有6个。请确保它们匹配。我将以6个类别为准。
invert_palette = {v: k for k, v in palette.items()}


# -----------------------------------------------------------------------------
# 2. 颜色与标签转换函数 (Color and Label Conversion) - [无改动]
# -----------------------------------------------------------------------------

def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


# -----------------------------------------------------------------------------
# 3. PyTorch 数据集类 (PyTorch Dataset Class) - [无改动]
# -----------------------------------------------------------------------------

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=CACHE, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.ids = ids
        self.data_files = [DATA_FOLDER.format(id) for id in self.ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in self.ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in self.ids]
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError(f'{f} is not a file !')
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return BATCH_SIZE * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            if DATASET == 'Potsdam':
                img = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(img[:, :, :3], dtype='float32').transpose((2, 0, 1))
            else:
                img = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(img.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_:
            dsm = self.dsm_cache_[random_idx]
        else:
            dsm_img = io.imread(self.dsm_files[random_idx])
            dsm = np.asarray(dsm_img, dtype='float32')
            min_val, max_val = np.min(dsm), np.max(dsm)
            dsm = (dsm - min_val) / (max_val - min_val + 1e-8)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            label_img = io.imread(self.label_files[random_idx])
            label = np.asarray(convert_from_color(label_img), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))


# -----------------------------------------------------------------------------
# 4. 辅助函数 (Utility Functions) - [无改动]
# -----------------------------------------------------------------------------

def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    output = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    reduction = 'mean' if size_average else 'sum'
    loss = F.cross_entropy(output, target, weight=weight, reduction=reduction)
    return loss


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step, window_size):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step, window_size):
    c = 0
    for _ in sliding_window(top, step, window_size):
        c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABEL_NAMES):
    """
    计算并打印语义分割的各项评估指标。
    【已更新】: 此版本会忽略'clutter'类别来计算平均指标(mIoU, mF1)。

    Args:
        predictions (np.ndarray): 预测的展平标签数组。
        gts (np.ndarray): 真实的展平标签数组。
        label_values (list): 类别名称列表。

    Returns:
        dict: 包含所有计算指标的字典。
    """
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("--- Confusion Matrix ---")
    print(cm)

    # --- 从混淆矩阵计算指标 ---
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    epsilon = 1e-7

    # 1. 计算每个类别的 IoU 和 F1-Score
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    # 2. 总体精度 (OA) - 这个指标不受影响，仍然计算所有像素
    overall_accuracy = np.sum(tp) / (np.sum(cm) + epsilon)

    # --- 【核心修改点】: 仅针对前五个类别计算平均值 ---
    # 找到 'clutter' 类别的索引
    try:
        clutter_idx = label_values.index('clutter')
        # 创建一个布尔掩码，除了 'clutter' 索引外都为 True
        mask = np.ones(len(label_values), dtype=bool)
        mask[clutter_idx] = False

        # 使用掩码来选择要计算平均值的类别
        miou_eval = np.nanmean(iou[mask])
        mean_f1_eval = np.nanmean(f1[mask])

        eval_classes = [name for i, name in enumerate(label_values) if mask[i]]
        print(f"\nCalculating mean metrics for classes: {eval_classes}")

    except ValueError:
        # 如果 'clutter' 不在列表中，则计算所有类别的平均值
        print("\n'clutter' not found. Calculating mean metrics for all classes.")
        miou_eval = np.nanmean(iou)
        mean_f1_eval = np.nanmean(f1)
    # --- 【修改结束】 ---

    # --- 打印结果 ---
    print("\n--- Per-class Metrics ---")
    print("{:^12} | {:^10} | {:^10}".format("Class", "IoU", "F1-Score"))
    print("-" * 38)
    for i, label in enumerate(label_values):
        is_ignored = (label == 'clutter')
        marker = "✘ (ignored)" if is_ignored else "✔"
        print("{:^12} | {:^10.4f} | {:^10.4f}  {}".format(label, iou[i], f1[i], marker))

    print("\n--- Summary Metrics ---")
    print(f"Overall Accuracy (OA): {overall_accuracy * 100:.2f}%")
    # 显示我们计算的、排除了clutter的mIoU和mF1
    print(f"Mean IoU (mIoU):       {miou_eval * 100:.2f}%  (excluding 'clutter')")
    print(f"Mean F1-Score:         {mean_f1_eval * 100:.2f}%  (excluding 'clutter')")

    # --- 返回一个包含关键指标的字典 ---
    # 返回的 'mIoU' 和 'F1' 是我们排除 'clutter' 后的结果
    return {
        'OA': overall_accuracy,
        'mIoU': miou_eval,
        'F1': mean_f1_eval
    }