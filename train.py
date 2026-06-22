# train.py

import numpy as np
from glob import glob
# from tqdm.notebook import tqdm # 如果在Jupyter Notebook中运行，使用这个
from tqdm import tqdm  # 在标准终端中运行，使用这个
from sklearn.metrics import confusion_matrix
import random
import time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *  # 确保 utils.py 在同一个目录下
from torch.autograd import Variable
from IPython.display import clear_output
from model.vitcross_seg_modeling import VisionTransformer as ViT_seg
from model.vitcross_seg_modeling import CONFIGS as CONFIGS_ViT_seg

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
print("Device :", nvmlDeviceGetName(handle))

# 模型配置和加载
config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
config_vit.n_classes = N_CLASSES  # 使用utils中定义的类别数
config_vit.n_skip = 3
config_vit.patches.grid = (int(WINDOW_SIZE[0] / 16), int(WINDOW_SIZE[1] / 16))
# 【重要】确保预训练权重的路径是正确的
config_vit.pretrained_path = '/home/zhangyubo/SSRS-main/pretrain/imagenet21k_R50+ViT-B_16.npz'
net = ViT_seg(config_vit, img_size=WINDOW_SIZE[0], num_classes=N_CLASSES).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))

params = sum(p.numel() for p in net.parameters())
print(f"模型总参数量: {params / 1e6:.2f}M")

# 加载数据集
print("训练集ID: ", train_ids)
print("测试集ID: ", test_ids)
print("BATCH_SIZE: ", BATCH_SIZE)
print("滑窗步长: ", Stride_Size)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

# 优化器和学习率调度器设置
base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder 权重使用基准学习率
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder 权重使用较小的学习率
        params += [{'params': [value], 'lr': base_lr / 10}]  # Encoder学习率通常设置更小

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 35, 45], gamma=0.1)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # 【已修改】在测试集上评估网络
    if DATASET == 'Potsdam':
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in
                       test_ids)
    else:
        test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)

    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []

    # 切换到评估模式
    net.eval()
    with torch.no_grad():  # 使用 no_grad 替代 Variable(volatile=True)
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids),
                                       desc="Testing on tiles"):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         desc="Sliding window", leave=False)):
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = torch.from_numpy(np.asarray(image_patches)).cuda()

                min_val, max_val = np.min(dsm), np.max(dsm)
                dsm_norm = (dsm - min_val) / (max_val - min_val + 1e-8)
                dsm_patches = [np.copy(dsm_norm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = torch.from_numpy(np.asarray(dsm_patches)).cuda()

                # 推理
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # 填充结果
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)

    # clear_output() # 在标准脚本中可以注释掉此行

    # 计算评估指标
    # 【已修改】接收返回的指标字典
    metrics_report = metrics(np.concatenate([p.ravel() for p in all_preds]),
                             np.concatenate([p.ravel() for p in all_gts]).ravel())

    if all:
        return metrics_report, all_preds, all_gts
    else:
        return metrics_report


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    # 【已修改】训练函数
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)  # 这个没有被直接使用，实际用的是下面的CrossEntropy2d
    iter_ = 0
    best_miou = 0  # 使用 mIoU 来跟踪最佳模型

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            data, dsm, target = data.cuda(), dsm.cuda(), target.cuda()
            optimizer.zero_grad()

            output = net(data, dsm)
            loss = CrossEntropy2d(output, target, weight=weights)

            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()  # 使用 .item() 获取标量值
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print(f'Train (Epoch {e}/{epochs}) [{(batch_idx + 1) * BATCH_SIZE}/{len(train_set)}] | '
                      f'Loss: {loss.item():.6f} | Mean Loss: {mean_losses[iter_]:.6f} | '
                      f'Batch Acc: {accuracy(pred, gt):.2f}%')

            iter_ += 1
            del data, dsm, target, loss, output

            # --- 【验证和模型保存逻辑已更新】 ---
            if iter_ % 500 == 0:
                print("\n" + "=" * 40)
                print(f"Validation at iteration {iter_}...")

                # 执行验证
                metrics_report = test(net, test_ids, all=False, stride=Stride_Size)

                net.train()  # 切换回训练模式

                # 使用 mIoU 作为模型保存的依据
                current_miou = metrics_report['mIoU']
                if current_miou > best_miou:
                    print(
                        f"🚀 New best model found! mIoU improved from {best_miou * 100:.2f}% to {current_miou * 100:.2f}%. Saving model...")
                    best_miou = current_miou
                    # 保存模型，文件名中包含 mIoU 分数
                    save_path = f'/home/zhangyubo/SSRS-main/ours/gaijin/segnet256_epoch{e}_miou{best_miou:.4f}.pth'
                    torch.save(net.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
                else:
                    print(f"Validation mIoU: {current_miou * 100:.2f}%. (Best mIoU: {best_miou * 100:.2f}%)")
                print("=" * 40 + "\n")

    print(f'🎉 Training finished! Best mIoU achieved: {best_miou * 100:.2f}%')
# #

####   主程序   ####
if __name__ == '__main__':
    # 创建保存模型的目录
    os.makedirs('', exist_ok=True)

    # 训练
    print("----------- Starting Training -----------")
    time_start = time.time()
    train(net, optimizer, 70, scheduler)
    time_end = time.time()
    print(f'Total Training Time: {(time_end - time_start) / 3600:.2f} hours')

    # 测试 (示例)
    print("\n----------- Starting Final Testing -----------")
    # 加载你最好的模型
    best_model_path = ''
    net.load_state_dict(torch.load(best_model_path))
    metrics_report, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
    print("Final Test Results:")
    print(f"Overall Accuracy: {metrics_report['OA']*100:.2f}%")
    print(f"Mean IoU: {metrics_report['mIoU']*100:.2f}%")
    print(f"Mean F1-Score: {metrics_report['F1']*100:.2f}%")
