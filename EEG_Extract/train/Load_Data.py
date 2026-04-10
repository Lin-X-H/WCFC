import random
import numpy as np
import torch
import os
from numpy.ma.core import indices
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
EEG_ROOT = os.path.dirname(CURRENT_DIR)               # 项目根目录路径

def worker_init_fn(worked_id, seed=42):
    """
    数据加载器工作进程初始化函数
    用于确保每个工作进程的随机种子一致，保证数据加载的可重复性
    :param worked_id: 工作进程ID
    :param seed: 随机种子，默认为42
    """
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def loadData(batch_size):
    """
    加载数据并创建数据加载器
    :param batch_size: 批次大小
    :return: dataloaders:包含训练集、验证集和测试集的数据加载器字典
    """
    # 初始化数据存储列表
    train_tensor = [] # 训练集数据
    dev_tensor = []   # 验证集数据
    test_tensor = []  # 测试集数据

    num_subjects = 32 # 总受试者数量，每个受试者有40个样本
    train_num = 20    # 每个受试者的训练样本数：50%
    dev_num = 8       # 每个受试者的验证样本数：20%
    test_num = 12     # 每个受试者的测试样本数：30%

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取数据并进行数据集划分
    for i in range(1, num_subjects + 1):
        # 加载小波散射变换后的数据
        data_path = os.path.join(EEG_ROOT, "WaveletScattering", "AfterWST", "DEAP", "sub{0}WST.pt".format(str(i).zfill(2)))
        # cur_data = torch.load('../WaveletScattering/AfterWST/DEAP/sub' + str(i).zfill(2) + 'WST.pt').to(device)
        cur_data = torch.load(data_path).to(device)

        # 随机打乱当前受试者的数据顺序
        indices = torch.randperm(cur_data.size(0)) # 生成随机索引
        shuffled_tensor = cur_data[indices]        # 根据随机索引重新排列数据

        # 按比例划分数据集
        train_tensor.append(shuffled_tensor[:train_num])                   # 前20个样本作为训练集
        dev_tensor.append(shuffled_tensor[train_num:train_num + dev_num])  # 中间8个样本作为验证集
        test_tensor.append(shuffled_tensor[train_num + dev_num:])          # 最后12个样本作为测试集

    # 将所有受试者的数据合并成完整的数据集
    train_tensor = torch.cat(train_tensor, dim=0) # [32*20, ...] = [640, ...]
    dev_tensor = torch.cat(dev_tensor, dim=0)     # [32*8 , ...] = [256, ...]
    test_tensor = torch.cat(test_tensor, dim=0)   # [32*12, ...] = [384, ...]

    # 生成对应的标签，每个受试者对应一个数字标签（0-31）
    train_labels = torch.arange(0, num_subjects).repeat_interleave(train_num, dim=0).unsqueeze(1) # [640, 1]
    dev_labels = torch.arange(0, num_subjects).repeat_interleave(dev_num, dim=0).unsqueeze(1)    # [256, 1]
    test_labels = torch.arange(0, num_subjects).repeat_interleave(test_num, dim=0).unsqueeze(1)   # [384, 1]

    # 创建数据集
    train_dataset = TensorDataset(train_tensor, train_labels)
    dev_dataset = TensorDataset(dev_tensor, dev_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, num_workers=0),
        'dev': DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, num_workers=0)
    }
    return dataloaders