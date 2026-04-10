import logging
import random
import time
import  copy
import numpy as np
import torch
from torch.optim import lr_scheduler
import Load_Data
from torch import nn, optim
import os
import Densenet
from Densenet import *
from train_validate import train_validate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))     # 当前文件绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) # 项目根目录路径

dir_path = os.path.join(PROJECT_ROOT, 'model')
os.makedirs(dir_path, exist_ok = True)

os.environ['TORCH_USE_CUDA_DSA'] = '1'
# 创建main专用logger
main_logger = logging.getLogger("main")
main_logger.setLevel(logging.INFO) # 设置日志级别

# 创建train专用logger
train_logger = logging.getLogger("train")
train_logger.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 为main logger添加文件和控制台处理器
main_file_handler = logging.FileHandler(dir_path+'/main.log')
main_file_handler.setFormatter(formatter)
main_logger.addHandler(main_file_handler)
main_logger.addHandler(logging.StreamHandler()) # 保持控制台输出

train_file_handler = logging.FileHandler(dir_path+'/train.log')
train_file_handler.setFormatter(formatter)
train_logger.addHandler(train_file_handler)
train_logger.addHandler(logging.StreamHandler())


def set_random_seed(seed):
    """
    设置随机种子，确保实验的可复现性
    :param seed: 随机种子值，用于控制所有随机数生成器的初始状态
    :return:
    """

    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 设置python内置随机模块的种子
    random.seed(seed)

    # 设置python哈希种子，影响字典等数据结构的哈希行为
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置Numpy随机数生成器的种子
    np.random.seed(seed)

    # 设置Pytorch CPU随机种子
    torch.manual_seed(seed)

    # 设置Pytorch GPU随机种子（单GPU）
    torch.cuda.manual_seed(seed)
    # 设置Pytorch所有GPU的随机种子（多GPU）
    torch.cuda.manual_seed_all(seed)

    # 关闭cuDNN的benchmark模式
    # benchmark=True时会自动寻找最优的卷积算法，但会导致不确定性
    torch.backends.cudnn.benchmark = False

    # 启用cuDNN的确定性模式，确保相同的输入总是使用相同的卷积算法
    torch.backends.cudnn.deterministic = True

    # 启用PyTorch的确定性算法，并只在出现非确定性操作时发出警告
    # warn_only=True表示当出现无法使用确定性算法的情况时只警告而不报错
    torch.use_deterministic_algorithms(True, warn_only=True)

def train(filename, isAttention, isCA, isEA, isTA, seed):
    """
    模型训练主函数
    :param filename: 模型保存的文件名
    :param isAttention: 是否使用注意力机制
    :param isCA: 是否使用CA (Channel Attention)通道注意力
    :param isEA: 是否使用EA (External Attention)外部注意力
    :param isTA: 是否使用TA (Temporal Attention)时序注意力
    :param seed: 随机种子
    """
    set_random_seed(seed)

    # 设置Pytorch随机数生成器，用于数据加载器等需要随机性的地方
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device("cuda:0")

    # 训练超参数配置
    total_epoch = 1000 # 训练总轮数
    batch_size = 64    # 每次取出样本数
    Lr = 0.1           # 初始学习率

    # 加载数据
    dataloaders = Load_Data.loadData(batch_size)

    # 初始化模型，根据参数配置不同的注意力机制
    model_ft = Densenet.densenet(isAttention, isCA, isEA, isTA)
    model = model_ft.to(device)

    # 创建损失函数 - 交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    # 记录模型需要训练的参数层
    train_logger.info("==============当前模型要训练的层==============")
    for name, params in model_ft.named_parameters():
        if params.requires_grad:
            train_logger.info(name)

    # =============== 训练状态变量初始化 ===============
    counter = 0      # 记录验证损失连续未改善的epoch数
    counter_stop = 0 # 早停计数器，连续未改善达到阈值则停止训练

    total_step = {'train': 0, 'dev': 0} # 记录训练和验证的总步数

    since = time.time() # 记录开始时间

    valid_loss_min = np.inf # 记录当前最小损失值，初始最小验证损失为无穷大

    best_acc = 0 # 保存最优正确率

    save_name_t = '' # 保存的文件名

    # 定义优化器和学习率调度器
    # SGD优化器，动量0.9，L2权重衰减5e-4
    optimizer = optim.SGD(model_ft.parameters(), lr=Lr, momentum=0.9, weight_decay=5e-4)
    # 基于验证损失调整学习率，当损失停止改善时学习率减半，耐心值为25epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50)

    # =============== 主循环训练 ===============
    for epoch in range(total_epoch):
        train_logger.info('Epoch {}/{}'.format(epoch + 1, total_epoch))
        train_logger.info('-' * 10)
        train_logger.info('')

        # 早停检查：如果连续100个epoch验证损失没有改善，停止训练
        if counter_stop == 150:
            break

        # 训练和验证 每一轮都是先训练train，再验证valid
        for phase in ['train', 'dev']:
            if phase == 'train':
                model_ft.train() # 训练
            else:
                model_ft.eval() # 验证

            # 执行训练或验证，返回该epoch的平均损失和准确率
            epoch_loss, epoch_acc = train_validate(
                model_ft, loss_fn,  optimizer, dataloaders, phase, device
            )
            total_step[phase] += 1 # 更新步数计数器

            # 计算并记录训练耗时
            time_elapsed = time.time() - since
            train_logger.info('')
            train_logger.info('当前总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            train_logger.info('{} Loss: {:.4f}[{}] Acc: {:.4f}'.format(phase, epoch_loss, counter, epoch_acc))

            # ========== 验证阶段特殊处理 ==========
            if phase == 'dev':
                # 基于验证损失更新学习率
                scheduler.step(epoch_loss)
                # 检查是否得到更好的模型（验证损失降低）
                if epoch_acc > best_acc: # epoch_loss < valid_loss_min:
                    best_acc = epoch_acc
                    # valid_loss_min = epoch_loss

                    # 深度拷贝当前模型参数
                    best_model_wts = copy.deepcopy(model_ft.state_dict())

                    state = {
                        'state_dict': model_ft.state_dict(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.state_dict(),
                    }
                    # 只保存最近1次的训练结果

                    save_name_t = '{}/{}.pth'.format(dir_path, filename)
                    torch.save(state, save_name_t)
                    train_logger.info("已保存最优模型，准确率：\033[1;31m {:.2f}%\033[0m，文件名：{}".format(best_acc * 100, save_name_t))

                    # 重置计数器，更新最小验证损失
                    valid_loss_min = epoch_loss
                    counter = 0
                    counter_stop = 0
                else:
                    # 验证损失没有改善，增加计数器
                    counter += 1
                    counter_stop += 1

        train_logger.info('')
        train_logger.info('当前学习率 : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        train_logger.info('')

    # =============== 训练结束后测试最佳模型 ===============
    # 加载训练过程中保存的最佳模型
    model_ft.load_state_dict(torch.load('{}/{}.pth'.format(dir_path,filename))['state_dict'])
    # 在测试集上评估最佳模型性能
    epoch_loss, epoch_acc = train_validate(model_ft, loss_fn, optimizer, dataloaders, 'test', device)


    time_elapsed = time.time() - since
    train_logger.info('')
    train_logger.info('任务完成！')
    train_logger.info('任务完成总耗时 {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    train_logger.info('最高验证集准确率: {:4f}'.format(best_acc))
    train_logger.info('测试集loss:{:4f}'.format(epoch_loss))
    train_logger.info('测试集准确率: {:4f}'.format(epoch_acc))

    save_name_percentage = save_name_t[:-4] + f'ACC{epoch_acc:.4f}.pth'
    os.rename(save_name_t, save_name_percentage)
    logging.info('最优模型保存在：{}'.format(save_name_t))

    return epoch_loss, epoch_acc

if __name__ == '__main__':
    CAADN_loss, CAADN_ACC = train('Noabl(CAADN)', True, True, True, True, 520)
    main_logger.info('Noabl(CAADN):' + str(CAADN_loss) + 'ACC:' + str(CAADN_ACC))
