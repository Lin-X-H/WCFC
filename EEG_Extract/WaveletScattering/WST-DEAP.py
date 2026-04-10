import os
import pickle
from kymatio.torch import Scattering1D
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))

save_dir = 'AfterWST/DEAP/sub'
use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

for i in range(1, 33):
    # 加载预处理后数据
    with open('../Preprocessing/ProcessedData/DEAP/s' + str(i).zfill(2) + '.pkl', 'rb') as f:
        sub = pickle.load(f)
        result = [] # 存储当前被试所有试次的散射系数

        # 遍历每个被试的40个试次
        for k in range(40):
            fs = 128
            sub_tensor = torch.from_numpy(sub[k]) # 将numpy数组转换为张量
            sub_tensor = sub_tensor[:, 3*fs:27*fs] # 截取从第3秒到第27秒，去除开始和结束的潜在噪声
            sub_tensor = sub_tensor.type(torch.float32).to(device)

            # 设置小波散射变换参数
            J = 7  # 散射尺度（最大尺度为2^7=128）
            N = 3072   # 信号长度（24秒×128Hz=3072个采样点）
            Q = 8   # 每个八度中的小波数量

            # 常见小波散射变换对象
            S = Scattering1D(J, N, Q).to(device)
            # 应用散射变换，提取时频特征
            Sx = S.scattering(sub_tensor).to(device)


            # 排除第零阶散射系数（原始信号），只保留一阶和高阶系数
            Sx = Sx[:, 1:, :]

            # 输出格式(通道*散射系数*时间)
            result.append(Sx.unsqueeze(0))

    result = torch.cat(result, dim=0)   # 合并所有trail
    save_path = save_dir+str(i).zfill(2) + 'WST.pt'
    if not os.path.exists(save_dir[:-4]):
        os.makedirs(save_dir[:-4])
    torch.save(result, save_path)
    print(f'被试 {i:02d} 处理完成：')
    print(result.shape)
    print(type(result))
    print("-"*50)
