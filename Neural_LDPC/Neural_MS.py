import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import lil_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # 项目根目录路径

# GPU设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取基图和生成矩阵
BG_PATH = os.path.join(CURRENT_DIR, "BaseGraph", "BaseGraph2_Set2.txt")
GM_PATH = os.path.join(CURRENT_DIR, "BaseGraph_GM", "LDPC_GM_BG2_5.txt")
code_PCM = np.loadtxt(BG_PATH, int, delimiter='\t')
code_GM = np.loadtxt(GM_PATH, int, delimiter=',')
Z = 5

# 码参数
N = 52
m = 42
code_n = N
code_k = N - m

# 将-1转换为0用于计数（保留原始 -1 信息用于移位计算）
code_PCM_for_count = code_PCM.copy()
for i in range(code_PCM_for_count.shape[0]):
    for j in range(code_PCM_for_count.shape[1]):
        if code_PCM_for_count[i, j] == -1:
            code_PCM_for_count[i, j] = 0

# 网络超参数
iters_max = 25  # 迭代次数
sum_edge_c = np.sum(code_PCM != -1, axis=1)
sum_edge_v = np.sum(code_PCM != -1, axis=0)
sum_edge = np.sum(code_PCM != -1)
neurons_per_layer = sum_edge  # 奇偶层神经元数量相同
input_output_layer_size = N

# AWGN信道参数
code_rate = 1.0 * (N - m) / (N - 2)
# # 训练SNR矩阵
# SNR_Matrix = np.array([
#     9.0, 6.05, 4.1, 2.95, 2.25, 1.8, 1.55, 1.3, 1.15, 1.05, 0.94, 0.85, 0.83, 0.81, 0.8, 0.8, 0.8, 0.75, 0.75, 0.7,
#      0.7, 0.7, 0.7, 0.7, 0.7
# ])
# SNR_lin = 10.0 ** (SNR_Matrix / 10.0)
# SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))

# 生成一个混合难度的 SNR 列表
SNR_dB_range = np.linspace(1.5, 4.5, 10)  # 生成 10 个难度等级
SNR_lin = 10.0 ** (SNR_dB_range / 10.0)
SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))

# 随机种子
word_seed = 2042
noise_seed = 1074
wordRandom = np.random.RandomState(word_seed)
noiseRandom = np.random.RandomState(noise_seed)

# 训练设置
learning_rate = 0.005
train_on_zero_word = False
numOfWordSim_train = 256
batch_size = numOfWordSim_train
num_of_batch = 2000

edge_mapping = {}  # 映射 (i,j) -> edge_index
edge_counter = 0

lift_shifts_1 = [] # per-edge shift for Lift_Matrix1 (order of edges)
lift_shifts_2 = [] # per-edge shift for Lift_Matrix2 (order of edges)

# 初始化循环移位的提升矩阵
for j in range(code_PCM.shape[1]):
    for i in range(code_PCM.shape[0]):
        if code_PCM[i, j] != -1:
            Lift_num = int(code_PCM[i, j] % Z)
            lift_shifts_1.append(Lift_num)
            lift_shifts_2.append(Lift_num)
            edge_mapping[(i, j)] = edge_counter
            edge_counter += 1

# 初始化层间连接矩阵
W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
W_output = np.zeros((sum_edge, N), dtype=np.float32)

# 重新计算累积边数用于索引
cumsum_edges_per_row = np.cumsum(np.concatenate(([0], sum_edge_c)))
cumsum_edges_per_col = np.cumsum(np.concatenate(([0], sum_edge_v)))

# 初始化W_odd2even（变量节点更新）
for j in range(code_PCM.shape[1]):
    # 找到该列的所有边
    col_edges = []
    for i in range(code_PCM.shape[0]):
        if code_PCM[i, j] != -1:
            edge_idx = edge_mapping[(i, j)]
            col_edges.append(edge_idx)

    # 对于该列的每条边，连接到同一列的其他边
    for edge_idx in col_edges:
        i, j_pos = None, None
        for key, val in edge_mapping.items():
            if val == edge_idx:
                i, j_pos = key
                break

        if i is None:
            continue

        # 连接到同一列的其他边（除了自己）
        for other_edge_idx in col_edges:
            if other_edge_idx != edge_idx:
                W_odd2even[other_edge_idx, edge_idx] = 1.0

# 初始化W_even2odd（校验节点更新）
for i in range(code_PCM.shape[0]):
    # 找到该行的所有边
    row_edges = []
    for j in range(code_PCM.shape[1]):
        if code_PCM[i, j] != -1:
            edge_idx = edge_mapping[(i, j)]
            row_edges.append(edge_idx)

    # 对于该行的每条边，连接到同一行的其他边
    for edge_idx in row_edges:
        for other_edge_idx in row_edges:
            if other_edge_idx != edge_idx:
                W_even2odd[edge_idx, other_edge_idx] = 1.0

# 初始化W_output（输出层）
for j in range(code_PCM.shape[1]):
    for i in range(code_PCM.shape[0]):
        if code_PCM[i, j] != -1:
            edge_idx = edge_mapping[(i, j)]
            W_output[edge_idx, j] = 1.0

# 初始化W_skipconn2even（信道输入）
for j in range(code_PCM.shape[1]):
    for i in range(code_PCM.shape[0]):
        if code_PCM[i, j] != -1:
            edge_idx = edge_mapping[(i, j)]
            W_skipconn2even[j, edge_idx] = 1.0

# 转换为PyTorch张量
W_odd2even = torch.tensor(W_odd2even, device=device, dtype=torch.float32)
W_skipconn2even = torch.tensor(W_skipconn2even, device=device, dtype=torch.float32)
W_even2odd = torch.tensor(W_even2odd, device=device, dtype=torch.float32)
W_output = torch.tensor(W_output, device=device, dtype=torch.float32)

connect_indices = []
for edge_idx in range(W_even2odd.shape[0]):
    conn = torch.nonzero(W_even2odd[edge_idx], as_tuple=False).squeeze()
    if conn.numel() == 0:
        connect_indices.append(torch.empty((0,), dtype=torch.long, device=device))
    else:
        conn = conn.to(torch.long).to(device)
        if conn.dim() == 0:
            conn = conn.unsqueeze(0)
        connect_indices.append(conn)

lift_shifts_1 = torch.tensor(lift_shifts_1, dtype=torch.long, device=device)
lift_shifts_2 = torch.tensor(lift_shifts_2, dtype=torch.long, device=device)

def create_mix_epoch(scaling_factors, wordRandom, noiseRandom, total_samples, code_n, code_k, Z, code_GM,
                     is_zeros_word, to_device=False):
    """生成混合SNR的训练样本"""
    print(f"开始生成数据: 总样本数 {total_samples}, SNR种类数 {len(scaling_factors)}")

    X = np.zeros([total_samples, code_n * Z], dtype=np.float32)
    Y = np.zeros([total_samples, code_n * Z], dtype=np.int64)

    # 这里的 total_samples 已经是 batch_size * num_of_batch
    # 我们将样本分摊给不同的 SNR
    num_snr = len(scaling_factors)
    samples_per_snr = total_samples // num_snr

    current_idx = 0

    for i, sf_i in enumerate(scaling_factors):
        # 计算当前SNR生成的样本量（处理除不尽的情况，最后一次补全）
        if i == num_snr - 1:
            n_samples = total_samples - current_idx
        else:
            n_samples = samples_per_snr

        if n_samples <= 0: continue

        # 生成码字
        if is_zeros_word:
            infoWord_i = np.zeros((n_samples, code_k * Z), dtype=np.int64)
        else:
            infoWord_i = wordRandom.randint(0, 2, size=(n_samples, code_k * Z))

        # 编码
        Y_i = np.dot(infoWord_i, code_GM) % 2

        # 加噪
        X_p_i = noiseRandom.normal(0.0, 1.0, Y_i.shape) * sf_i + (-1) ** (1 - Y_i)
        x_llr_i = 2 * X_p_i / (sf_i ** 2)

        # 存入大数组
        X[current_idx: current_idx + n_samples] = x_llr_i
        Y[current_idx: current_idx + n_samples] = Y_i

        current_idx += n_samples
        print(f"  - 已生成 SNR sigma={sf_i:.3f} 的样本 {n_samples} 个")

        # 打乱数据顺序 (Shuffle)，防止模型在一个 batch 里只看到同一种 SNR
        # 注意：为了保持 X 和 Y 对应，使用相同的 permutation
    perm = np.random.permutation(total_samples)
    X = X[perm]
    Y = Y[perm]

    X = X.reshape([-1, code_n, Z]).astype(np.float32)
    Y = Y.astype(np.int64)

    print(f"数据生成完成 - X形状: {X.shape}, 标签中1的比例: {Y.mean():.3f}")

    if to_device:
        X = torch.tensor(X, dtype=torch.float32, device=device)
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
    return X, Y

class selfLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        bce_loss = self.bce(outputs, targets)

        # 添加正则化项 - 鼓励更确定的输出
        probs = torch.sigmoid(outputs)
        entropy = - (probs * torch.log(probs + 1e-8) +
                     (1 - probs) * torch.log(1 - probs + 1e-8))
        entropy_loss = entropy.mean()

        return bce_loss + self.alpha * entropy_loss


class LDPCNeuralDecoder(nn.Module):
    """LDPC神经网络译码器"""

    def __init__(self, num_iters, num_edges, connect_indices, lift_shifts_1, lift_shifts_2):
        super(LDPCNeuralDecoder, self).__init__()
        self.num_iters = num_iters
        self.num_edges = num_edges

        # 可学习参数
        self.weights = nn.ParameterList([
            nn.Parameter(torch.full((num_edges,), 0.5, device=device))
            for _ in range(num_iters)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_edges, device=device))
            for _ in range(num_iters)
        ])
        self.scale_factors = nn.ParameterList([
            nn.Parameter(torch.ones(num_edges, device=device))
            for _ in range(num_iters)
        ])
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, device=device))
            for _ in range(num_iters)
        ])

        self.loss_fn = selfLoss(alpha=0.05)

        self.register_buffer('lift_shifts_1', lift_shifts_1)
        self.register_buffer('lift_shifts_2', lift_shifts_2)
        self.connect_indices = connect_indices

        # 添加层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([Z, num_edges]) for _ in range(num_iters)
        ])

        # 预计算向量化索引
        # 移位索引构建，构建一个 gather index，一次性完成所有边的 roll 操作
        # Target shape: [1, Z, num_edges]
        shift_idx_1 = torch.zeros(1, Z, num_edges, dtype=torch.long, device=device)
        shift_idx_2 = torch.zeros(1, Z, num_edges, dtype=torch.long, device=device)

        for e in range(num_edges):
            s1 = int(lift_shifts_1[e].item())
            s2 = int(lift_shifts_2[e].item())
            # gather index 公式: idx[z] = (z + s) % Z
            shift_idx_1[0, :, e] = torch.arange(Z, device=device).roll(-s1)  # gather 实际上是在选索引，所以要对应
            shift_idx_1[0, :, e] = (torch.arange(Z, device=device) + s1) % Z

            # 反向移位：shifts=s (右移) -> x[(i-s)%Z]
            shift_idx_2[0, :, e] = (torch.arange(Z, device=device) - s2) % Z

        self.register_buffer('shift_idx_1', shift_idx_1)
        self.register_buffer('shift_idx_2', shift_idx_2)

        # 校验节点聚合索引构建
        # 找出最大的连接度数 (max degree)
        degrees = [len(c) for c in connect_indices]
        max_degree = max(degrees) if degrees else 0

        # cn_gather_idx: [num_edges, max_degree]
        # 用于gather那些连接到同一 CN 的其他边
        cn_gather_idx = torch.zeros(num_edges, max_degree, dtype=torch.long, device=device)
        # cn_mask: [num_edges, max_degree] - 用于标记 padding 的位置
        cn_mask = torch.zeros(num_edges, max_degree, dtype=torch.bool, device=device)

        for e in range(num_edges):
            conn = connect_indices[e]  # Tensor of indices
            deg = len(conn)
            if deg > 0:
                cn_gather_idx[e, :deg] = conn
                cn_mask[e, :deg] = True

        self.register_buffer('cn_gather_idx', cn_gather_idx)
        self.register_buffer('cn_mask', cn_mask)  # True 表示有效数据，False 表示 Padding

    def calculate_accuracy(self, outputs, targets):
        with torch.no_grad():
            pred = torch.sigmoid(outputs) > 0.5
            correct = (pred == targets.bool()).float()
            accuracy = correct.mean().item()
        return accuracy

    def calculate_ber(self, outputs, targets):
        with torch.no_grad():
            pred = torch.sigmoid(outputs) > 0.5
            errors = (pred != targets.bool()).float()
            ber = errors.mean().item()
        return ber

    def calculate_fer(self, outputs, targets, error_threshold=0.02):
        with torch.no_grad():
            pred = (torch.sigmoid(outputs) > 0.5).long()
            targets_int = targets.long()
            error_ratio = (pred != targets_int).float().mean(dim=1)
            frame_errors = (error_ratio > error_threshold).float()
            fer = frame_errors.mean().item()
        return fer

    def forward(self, x):
        """
        x: (batch_size, N, Z)
        """
        batch_size, N, Z = x.shape
        num_edges = self.num_edges

        # 初始化LLR
        llr = torch.zeros((batch_size, Z, num_edges), device=device)
        output = None

        x_input = x.transpose(1, 2)  # (batch, Z, N)
        x0 = torch.matmul(x_input, W_skipconn2even)  # (batch, Z, sum_edge) -> 矩阵乘法本身就是向量化的

        residual = x0

        # 为了 gather 方便，扩展维度
        # shift_idx 扩展为 [B, Z, E]
        batch_shift_idx_1 = self.shift_idx_1.expand(batch_size, -1, -1)
        batch_shift_idx_2 = self.shift_idx_2.expand(batch_size, -1, -1)

        # cn_gather_idx 扩展为 [B, Z, E, max_degree]
        # x2_shifted 是 [B, Z, E]
        # 需要 gather 出来的结果是 [B, Z, E, max_degree]
        cn_idx_expanded = self.cn_gather_idx.view(1, 1, num_edges, -1).expand(batch_size, Z, -1, -1)

        # Mask 只需要 [1, 1, E, max_degree] 用于 broadcasting
        cn_mask_expanded = self.cn_mask.view(1, 1, num_edges, -1)

        for i in range(self.num_iters):
            # 变量节点更新
            x1 = torch.matmul(llr, W_odd2even)
            x2 = x0 + x1
            x2 = x2 + self.residual_weights[i] * residual
            x2 = self.layer_norms[i](x2)

            # 循环移位
            # x2: [B, Z, E]
            x2_shifted = torch.gather(x2, 1, batch_shift_idx_1)

            # 校验节点更新
            # 从 x2_shifted 中聚合邻居信息
            # gather source: [B, Z, E], index: [B, Z, E, max_degree] -> out: [B, Z, E, max_degree]
            # 我们要在 dim=2 (edge维度) 上取值
            neighbors = torch.gather(x2_shifted.unsqueeze(3).expand(-1, -1, -1, cn_idx_expanded.shape[3]), 2, cn_idx_expanded)

            # 处理 Padding：Min 算子填无穷大，Prod 算子填 1
            neighbors_abs = neighbors.abs()
            neighbors_abs = neighbors_abs.masked_fill(~cn_mask_expanded, 1e9)  # 填充无穷大

            neighbors_sign = torch.sign(neighbors)
            neighbors_sign = neighbors_sign.masked_fill(~cn_mask_expanded, 1.0)  # 填充1

            # 聚合计算
            min_abs = torch.min(neighbors_abs, dim=3).values  # [B, Z, E]
            prod_signs = torch.prod(neighbors_sign, dim=3)  # [B, Z, E]

            x_output_0 = min_abs * (-prod_signs)

            # 反向移位 (向量化 Gather)
            x_output_0_unshift = torch.gather(x_output_0, 1, batch_shift_idx_2)

            # 激活输出
            weighted_abs = torch.abs(x_output_0_unshift) * self.weights[i] * self.scale_factors[i]
            weighted_abs = weighted_abs + self.biases[i]
            x_output_1 = F.leaky_relu(weighted_abs, negative_slope=0.1)
            x_output_1 = torch.clamp(x_output_1, -10, 10)

            llr = x_output_1 * torch.sign(x_output_0_unshift)
            residual = x_output_0_unshift

            # 计算最终输出
            y_output_2 = torch.matmul(llr, W_output)
            y_output_3 = y_output_2.transpose(1, 2)
            y_output_4 = x + y_output_3
            output = y_output_4.reshape(batch_size, N * Z)

        return output


def save_model(model, filepath):
    """保存训练好的模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_iters': model.num_iters,
            'num_edges': model.num_edges,
            'connect_indices': model.connect_indices,
            'lift_shifts_1': model.lift_shifts_1,
            'lift_shifts_2': model.lift_shifts_2
        },
        'training_config': {
            'iters_max': iters_max,
            'sum_edge': sum_edge,
            'Z': Z,
            'code_n': code_n,
            'code_k': code_k
        }
    }, filepath)
    print(f"模型已保存到: {filepath}")


def load_model(filepath):
    """加载训练好的模型"""
    if not os.path.exists(filepath):
        return None

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']

    model = LDPCNeuralDecoder(
        num_iters=model_config['num_iters'],
        num_edges=model_config['num_edges'],
        connect_indices=model_config['connect_indices'],
        lift_shifts_1=model_config['lift_shifts_1'],
        lift_shifts_2=model_config['lift_shifts_2']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型已从 {filepath} 加载")
    return model


def train_model(force_retrain=False):
    """训练LDPC神经网络解码器"""
    model_dir = os.path.join(PROJECT_ROOT, "model")
    # 确保模型目录存在
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ldpc_neural_decoder.pth")

    # 检查是否已存在训练好的模型
    if not force_retrain and os.path.exists(model_path):
        print("发现已训练好的模型，跳过训练...")
        return load_model(model_path)

    print("开始训练LDPC神经网络解码器...")

    # 初始化模型和优化器
    model = LDPCNeuralDecoder(iters_max, sum_edge, connect_indices, lift_shifts_1, lift_shifts_2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.5)  # 学习率衰减

    # === 添加内存映射数据预生成 ===
    data_file_X = f'temp_data/X_train.npy'
    data_file_Y = f'temp_data/Y_train.npy'
    os.makedirs('temp_data', exist_ok=True)

    # 检查是否已存在预生成的数据文件
    if not os.path.exists(data_file_X) or not os.path.exists(data_file_Y):
        print("预生成训练数据...")
        X_all, Y_all = create_mix_epoch(SNR_sigma, wordRandom, noiseRandom,
                                        numOfWordSim_train * num_of_batch, code_n, code_k,
                                        Z, code_GM, train_on_zero_word, to_device=False)
        np.save(data_file_X, X_all)
        np.save(data_file_Y, Y_all)
        print(f"数据已保存: {data_file_X}, {data_file_Y}")
    else:
        print("使用已预生成的训练数据")

    # 加载内存映射
    X_memmap = np.load(data_file_X, mmap_mode='r')
    Y_memmap = np.load(data_file_Y, mmap_mode='r')

    # 跟踪最佳值
    best_accuracy = 0.0
    best_ber = float('inf')
    best_fer = float('inf')
    best_loss = float('inf')
    best_batch_idx = 0

    # 用于统计整个iteration的准确度
    total_accuracy = 0
    total_ber = 0
    total_fer = 0

    for batch_idx in range(num_of_batch):
        start = batch_idx * batch_size
        end = start + batch_size
        X_batch_np = X_memmap[start:end]  # 从内存映射读取
        Y_batch_np = Y_memmap[start:end]

        # 转换为PyTorch张量并移到设备
        X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
        Y_batch = torch.tensor(Y_batch_np, dtype=torch.float32, device=device)

        # 前向
        outputs = model(X_batch)

        # 计算损失
        loss = model.loss_fn(outputs, Y_batch)

        # 计算准确度指标
        accuracy = model.calculate_accuracy(outputs, Y_batch)
        ber = model.calculate_ber(outputs, Y_batch)
        fer = model.calculate_fer(outputs, Y_batch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if ber < best_ber:
            best_ber = ber
        if fer < best_fer:
            best_fer = fer
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_batch_idx = batch_idx
            # 保存模型
            save_model(model, model_path)

        # 累加统计
        total_accuracy += accuracy
        total_ber += ber
        total_fer += fer

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
        optimizer.step()

        # 打印日志
        if batch_idx % 1 == 0:  ### %200
            crt_lr = scheduler.get_last_lr()[0]
            print(f'batch: [{batch_idx}/{num_of_batch}]\t'
                  f'loss: {loss.item():.6f}\t'
                  f'current_lr: {crt_lr:.6f}\t'
                  f'acc: {accuracy:.4f}\t'
                  f'BER: {ber:.6f}\t'
                  f'FER: {fer:.4f}'
                  )

        scheduler.step()

    # 计算平均指标
    avg_accuracy = total_accuracy / num_of_batch
    avg_ber = total_ber / num_of_batch
    avg_fer = total_fer / num_of_batch

    print(f"\n训练完成 - 最佳准确度: {best_accuracy:.4f}, 最佳BER: {best_ber:.6f}, 最佳FER: {best_fer:.4f}")
    print(f"\n训练完成 - 平均准确度: {avg_accuracy:.4f}, 平均BER: {avg_ber:.6f}, 平均FER: {avg_fer:.4f}")
    print(f"最佳损失出现在 batch {best_batch_idx}")

    return model


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='LDPC神经网络解码器')
    parser.add_argument('--train', action='store_true', help='强制重新训练模型')
    parser.add_argument('--eval', action='store_true', help='仅加载模型进行评估')

    args = parser.parse_args()

    if args.train:
        print("=== 强制重新训练模式 ===")
        model = train_model(force_retrain=True)
    elif args.eval:
        print("=== 评估模式 ===")
        model = load_model()
        if model is None:
            print("错误：未找到训练好的模型，请先运行训练")
            return
        # 这里可以添加评估代码
        print("模型加载成功，准备进行评估...")
    else:
        print("=== 默认模式 ===")
        model = train_model(force_retrain=True)

    print("程序执行完成")


if __name__ == "__main__":
    main()