import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取LDPC生成矩阵
BaseGraph = 2
Z = 5         # 提升因子
ldpc_i_ls = 2 # 基图索引

# 读取基图矩阵
base_graph_path = os.path.join(CURRENT_DIR, "BaseGraph", "BaseGraph{0}_Set{1}.txt".format(BaseGraph, ldpc_i_ls))
code_BG = np.loadtxt(base_graph_path, int, delimiter='\t')
code_N = code_BG.shape[1]  # 基图矩阵的列数 —— LDPC码的变量节点数
code_m = code_BG.shape[0]  # 基图矩阵的行数 —— LDPC码的校验结点数
code_k = code_N - code_m   # 信息位长度(k)

# 构造校验矩阵PCM —— 通过提升因子Z将基图扩展为更大的矩阵
PCM = np.zeros([code_m * Z, code_N * Z], dtype = int)
for i in range(code_m):
    for j in range(code_N):
        if code_BG[i, j] != -1:
            a = code_BG[i, j] % Z
            for k_shift in range(Z):
                PCM[i * Z + k_shift, j * Z + (k_shift + a) % Z] = 1

def roll_left(vec, L, Z):
    """向左循环移位"""
    vec1 = np.zeros([Z], dtype = int)
    vec1[0:Z - L] = vec[L:Z]
    if L != 0:
        vec1[Z - L:] = vec[0:L]
    return vec1

def roll_right(vec, L, Z):
    """向右循环移位"""
    vec1 = np.zeros([Z], dtype = int)
    vec1[0:L] = vec[Z - L:Z]
    vec1[L:Z] = vec[0:Z - L]
    if L != 0:
        vec1[0:L] = vec[Z-L:Z]
    return vec1

# LDPC编码器
def LDPC_encoder(infoWord, code_PCM, code_n, code_m, Z):
    code_k = code_n - code_m
    infoWord = np.reshape(infoWord, [code_k, Z])
    encodeWord = np.zeros((code_n, Z), dtype = int)
    shift = np.zeros((code_m, code_n, Z), dtype = int)
    encodeWord[:code_k, :] = infoWord[:, :]

    # 处理信息位
    for i in range(code_m):
        for j in range(code_k):
            if code_PCM[i, j] != -1:
                shift[i, j, :] = roll_left(infoWord[j, :], code_PCM[i, j] % Z, Z)

    check_vec = np.sum(shift, axis = 1) % 2
    encodeWord[code_k, :] = np.sum(check_vec[0:4, :], axis=0) % 2

    # 移位调整
    if BaseGraph == 1 and ldpc_i_ls == 2:
        encodeWord[code_k, :] = roll_right(encodeWord[code_k, :], 105, Z)
    if BaseGraph == 2 and ldpc_i_ls not in (3, 7):
        encodeWord[code_k, :] = roll_right(encodeWord[code_k, :], 1, Z)

    # 处理校验位
    for i in range(code_m):
        if code_PCM[i, code_k] != -1:
            shift[i, code_k, :] = roll_left(encodeWord[code_k, :], code_PCM[i, code_k] % Z, Z)
    check_vec = (check_vec + shift[:, code_k, :]) % 2

    for j in range(1, 4):
        encodeWord[code_k + j, :] = check_vec[j - 1, :]
        for i in range(code_m):
            if code_PCM[i, code_k + j] != -1:
                shift[i, code_k + j, :] = roll_left(encodeWord[code_k + j, :],
                                                    code_PCM[i, code_k + j] % Z, Z)
        check_vec = (check_vec + shift[:, code_k + j, :]) % 2

    encodeWord[code_k + 4:code_n, :] = check_vec[4:code_m, :]
    return np.reshape(encodeWord, [code_n * Z])

# 生成生成矩阵G
LDPC_G = np.zeros([code_k * Z, code_N * Z], dtype = int)
for i in range(code_k * Z):
    infoWord = np.zeros(code_k * Z, dtype = int)
    infoWord[i] = 1
    LDPC_G[i, :] = LDPC_encoder(infoWord, code_BG, code_N, code_m, Z)

# 保存生成矩阵
save_path = os.path.join(CURRENT_DIR, 'BaseGraph_GM', 'LDPC_GM_BG{0}_{1}.txt'.format(BaseGraph, Z))
np.savetxt(save_path, LDPC_G, fmt='%s', delimiter=',')