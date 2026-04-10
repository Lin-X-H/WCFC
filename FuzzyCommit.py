import torch
import numpy as np
import os
import hashlib
from EEG_Extract.train import Densenet
from EEG_Extract.train.Load_Data import loadData
from Neural_LDPC.GenerateMatrix import LDPC_encoder
from Neural_LDPC.Neural_MS import load_model
import torch.nn.functional as F
from feature_diagnostics import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件绝对路径
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # 项目根目录路径

class EEGFuzzyCommitmentSystem:
    """脑电信号模糊承诺系统"""

    def __init__(self, densenet_model_path, ldpc_model_path, ldpc_config):
        """
        初始化模糊承诺系统
        Args:
            densenet_model_path: 训练好的DenseNet模型路径
            ldpc_config: LDPC配置参数
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载脑电特征提取模型
        self.feature_extractor = self._load_densenet_model(densenet_model_path)
        self.feature_extractor.eval()

        # LDPC配置
        self.ldpc_config = ldpc_config

        # 初始化LDPC解码器（用于认证阶段）
        self.ldpc_decoder = self._initialize_ldpc_decoder(ldpc_model_path)

        # 初始化随机交织器
        np.random.seed(42)
        self.perm_indices = np.random.permutation(ldpc_config['encoded_length'])
        # 生成逆变换索引，用于恢复顺序
        self.inv_perm_indices = np.argsort(self.perm_indices)

    def _load_densenet_model(self, model_path):
        """加载预训练的DenseNet模型"""
        # 根据模型结构创建实例
        model = Densenet.densenet()

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)

        return model

    def _initialize_ldpc_decoder(self, model_path):
        """初始化LDPC神经网络解码器"""
        decoder = load_model(model_path)

        if decoder is None:
            raise FileNotFoundError(
                f"未找到训练好的LDPC解码器模型: {model_path}\n"
            )

        decoder.to(self.device)
        decoder.eval()
        return decoder

    def extract_continuous_features(self, eeg_signal):
        """从脑电信号提取连续特征"""
        with torch.no_grad():
            continuous_features = self.feature_extractor(
                eeg_signal.to(self.device), return_binary=True
            )
            return continuous_features.detach().cpu().numpy()[0]

    def feature_binarize(self, cont_feat):
        """二值化特征"""
        # 错位比较
        diff = cont_feat[:-1] - cont_feat[1:]
        bio_bin = (diff > 0).astype(np.int8)
        # 循环比较：最后一位和第一位比
        last_bit = 1 if cont_feat[-1] > cont_feat[0] else 0
        bio_bin = np.append(bio_bin, last_bit)
        return bio_bin

    def generate_random_key(self, key_length=None):
        """
        生成随机密钥
        Args:
            key_length: 密钥长度，默认为LDPC信息位长度
        Returns:
            key: 随机二进制密钥
        """
        if key_length is None:
            key_length = self.ldpc_config['info_length']

        # 生成随机二进制密钥
        key = np.random.randint(0, 2, size=key_length, dtype=int)
        return key

    def _ldpc_decode(self, encoded_data, confidence_val=2.0):
        """
        LDPC解码包装函数
        Args:
            encoded_data: 编码后的数据
            confidence_val: LLR幅值，越小表示对输入的信心越低（允许纠正更多错误）
        Returns:
            decoded_data: 解码后的数据
        """
        llr_input = np.where(encoded_data == 0, -confidence_val, confidence_val).astype(np.float32)
        llr_reshaped = llr_input.reshape(1, self.ldpc_config['code_n'], self.ldpc_config['Z'])  # 调整形状以匹配解码器输入 [1, N, Z]
        llr_tensor = torch.tensor(llr_reshaped, device=self.device, dtype=torch.float32)  # 转换为张量

        # LDPC解码
        with torch.no_grad():
            decoded_output = self.ldpc_decoder(llr_tensor)
            decoded_binary = (torch.sigmoid(decoded_output) > 0.5).cpu().numpy().astype(int)

        # 提取信息位
        info_bits = decoded_binary[0, :self.ldpc_config['info_length']]

        return info_bits

    def enroll(self, eeg_signal, user_id, key=None):
        """
        用户注册：创建模糊承诺模板
        Args:
            eeg_signal: 注册脑电信号
            user_id: 用户标识
            key: 可选，自定义密钥；如为None则自动生成
        Returns:
            template: 注册模板 (commitment, key_hash)
        """
        # ---准备注册数据---
        cont_reg = self.extract_continuous_features(eeg_signal)  # 提取连续特征
        bio_reg = self.feature_binarize(cont_reg)  # 生物特征模板
        key = self.generate_random_key()  # 生成随机密钥

        # ---LDPC编码---
        encoded_key = LDPC_encoder(
            infoWord=key,
            code_PCM=self.ldpc_config['code_PCM'],
            code_n=self.ldpc_config['code_n'],
            code_m=self.ldpc_config['code_m'],
            Z=self.ldpc_config['Z']
        )

        # 对生物特征进行交织 (打乱顺序)
        bio_reg_permuted = bio_reg[self.perm_indices]

        # ---创建模版---
        commitment = (encoded_key ^ bio_reg_permuted)  # 创建模糊承诺：承诺值 = 编码密钥 ⊕ 生物特征
        key_hash = hashlib.sha256(key.tobytes()).hexdigest()  # 计算密钥哈希：用于验证

        # ---存储模板---
        template = {
            'user_id': user_id,
            'commitment': commitment,
            'key_hash': key_hash,
            'key_length': len(key),
            'bio_register': bio_reg
        }

        print(f"用户 {user_id} 注册成功")
        print(f"密钥长度: {len(key)}, 承诺值长度: {len(commitment)}")

        return template

    def verify(self, eeg_signal, template):
        """
        用户认证：验证脑电信号
        Args:
            eeg_signal: 待认证脑电信号
            template: 注册模板
        Returns:
            result: 认证结果 (success, confidence, recovered_key)
        """
        # ---准备认证数据---
        cont_test = self.extract_continuous_features(eeg_signal)  # 连续特征
        bio_test = self.feature_binarize(cont_test)  # 生物特征

        # 同样的交织 (打乱顺序)
        bio_test_permuted = bio_test[self.perm_indices]

        # ---LDPC译码---
        recovered_codeword = template['commitment'] ^ bio_test_permuted  # 恢复编码码字
        recovered_key = self._ldpc_decode(recovered_codeword)  # LDPC译码

        # ---验证密钥---
        recovered_key_hash = hashlib.sha256(recovered_key.tobytes()).hexdigest()
        verification_success = (recovered_key_hash == template['key_hash'])

        # ---计算置信度---
        bio_reg = template['bio_register']
        hamming_distance = np.sum(bio_reg != bio_test)  # 计算汉明距
        confidence = 1.0 - hamming_distance / len(bio_reg) # 计算置信度

        # ---自适应LLR机制---
        # 尝试一系列的 LLR 值
        # 0.5: 认为噪声极大，译码器会非常积极地翻转比特 / 3.0: 认为噪声极小，译码器倾向于保持原样
        llr_candidates = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]

        final_success = False
        final_key = None

        for conf in llr_candidates:
            # 尝试解码
            candidate_key = self._ldpc_decode(recovered_codeword, confidence_val=conf)

            # 哈希校验
            candidate_hash = hashlib.sha256(candidate_key.tobytes()).hexdigest()

            if candidate_hash == template['key_hash']:
                print(f"在LLR={conf}时解码成功！")
                final_success = True
                final_key = candidate_key
                break

        # ---暂存结果---
        result = {
            'success': final_success,
            'confidence': confidence,
            'recovered_key': final_key,
            'hamming_distance': hamming_distance,
            'user_id': template['user_id']
        }

        return result

def fuzzy_commitment():
    """模糊承诺系统"""
    # ---文件路径---
    pcm_path = os.path.join(PROJECT_ROOT, "WCFC", "Neural_LDPC", "BaseGraph", "BaseGraph2_Set2.txt")
    densenet_path = os.path.join(PROJECT_ROOT, 'WCFC', 'model', 'Noabl(CAADN)ACC0.9922.pth')
    ldpc_decoder_path = os.path.join(PROJECT_ROOT, 'WCFC', 'model', 'ldpc_neural_decoder.pth')

    # ---LDPC配置---
    ldpc_config = {
        'Z': 5,
        'code_k': 10,
        'code_m': 42,
        'code_n': 52,
        'info_length': 50,  # code_k * Z = 10 * 4
        'encoded_length': 260,  # 编码后长度
        'code_PCM': np.loadtxt(pcm_path, int, delimiter='\t'),
    }

    # ---初始化系统---
    system = EEGFuzzyCommitmentSystem(
        densenet_model_path=densenet_path,
        ldpc_model_path=ldpc_decoder_path,
        ldpc_config=ldpc_config
    )

    # ---加载脑电数据---
    try:
        dataloaders = loadData(batch_size=1)
        test_loader = dataloaders['test']

        # 字典用于存储不同用户的样本
        user_samples = {}

        print("正在收集不同用户的样本...")
        for i, (eeg_data, label) in enumerate(test_loader):
            uid = label.item()
            if uid not in user_samples:
                user_samples[uid] = []

            # 每个用户只存一个样本用于测试
            if len(user_samples[uid]) < 2:
                if len(eeg_data.shape) == 3:
                    eeg_data = eeg_data.unsqueeze(0)
                user_samples[uid].append(eeg_data)

            # 停止条件：至少有一个用户收集到了 2 个样本 (用于做合法测试) / 至少收集到了 5 个不同的用户 (用于做非法测试)
            users_with_two_samples = sum(1 for samples in user_samples.values() if len(samples) == 2)
            if users_with_two_samples >= 1 and len(user_samples) >= 10:
                break

        target_uid = None
        for uid, samples in user_samples.items():
            if len(samples) == 2:
                target_uid = uid
                break

        if target_uid is None:
            raise ValueError("未能在测试集中找到拥有两个样本的同一用户，无法进行鲁棒性测试。")

        print(f"数据加载完成。选定测试用户 ID: {target_uid}")
        print(f"总共加载了 {len(user_samples)} 个不同用户的数据")

    except Exception as e:
        print(f"加载真实数据失败: {e}，使用模拟数据")
        samples = [
            (torch.randn(1, 32, 175, 24), "模拟用户1"),
            (torch.randn(1, 32, 175, 24), "模拟用户2")
        ]

    # # ---feature_diagnostics检查脑电特征数据有效性---
    # feats, labels = extract_features(
    #     system.feature_extractor,
    #     test_loader,
    #     system.device,
    #     max_samples=30
    # )
    #
    # stats = continuous_feature_stats(feats)
    # bins = binarize(feats, threshold="median")
    # ham = hamming_analysis(bins, labels)
    # ent = bit_entropy(bins)
    #
    # print("\n=====Feature Diagnostic Report=====")
    # print("Continuous stats:", stats)
    # print("Hamming:", ham)
    # print("Entropy:", ent)

    # ---用户注册---
    print("\n=====用户注册=====")
    register_data = user_samples[target_uid][0]  # 取第一个
    register_id = f"user_{target_uid}"
    template = system.enroll(register_data, user_id=register_id)

    print(f"承诺值: {template['commitment'][:10]}...")
    print(f"密钥哈希: {template['key_hash']}")

    # ---用户认证 (同一用户不同样本)---
    print("\n=====合法用户认证=====")
    verify_data = user_samples[target_uid][1]  # 取第2个
    legitimate_result = system.verify(verify_data, template)

    print(f"认证结果: {'成功' if legitimate_result['success'] else '失败'}")
    print(f"置信度: {legitimate_result['confidence']:.4f}")
    print(f"汉明距离: {legitimate_result['hamming_distance']}")

    # ---非法用户尝试---
    print("\n=====非法用户尝试=====")
    for uid, samples in user_samples.items():
        if uid == target_uid:
            continue

        illegal_data = samples[0]
        false_result = system.verify(illegal_data, template)

        print(f"用户 {uid} 攻击 -> 结果: {'成功' if false_result['success'] else '失败'} | "
              f"置信度: {false_result['confidence']:.4f} | "
              f"汉明距离: {false_result['hamming_distance']}")


if __name__ == "__main__":
    fuzzy_commitment()