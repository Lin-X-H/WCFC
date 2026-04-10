import mne
import numpy as np
import pandas as pd
import os
from mne.preprocessing import ICA
import pickle as pkl
import math
import picard


class PreProcessing():
    def __init__(self, raw):
        self.nchns = raw.info['nchan']  # 通道数量
        self.freq = raw.info['sfreq']  # 采样频率

        new_chn_names = raw.info['ch_names']

        # 分离EEG通道（前32个）
        eeg_raw = raw.copy().pick(range(32))
        eeg_raw.set_montage('standard_1020')  ### 这里删去一次set_montage()，下面设置一次电极位置即可？
        non_eeg_raw = raw.copy().pick(range(32, len(raw.ch_names)))
        # 设置标准10-20导联布局
        montage = mne.channels.make_standard_montage('standard_1020')
        eeg_raw.set_montage(montage, on_missing='warn')
        raw.set_montage(montage, on_missing='ignore')  # 主数据保留通道

        # 构建通道索引与名称的映射 montage_index
        self.montage_index = {
            'eeg': {idx: ch for idx, ch in enumerate(eeg_raw.ch_names)},
            'biosig': {idx + len(eeg_raw.ch_names): ch
                       for idx, ch in enumerate(non_eeg_raw.ch_names)},
        }

        self.montage_index = dict(zip(np.arange(self.nchns), new_chn_names))
        self.raw = raw
        self.data = self.raw.get_data()

    # 降采样
    def down_sample(self, n_freq):
        self.raw.resample(n_freq)

    # 带通滤波
    def band_pass_filter(self, l_freq, h_freq):
        self.raw.filter(l_freq, h_freq)

    #坏道插值
    def bad_channels_interpolate(self, thresh1=None, thresh2=None, proportion=0.3):
        data = self.raw.get_data()
        # 数据准备：如果是3D，压缩为2D便于处理
        if len(data.shape) > 2:
            data = np.squeeze(data)

        # 检测坏道
        Bad_chns = []
        value = 0
        if thresh1 is not None:
            md = np.median(np.abs(data))
            value = np.where(np.abs(data) > (thresh1 * md), 0, 1)[0]
        if thresh2 is not None:
            value = np.where(np.abs(data) > thresh2, 0, 1)[0]

        Bad_chns = np.argwhere(np.mean((1 - value), axis=0) > proportion).flatten()

        if Bad_chns.size > 0:
            # 执行插值
            self.raw.info['bads'].extend([self.montage_index[bad] for bad in Bad_chns])
            print('坏道：', self.raw.info['bad'])
            self.raw = self.raw.interpolate_bads()
        else:
            print('未检测到坏道')

    def eeg_ica(self, check_ica=None):
        ica = ICA(random_state=97, max_iter='auto', method='fastica')
        ica.fit(self.raw)

        # 检测眼电伪迹
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='EXG1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='EXG2')
        eog_indices3, eog_score3 = ica.find_bads_eog(self.raw, ch_name='EXG3')
        eog_indices4, eog_score4 = ica.find_bads_eog(self.raw, ch_name='EXG4')
        remove_indices = list(set(eog_indices1 + eog_indices2 + eog_indices3 + eog_indices4))
        ica.exclude = remove_indices
        # 应用第一轮ICA去除眼电伪迹
        ica.apply(self.raw)

        ica.exclude = []
        self.raw.pick(range(32))  # 只选择EEG通道
        print(f'ICA后通道数：{self.raw.info["nchan"]}')

        ### 是否需要重新创建ICA对象再进行第二轮分析?
        ica2 = ICA(random_state=97, max_iter='auto', method='fastica')
        ica2.fit(self.raw)
        # 检测肌电伪迹
        muscle_indices, scores = ica2.find_bads_muscle(self.raw, threshold=0.91)
        # 合并需要去除的部分
        remove_indices = list(set(muscle_indices))
        ica2.exclude = remove_indices
        # 应用ICA
        ica2.apply(self.raw)

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')


def data_concat(eegData, videoData: np.array, video: int):
    fs = 128
    secs = 30

    if len(videoData.shape) > 2:
        videoData = np.squeeze(videoData)

    trigger = np.zeros((1, fs * secs))
    trigger[0][0] = video  # 在数据开头添加视频ID作为触发标记

    # 确保数据长度正好是30秒
    if videoData.shape[1] > fs * secs:
        videoData = videoData[:, :-fs * secs]
    elif videoData.shape[1] < fs * secs:
        raise RuntimeError("试验长度错误")

    # 添加触发标记通道
    videoData = np.vstack((videoData, trigger))

    # 拼接数据
    if eegData is None:
        eegData = videoData
    else:
        eegData = np.hstack((eegData, videoData))
    return eegData

def unit_check(rawdata):
    original_raw = rawdata.copy()
    data_mean = np.mean(np.abs(original_raw._data[:32, :]))
    unit = 'μV'

    if math.log(data_mean) < 0:
        print(f'单位转换：{data_mean} -> {data_mean * 1000 * 1000}')
        original_raw._data = original_raw._data * 1000 * 1000
        unit = 'V'

    return original_raw, unit


def eeg_save(subject: str, eegData_trigger: np.array, filepath):
    filepath = os.path.join(filepath) + '/' + subject[:3] + '.pkl'
    parent_dir = os.path.dirname(filepath)
    os.makedirs(parent_dir, exist_ok=True)

    with open(filepath, 'wb') as f:
        pkl.dump(eegData_trigger, f)
        f.close()


def channel_modify(data, first_or_second):
    new_order = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 14, 15, 12, 29, 28, 30, 26, 27, 24, 25, 31, 22, 23, 20, 21,
                 18, 19, 17, 16, 32]

    if first_or_second == 1:
        data = data[np.array(new_order), :]

    chns = 32
    fs = 128
    n_vids = 40
    sec = 30

    eegdata = np.zeros((n_vids, chns, fs * sec))
    video_index = np.where(data[-1:, :].T > 0)[0]
    video = data[-1, video_index]
    video_arange = np.argsort(data[-1, video_index])
    video_arange_index = video_index[video_arange]

    for idx, vid in enumerate(video_arange_index):
        eegdata[idx, :, :] = data[:-1, vid:vid + fs * sec]

    return eegdata