import mne
from PreProcessing import *

if __name__ == '__main__':
    fold_path = 'E:/BrainPrint/DEAPDataset/participant_ratings.xls'
    # fold_path = 'WCFC/Dataset/DEAPDataset/participant_ratings.xls'
    data_dir = 'E:/BrainPrint/DEAPDataset/DEAP_Raw'
    # data_dir = 'WCFC/Dataset/DEAPDataset/DEAP_Raw'
    save_dir = 'ProcessedData/DEAP'

    sub_info = [f's{i:02d}.bdf' for i in range(1, 33)]
    sub_batch = [1] * 22 + [2] * 10

    for idx, sub in enumerate(sub_info):

        print(f"\n\n正在处理被试{idx + 1}: {sub}")

        sub_path = os.path.join(data_dir, sub)
        rawdata = mne.io.read_raw_bdf(sub_path, preload=True)

        print(f"被试{sub}的所有通道名称: {rawdata.ch_names}")


        # 通道名称修复
        if 23 <= idx <= 27:
            rename_dict = {'': 'Status'}
            rawdata.rename_channels(rename_dict)
        elif 28 <= idx <= 32:
            rename_dict = {'-1': 'Status'}
            rawdata.rename_channels(rename_dict)

        # 事件检测（刺激标记）
        events = mne.find_events(rawdata, stim_channel='Status')
        print(f"找到 {len(events)}个事件")

        df = pd.read_excel(fold_path, engine='xlrd')
        start_row = idx * 40
        end_raw = start_row + 40
        vids = df.iloc[start_row:end_raw, 2].tolist()
        print(f"视频列表：{vids[:5]}...")  # 只显示前5个

        rawdata, unit = unit_check(rawdata)
        original_raw = rawdata.copy()
        print(f"数据单位：{unit}")

        # 数据分段
        if idx <= 22:
            cut_seconds = -30
            event_id = 5
        elif idx == 27:
            cut_seconds = -30
            # event_id = 1638149
            # event_id = [event_id, 5832452]

            # 先查看该被试实际包含的事件ID
            unique_events = np.unique(events[:, 2])
            print(f"s27.bdf的事件ID列表: {unique_events}")

            # 尝试从实际事件中自动匹配（优先找接近的大数值ID）
            candidate_ids = [uid for uid in unique_events if uid > 1000000]
            if len(candidate_ids) >= 2:
                event_id = candidate_ids  # 使用找到的大数值事件ID
            else:
                # 如果找不到足够的候选ID，使用最常见的事件ID
                event_id = np.argmax(np.bincount(events[:, 2]))
                print(f"自动选择事件ID: {event_id}")
        else:
            cut_seconds = -30
            event_id = 1638149

        # 创建epochs
        epochs = mne.Epochs(original_raw, events, event_id, tmin=cut_seconds, tmax=0, preload=True)
        if idx >= 22:
            epochs = epochs[2:]
        print(f"创建了{len(epochs)}个epochs")

        # 初始化保存的数据
        eeg_Data_saved = None

        process_count = 40
        if len(epochs) < process_count:
            print(f"警告：被试{sub}的Epochs数量不足（{len(epochs)}/40）")

            process_count = len(epochs)

        for index in range(process_count):

            if len(vids) > process_count:
                vids = vids[:process_count]  # 确保视频列表与处理的Epochs数量一致
            video = vids[index]

            # 获取当前试验
            epoch = epochs[index]
            # 预处理
            processed_epoch_ = PreProcessing(epoch)
            # 降采样
            processed_epoch_.down_sample(128)
            # 带通滤波
            processed_epoch_.band_pass_filter(0.5, 42)
            # 坏道插值
            processed_epoch_.bad_channels_interpolate(thresh1=3, proportion=0.3)
            # ICA
            processed_epoch_.eeg_ica()
            # 平均参考
            processed_epoch_.average_ref()

            # 保存数据
            eeg_Data_saved = data_concat(eeg_Data_saved, processed_epoch_.raw.get_data(), video)

        # 通道修改
        if int(sub_batch[idx] == 1):
            batch = 1
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)
        elif int(sub_batch[idx] == 2):
            batch = 2
            eeg_Data_saved = channel_modify(eeg_Data_saved, batch)

        # 保存为pkl文件
        eeg_save(sub, eeg_Data_saved, save_dir)
        print(f"已保存被试{sub}的数据")