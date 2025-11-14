# import os
# import random

# def split_data(txtA_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
#     # 读取txtA文件中的数据行
#     with open(txtA_path, 'r', encoding='utf-8') as txtA_file:
#         lines = txtA_file.readlines()

#     # 计算划分的索引
#     total_lines = len(lines)
#     train_end = int(total_lines * train_ratio)
#     val_end = int(total_lines * (train_ratio + val_ratio))

#     # 随机打乱数据
#     random.shuffle(lines)

#     # 将数据划分到不同的文件中
#     train_data = lines[:train_end]
#     val_data = lines[train_end:val_end]
#     test_data = lines[val_end:]

#     # 写入到相应的文件中
#     write_to_file(train_path, train_data)
#     write_to_file(val_path, val_data)
#     write_to_file(test_path, test_data)

#     print(f"数据已按照 {train_ratio}:{val_ratio}:{test_ratio} 的比例划分并保存到文件。")

# def write_to_file(file_path, data):
#     # 创建或打开文件
#     with open(file_path, 'w', encoding='utf-8') as file:
#         # 写入数据
#         file.writelines(data)

# # 替换为实际的文件路径
# txtA_path = '/mnt/sdd/yrh/Fei_Bing/code_and_txt_data_process_Xi_Ru/total.txt'
# train_path = '/mnt/sdd/yrh/Fei_Bing/code_and_txt_data_process_Xi_Ru/train.txt'
# val_path =   '/mnt/sdd/yrh/Fei_Bing/code_and_txt_data_process_Xi_Ru/val.txt'
# test_path =  '/mnt/sdd/yrh/Fei_Bing/code_and_txt_data_process_Xi_Ru/test.txt'

# # 调用函数进行划分
# split_data(txtA_path, train_path, val_path, test_path)

import os
import random
from sklearn.model_selection import KFold

def make_kfold_splits(total_txt_path, save_dir, n_splits=5, val_ratio=0.1, seed=1337):
    """
    根据 total.txt 生成 k 折交叉验证的 train/val/test 文件。
    每折：
        - test_fold{i}.txt：作为测试集
        - train_fold{i}.txt：从其余折中划分出 90% 训练、10% 验证
        - val_fold{i}.txt：从训练中划出的验证集
    """
    os.makedirs(save_dir, exist_ok=True)

    # 读取所有样本行
    with open(total_txt_path, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f.readlines()]

    total_samples = len(all_lines)
    print(f"总样本数: {total_samples}")

    # 初始化KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 生成每折划分
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_lines)):
        train_val_data = [all_lines[i] for i in train_val_idx]
        test_data = [all_lines[i] for i in test_idx]

        # 从 train_val 中再划出 val_ratio 比例作为验证集
        random.seed(seed + fold)
        random.shuffle(train_val_data)
        val_split = int(len(train_val_data) * val_ratio)
        val_data = train_val_data[:val_split]
        train_data = train_val_data[val_split:]

        # 保存三个文件
        write_to_file(os.path.join(save_dir, f"train_fold{fold}.txt"), train_data)
        write_to_file(os.path.join(save_dir, f"val_fold{fold}.txt"), val_data)
        write_to_file(os.path.join(save_dir, f"test_fold{fold}.txt"), test_data)

        print(f"Fold {fold}: 训练集 {len(train_data)} | 验证集 {len(val_data)} | 测试集 {len(test_data)}")

    print(f"已生成 {n_splits} 折划分文件，保存路径：{save_dir}")

def write_to_file(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data_list))

if __name__ == "__main__":
    total_txt = '/mnt/sdd/wjh/Fei_Bing/code_and_txt_data_process_Xi_Ru/total.txt'
    save_dir = './kfold_splits'  # 当前目录下新建
    make_kfold_splits(total_txt, save_dir, n_splits=5, val_ratio=0.1)
