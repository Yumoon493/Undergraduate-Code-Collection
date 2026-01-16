import os
import json
from tqdm import tqdm


def convert_file(input_path, output_path):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件 {input_path} 不存在！")

    # 创建输出目录（关键修复！）
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)  # 自动创建缺失目录

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc=f"处理 {os.path.basename(input_path)}"):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 数据转换逻辑
            # 1. 将原始文本按空格分割成词语列表
            words = line.split()

            # 2. 初始化字符和标签列表
            chars = []  # 存储字符
            labels = []  # 存储对应的标签

            # 3. 遍历每个词语，生成字符和标签
            for word in words:
                if len(word) == 1:
                    # 单字词：标记为 S
                    chars.append(word)
                    labels.append("S")
                else:
                    # 多字词：首字标记为 B，中间字标记为 M，末字标记为 E
                    chars.extend(list(word))  # 将词语拆分为字符
                    labels.append("B")  # 首字
                    labels.extend(["M"] * (len(word) - 2))  # 中间字
                    labels.append("E")  # 末字

            # 4. 将字符和标签转换为 JSON 格式并写入文件
            json.dump({
                "text": " ".join(chars),  # 字符序列，用空格分隔
                "label": " ".join(labels)  # 标签序列，用空格分隔
            }, f_out, ensure_ascii=False)  # ensure_ascii=False 支持中文字符
            f_out.write("\n")  # 每条数据占一行


# 示例：调用函数
input_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\raw\icwb2-data\icwb2-data\training\pku_training.utf8"
output_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\processed\train.json"

convert_file(input_path, output_path)


# 从训练集中划分验证集
def split_train_dev(train_path, dev_path, dev_ratio=0.05):
    with open(train_path, 'r', encoding='utf-8') as f_train:
        lines = f_train.readlines()

    # 计算验证集大小
    dev_size = int(len(lines) * dev_ratio)
    dev_data = lines[:dev_size]  # 前 dev_size 条作为验证集
    train_data = lines[dev_size:]  # 剩余部分作为训练集

    # 写入验证集
    with open(dev_path, 'w', encoding='utf-8') as f_dev:
        f_dev.writelines(dev_data)

    # 覆盖写入训练集
    with open(train_path, 'w', encoding='utf-8') as f_train:
        f_train.writelines(train_data)


# 调用函数划分验证集
train_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\processed\train.json"
dev_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\processed\dev.json"
split_train_dev(train_path, dev_path, dev_ratio=0.05)  # 5% 作为验证集

# 处理测试集
test_input_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\raw\icwb2-data\icwb2-data\testing\pku_test.utf8"
test_output_path = r"D:\HuaweiMoveData\Users\24901\Desktop\大学存档\自然语言处理\.venv\data\processed\test.json"
convert_file(test_input_path, test_output_path)