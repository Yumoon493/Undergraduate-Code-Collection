# -*- coding: utf-8 -*-
# 最低内存要求：4GB RAM
# 安装命令（在CMD中执行）：
# pip install paddlepaddle==2.6.0 paddlenlp==2.6.0 numpy==1.26.0 psutil datasets

import os
import gc
import signal
import psutil
import paddle
import numpy as np
from functools import partial
from paddle.io import DataLoader
from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.transformers import BertTokenizer, BertForTokenClassification
from paddlenlp.metrics import ChunkEvaluator

# ==== 内存管理初始化 ====
paddle.set_device('cpu')  # 强制CPU模式
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 禁用tokenizer并行
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'  # 动态内存分配

# 内存监控配置
MAX_MEMORY_MB = 2500  # 设置最大允许内存（单位MB）
process = psutil.Process(os.getpid())


def memory_safe():
    """内存安全检查"""
    current_mem = process.memory_info().rss / 1024 ** 2
    if current_mem > MAX_MEMORY_MB * 0.8:
        print(f"⚠️ 内存告警: {current_mem:.1f}MB > {MAX_MEMORY_MB * 0.8:.1f}MB")
        return False
    return True


# ==== 数据流式加载 ====
def load_streaming_data():
    from datasets import load_dataset
    return load_dataset(
        "json",
        data_files={
            "train": "data/processed/train.json",
            "dev": "data/processed/dev.json",
            "test": "data/processed/test.json"
        },
        streaming=True  # 启用流式加载
    )


# ==== 数据处理管道 ====
class DataProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.label2id = {"O": 0, "B": 1, "M": 2, "E": 3, "S": 4}
        self.collator = DataCollatorForTokenClassification(
            self.tokenizer,
            label_pad_token_id=self.label2id["O"],
            pad_to_multiple_of=32  # 内存对齐优化
        )

    def process(self, example):
        """流式处理单个样本"""
        if not memory_safe():
            raise MemoryError("内存超出安全阈值")

        # 文本处理
        text = example["text"].strip().split(" ")
        label = example.get("label", "")

        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=64,  # 进一步缩短长度
            truncation=True,
            is_split_into_words=True,
            return_length=True
        )

        # 标签处理
        if label:
            label_ids = [self.label2id.get(tag, 0) for tag in label.split()]
            label_ids = label_ids[:64 - 2]  # 截断
            inputs["labels"] = [0] + label_ids + [0]
            assert len(inputs["labels"]) == len(inputs["input_ids"])

        return inputs


# ==== 模型定义 ====
def load_model():
    model = BertForTokenClassification.from_pretrained(
        'bert-base-chinese',
        num_classes=5,
        ignore_mismatched_sizes=True
    )
    model.eval()  # 初始设为评估模式节省内存
    return model


# ==== 训练流程 ====
class SafeTrainer:
    def __init__(self):
        self.batch_size = 2  # 更小的批次
        self.num_epochs = 1  # 初始设为1个epoch调试

    def create_streaming_loader(self, dataset):
        return DataLoader(
            dataset=dataset.map(self.processor.process, batched=False),
            batch_size=self.batch_size,
            collate_fn=self.processor.collator,
            num_workers=0  # 必须设为0
        )

    def train(self):
        try:
            # 初始化组件
            self.processor = DataProcessor()
            model = load_model()
            optimizer = paddle.optimizer.AdamW(
                learning_rate=3e-5,
                parameters=model.parameters()
            )

            # 流式数据加载
            dataset = load_streaming_data()
            train_loader = self.create_streaming_loader(dataset["train"])

            # 精简训练循环
            model.train()
            for epoch in range(self.num_epochs):
                print(f"==== Epoch {epoch + 1} =====")

                for step, batch in enumerate(train_loader):
                    if not memory_safe():
                        raise MemoryError("训练终止：内存超限")

                    # 前向计算
                    outputs = model(**batch)
                    loss = outputs.loss

                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    optimizer.clear_grad()

                    # 内存清理
                    del batch, outputs, loss
                    gc.collect()

                    if step % 10 == 0:
                        print(f"Step {step} | Mem: {process.memory_info().rss / 1024 ** 2:.1f}MB")

        except Exception as e:
            print(f"❌ 训练异常: {str(e)}")
            print("建议操作：")
            print("1. 关闭其他程序释放内存")
            print("2. 将batch_size设为1")
            print("3. 减少max_length到32")


# ==== 执行入口 ====
if __name__ == "__main__":
    # 注册中断信号处理
    signal.signal(signal.SIGINT, lambda *_: (print("\n🛑 用户中断"), exit()))

    # 启动训练
    print("=== 安全训练模式启动 ===")
    print(f"当前内存: {process.memory_info().rss / 1024 ** 2:.1f}MB")

    try:
        trainer = SafeTrainer()
        trainer.train()
    except KeyboardInterrupt:
        print("正常退出")
    finally:
        print("最终内存:", process.memory_info().rss / 1024 ** 2, "MB")