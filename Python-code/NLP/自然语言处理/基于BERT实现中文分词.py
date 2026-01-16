import paddlenlp
from datasets import load_dataset

# 利用datasets加载数据集
train_path = "data/processed/train.json"
dev_path = "data/processed/dev.json"
test_path = "data/processed/test.json"
dataset = load_dataset("json", data_files={"train":train_path, "dev":dev_path, "test":test_path})

# 打印训练集中的前3条数据
print(dataset["test"][:3])



from paddlenlp.transformers import BertTokenizer
# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 使用tokenizer将数据转换成对应特征形式
inputs = tokenizer(text='中文分词是一项重要的自然语言处理领域任务')
print(inputs)

inputs = tokenizer('中文分词是一项重要的自然语言处理领域任务', return_length=True, max_length=128, truncation=True,return_position_ids=True, return_offsets_mapping=True, return_attention_mask=True)
print(inputs)



# 对文本序列进行分词
tokens = tokenizer.tokenize('中文分词是一项重要的自然语言处理领域任务')
print(tokens)
# 将token转换为id
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
# 将id转换为token
tokens = tokenizer.convert_ids_to_tokens(input_ids)
print(tokens)
# 将token转换为字符串
text = tokenizer.convert_tokens_to_string(tokens)
print(text)


from paddlenlp.transformers import AutoTokenizer
# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer.__class__)

# 使用tokenizer将数据转换成对应特征形式
inputs = tokenizer(text='中文分词是一项重要的自然语言处理领域任务')
print(inputs)




from paddlenlp.transformers import AutoTokenizer


def convert_example_to_feature(example, tokenizer, label2id, max_length=128, truncation=True,is_infer=False):

    # 利用tokenizer将输入数据转换成特征形式
    text = example["text"].strip().split(" ")
    encoded_inputs = tokenizer(text, max_length=max_length, truncation=True,is_split_into_words="token", return_length=True)

    # 处理带有标签的数据
    if not is_infer:
        label = [label2id[item] for item in example["label"].split(" ")][:max_length - 2]
        encoded_inputs["label"] = [label2id["O"]] + label + [label2id["O"]]
        assert len(encoded_inputs["label"]) == len(encoded_inputs["input_ids"])

    return encoded_inputs

# 初始化tokenizer
model_name = "bert-base-chinese"
label2id = {"O": 0, "B": 1, "M": 2, "E": 3, "S": 4}
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 展示一个样例
example = {"text":"钱 其 琛 访 问 德 班", "label":"S B E B E B E"}
features = convert_example_to_feature(example, tokenizer, label2id, max_length=128, truncation=True,is_infer=False)
print(features)

from functools import partial

max_length = 128
trans_fn = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_length=max_length)

# 将输入数据训练集、验证集和测试集统一转换成特征形式
columns = ["text", "label"]
train_dataset = dataset["train"].map(trans_fn, batched=False, remove_columns=columns)
dev_dataset = dataset["dev"].map(trans_fn, batched=False, remove_columns=columns)
test_dataset = dataset["test"].map(trans_fn, batched=False, remove_columns=columns)

# 输出每个数据集中的样本数量
print("train_dataset:", len(train_dataset))
print("dev_dataset:", len(dev_dataset))
print("test_dataset:", len(test_dataset))

import paddle

def collate_fn(batch_data, pad_token_id=0, pad_token_type_id=0, pad_label_id=0):

    input_ids_list, token_type_ids_list, label_list = [], [], []
    max_len = 0
    for example in batch_data:
        input_ids, token_type_ids, label = example["input_ids"], example["token_type_ids"], example["label"]
        # 对各项数据进行文本填充
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        label_list.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(input_ids))
    # 对数据序列进行填充至最大长度
    for i in range(len(input_ids_list)):
        cur_len = len(input_ids_list[i])
        input_ids_list[i] = input_ids_list[i] + [pad_token_id] * (max_len - cur_len)
        token_type_ids_list[i] = token_type_ids_list[i] + [pad_token_type_id] * (max_len - cur_len)
        label_list[i] = label_list[i] + [pad_label_id] * (max_len - cur_len)

    return paddle.to_tensor(input_ids_list),  paddle.to_tensor(token_type_ids_list), paddle.to_tensor(label_list)

from paddle.io import BatchSampler, DataLoader

batch_size= 4
train_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
dev_sampler = BatchSampler(dev_dataset, batch_size=batch_size, shuffle=False)

train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
dev_loader = DataLoader(dataset=dev_dataset, batch_sampler=dev_sampler, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_sampler=dev_sampler, collate_fn=collate_fn)

print(next(iter(train_loader)))

#In 12
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import DataCollatorForTokenClassification

batch_size= 4
train_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
dev_sampler = BatchSampler(dev_dataset, batch_size=batch_size, shuffle=False)

# 使用预置的DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=label2id["O"])
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=data_collator)
dev_loader = DataLoader(dataset=dev_dataset, batch_sampler=dev_sampler, collate_fn=data_collator)
test_loader = DataLoader(dataset=test_dataset, batch_sampler=dev_sampler, collate_fn=data_collator)

# 打印训练集中的第1个batch数据
print(next(iter(train_loader)))

import paddle
import paddle.nn as nn

class BertForTokenClassification(nn.Layer):
    """
    BERT模型上层叠加线性层，用以对输入序列的token进行分类， 如NER任务

    输入:
        - bert: BERT模型的实例
        - num_classes: 分类的类别数，默认为2
        - dropout: 对于BERT输出向量的dropout概率，如果为None，则会使用BERT内部设置的hidden_dropout_prob
    """

    def __init__(self, bert, num_classes=2, dropout=None):
        super(BertForTokenClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert
        self.dropout = nn.Dropout(dropout if dropout is not None else self.bert.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"], num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        # 将输入传入BERT模型进行处理
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask)

        # 获取输入序列对应的向量序列
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # 通过线性层将向量映射为词位标记的logits
        logits = self.classifier(sequence_output)

        return logits


from paddlenlp.transformers import BertForTokenClassification

# model_name: bert-base-chinese
model = BertForTokenClassification.from_pretrained(model_name, num_classes=5)

from paddlenlp.transformers import AutoModelForTokenClassification

# model_name: bert-base-chinese
model = AutoModelForTokenClassification.from_pretrained(model_name, num_classes=5)
print(model.__class__)


#1.2.5训练配置
from paddlenlp.metrics import ChunkEvaluator

# 训练轮数
num_epochs = 3
# 学习率
learning_rate = 3e-5
# 设定每隔多少步进行一次模型评估
eval_steps = 100
# 设定每隔多少步进行打印一次日志
log_steps = 10
# 模型保存目录
save_dir = "./checkpoints"

# 训练过程中的权重衰减系数
weight_decay = 0.01
# 训练过程中的暖启动训练比例
warmup_proportion = 0.1
# 总共需要的训练步数
num_training_steps = len(train_loader) * num_epochs

# 除bias和LayerNorm的参数除外，其他参数在训练过程中执行衰减操作
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

# 初始化优化器
optimizer = paddle.optimizer.AdamW(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义评估指标计算方式
metric = ChunkEvaluator(label_list=label2id.keys())

def evaluate(model, data_loader, metric):
    """
    模型评估函数

    输入:
        - model: 待评估的模型实例
        - data_loader: 待评估的数据集
        - metric: 用以统计评估指标的类实例
    """
    model.eval()
    metric.reset()
    precision, recall, f1_score = 0, 0, 0
    # 读取dataloader里面的数据
    for batch_data in data_loader:
        input_ids, token_type_ids, labels, seq_lens = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["label"], batch_data["seq_len"]
        # 模型预测
        logits = model(input_ids, token_type_ids)
        preditions = logits.argmax(axis=-1)
        # 评估
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(seq_lens, preditions, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()

    return precision, recall, f1_score




import os


def train(model):
    """
    模型训练函数

    输入:
        - model: 待训练的模型实例
    """

    # 开启模型训练模式
    model.train()
    global_step = 0
    best_score = 0.
    # 记录训练过程中的损失和在验证集上模型评估的分数
    train_loss_record = []
    train_score_record = []
    # 进行num_epochs轮训练
    for epoch in range(num_epochs):
        for step, batch_data in enumerate(train_loader):
            inputs, token_type_ids, labels = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["label"]
            # 获取模型预测
            logits = model(input_ids=inputs, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)  # 默认求mean
            train_loss_record.append((global_step, loss.item()))
            # 梯度反向传播
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % log_steps == 0:
                print(
                    f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")

            if global_step != 0 and (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):
                precision, recall, F1 = evaluate(model, dev_loader, metric)
                train_score_record.append((global_step, F1))

                model.train()
                # 如果当前指标为最优指标，保存该模型
                if F1 > best_score:
                    print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {F1:.5f}")
                    best_score = F1
                    save_path = os.path.join(save_dir, "best.pdparams")
                    paddle.save(model.state_dict(), save_path)
                print(f"[Evaluate]  precision: {precision: .5f}, recall: {recall: .5f}, dev score: {F1:.5f}")

            global_step += 1

    save_path = os.path.join(save_dir, "final.pdparams")
    paddle.save(model.state_dict(), save_path)
    print("[Train] Training done!")

    return train_loss_record, train_score_record


train_loss_record, train_score_record = train(model)