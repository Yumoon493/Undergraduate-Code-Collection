import paddlenlp
from datasets import load_dataset

# Load datasets using datasets
train_path = "data/processed/train.json"
dev_path = "data/processed/dev.json"
test_path = "data/processed/test.json"
dataset = load_dataset("json", data_files={"train":train_path, "dev":dev_path, "test":test_path})

#  = = = new code: random sampling training data = = = 10%
from datasets import Dataset
import random
random.seed(42)  # Fixed random seeds to ensure reproducibility

full_train_data = dataset["train"]
# Calculate 10% of the data volume
sample_size = int(0.1 * len(full_train_data))
# Random sampling
sampled_indices = random.sample(range(len(full_train_data)), sample_size)
small_train_data = full_train_data.select(sampled_indices)

# Replace the original training set
dataset["train"] = small_train_data

# Print the amount of sampled data
print(f"Training set size after sampling: {len(dataset['train'])} (Life size: {len(full_train_data)})")

# Print the first 3 pieces of data in the training set
print(dataset["test"][:3])


from paddlenlp.transformers import AutoTokenizer
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer.__class__)

# Use tokenizer to convert the data into the corresponding feature form
inputs = tokenizer(text='本项目的目的是实现中文分词')
print(inputs)

max_seq_len = 512  # Consistent with the maximum sequence length used in training

def convert_example_to_feature(example, tokenizer, label2id, max_length=512, truncation=True,is_infer=False):
    # Transform the input data into a characteristic form with the tokenizer
    text = example["text"].strip().split(" ")
    encoded_inputs = tokenizer(text, max_length=max_length, truncation=True,is_split_into_words="token", return_length=True)

    # Process labeled data
    if not is_infer:
        label = [label2id[item] for item in example["label"].split(" ")][:max_length - 2]
        encoded_inputs["label"] = [label2id["O"]] + label + [label2id["O"]]
        assert len(encoded_inputs["label"]) == len(encoded_inputs["input_ids"])

    return encoded_inputs

# Initialize tokenizer
model_name = "bert-base-chinese"
label2id = {"O": 0, "B": 1, "M": 2, "E": 3, "S": 4}
tokenizer = AutoTokenizer.from_pretrained(model_name)

from functools import partial

max_length = 512
trans_fn = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id, max_length=max_length)

# Transform the input data training set, verification set and test set into a unified feature form
columns = ["text", "label"]
train_dataset = dataset["train"].map(trans_fn, batched=False, remove_columns=columns)
dev_dataset = dataset["dev"].map(trans_fn, batched=False, remove_columns=columns)
test_dataset = dataset["test"].map(trans_fn, batched=False, remove_columns=columns)

# Output the number of samples in each dataset
print("train_dataset:", len(train_dataset))
print("dev_dataset:", len(dev_dataset))
print("test_dataset:", len(test_dataset))


from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import DataCollatorForTokenClassification

batch_size= 16
train_sampler = BatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
dev_sampler = BatchSampler(dev_dataset, batch_size=batch_size, shuffle=False)

# Use the preset DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=label2id["O"])
train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=data_collator)
dev_loader = DataLoader(dataset=dev_dataset, batch_sampler=dev_sampler, collate_fn=data_collator)
test_loader = DataLoader(dataset=test_dataset, batch_sampler=dev_sampler, collate_fn=data_collator)

# Print the first batch of data from the training set
print(next(iter(train_loader)))



# Model construction
import paddle
import paddle.nn as nn

class BertForTokenClassification(nn.Layer):
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
        # Input is passed into the BERT model for processing
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask)

        # Gets the vector sequence corresponding to the input sequence
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # Maps vectors to logits of lexemic markers via a linear layer
        logits = self.classifier(sequence_output)

        return logits


from paddlenlp.transformers import AutoModelForTokenClassification

# model_name: bert-base-chinese
model = AutoModelForTokenClassification.from_pretrained(model_name, num_classes=5)
print(model.__class__)


# Training configuration
from paddlenlp.metrics import ChunkEvaluator

num_epochs = 3              # Number of training rounds
learning_rate = 3e-5        # Learning rate
eval_steps = 100            # Set the number of steps at which the model is evaluated
log_steps = 10              # Set how many steps to print the log at
save_dir = "./checkpoints"  # Model save directory

weight_decay = 0.01                                 # Weight attenuation coefficient during training
warmup_proportion = 0.1                             # Warm start training ratio during training
# After modification (based on the actual amount of sampled data) :
num_training_steps = (len(dataset["train"]) // batch_size) * num_epochs

# Except for the parameters of bias and LayerNorm, the other parameters perform attenuation operations during training
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

# Initialize the optimizer
optimizer = paddle.optimizer.AdamW(
        learning_rate=learning_rate,
        parameters=model.parameters(),
        weight_decay=weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define how evaluation metrics are calculated
metric = ChunkEvaluator(label_list=label2id.keys())


# Training cycle
def evaluate(model, data_loader, metric):
    model.eval()
    metric.reset()
    precision, recall, f1_score = 0, 0, 0
    # Read the data in the dataloader
    for batch_data in data_loader:
        input_ids, token_type_ids, labels, seq_lens = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["label"], batch_data["seq_len"]
        # Model prediction
        logits = model(input_ids, token_type_ids)
        preditions = logits.argmax(axis=-1)
        # Evaluation
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(seq_lens, preditions, labels)
        metric.update(num_infer_chunks.numpy(), num_label_chunks.numpy(), num_correct_chunks.numpy())
    precision, recall, f1_score = metric.accumulate()

    return precision, recall, f1_score

import os

def train(model):
 # Turn on model training mode
    model.train()
    global_step = 0
    best_score = 0.
 # Record losses during training and scores evaluated by the model on the validation set
    train_loss_record = []
    train_score_record = []
 # Do num_epochs round training
    for epoch in range(num_epochs):
        for step, batch_data in enumerate(train_loader):
            inputs, token_type_ids, labels = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["label"]
            # Get model predictions
            logits = model(input_ids=inputs, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)  # mean by default
            train_loss_record.append((global_step, loss.item()))
            # Gradient backpropagation
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
                # If the current indicator is optimal, save the model
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


import matplotlib.pyplot as plt

#Plot the loss change
def plot_training_loss(train_loss_record, fig_name, fig_size=(8, 6), sample_step=10, loss_legend_loc="lower left"):
    plt.figure(figsize=fig_size)
    train_steps = [x[0] for x in train_loss_record][::sample_step]
    train_losses = [x[1] for x in train_loss_record][::sample_step]
    plt.plot(train_steps, train_losses, color='#e4007f', label="Train Loss")
    # Draw axes and legends
    plt.ylabel("Loss", fontsize='large')
    plt.xlabel("Step", fontsize='large')
    plt.legend(loc=loss_legend_loc, fontsize='x-large')

    plt.savefig(fig_name)
    plt.show()

# Plot the change of assessment scores
def plot_training_acc(train_score_record, fig_name, fig_size=(8, 6), sample_step=10, acc_legend_loc="lower left"):
    plt.figure(figsize=fig_size)
    train_steps=[x[0] for x in train_score_record]
    train_losses = [x[1] for x in train_score_record]
    plt.plot(train_steps, train_losses, color='#e4007f', label="Dev Score")
    # Draw axes and legends
    plt.ylabel("Score", fontsize='large')
    plt.xlabel("Step", fontsize='large')
    plt.legend(loc=acc_legend_loc, fontsize='x-large')
    plt.savefig(fig_name)
    plt.show()

fig_path = "./chapter6_loss1.pdf"
plot_training_loss(train_loss_record, fig_path, loss_legend_loc="upper right", sample_step=5)

fig_path = "./chapter6_acc1.pdf"
plot_training_acc(train_score_record, fig_path, sample_step=1, acc_legend_loc="lower right")



# Load the trained model for prediction, re-instantiate a model, and then load the trained model parameters into the new model
saved_state = paddle.load("./checkpoints/best.pdparams")
model = AutoModelForTokenClassification.from_pretrained(model_name, num_classes=5)
model.load_dict(saved_state)

# Evaluation model
precision, recall, F1 = evaluate(model, test_loader, metric)
print(f"[Evaluate]  precision: {precision: .5f}, recall: {recall: .5f}, dev score: {F1:.5f}")


# Model prediction
from seqeval.metrics.sequence_labeling import get_entities

def parsing_label_sequence(tokens, label_sequence):

    prev = 0
    words = []
    items = get_entities(label_sequence, suffix=False)
    for name, start, end in items:
        if prev != start:
            words.extend(tokens[prev:start])
        words.append("".join(tokens[start:end + 1]))
        prev = end + 1

    return words

# Model prediction function
def infer(model, text, tokenizer, id2label):

    model.eval()
    # Data processing
    encoded_inputs = tokenizer(text, max_seq_len=max_seq_len)

    # Construct the data input to the model
    input_ids = paddle.to_tensor(encoded_inputs["input_ids"], dtype="int64").unsqueeze(0)
    token_type_ids = paddle.to_tensor(encoded_inputs["token_type_ids"], dtype="int64").unsqueeze(0)

    # Calculate the emission score
    logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
    predictions = logits.argmax(axis=-1).tolist()[0]
    label_sequence = [id2label[label_id] for label_id in predictions[1:-1]]

    # Parse the tags with the highest scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()[1:-1])
    words = parsing_label_sequence(tokens, label_sequence)

    print("tokenize sequence:", " | ".join(words))


text = "她会成为一名优秀的刑辩律师"
id2label = {v: k for k, v in label2id.items()}
infer(model, text, tokenizer, id2label)
