import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification, AdamW
from os.path import join, exists, split
from os import makedirs
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

batch_size = 16
learning_rate = 5e-6
num_epochs = 10
model_name = 'microsoft/deberta-base'
log_path = 'log/deberta_base_1c_5e-6.txt'
save_path = 'checkpoints/deberta_base_1c_5e-6'

class HumorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_train_val_datasets(data_path, tokenizer):
    print('Creating the training and validation datasets...')
    df_train = pd.read_csv(join(data_path, 'train.csv'))
    df_val = pd.read_csv(join(data_path, 'dev.csv'))
    def encode(df):
        df = df[df['is_humor'] == 1]
        texts = df['text'].tolist()
        labels = df['humor_controversy'].values.astype(np.int32).tolist()
        encodings = tokenizer(texts, truncation = True, padding = True)
        dataset = HumorDataset(encodings, labels)
        return dataset
    train_dataset = encode(df_train)
    val_dataset = encode(df_val)
    return train_dataset, val_dataset

if __name__ == '__main__':
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = create_train_val_datasets('../data', tokenizer)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model = DebertaForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.device_count() > 1:
        print("Let's use {} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    model.to(device)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    optim = AdamW(model.parameters(), lr = learning_rate)

    if not exists(split(log_path)[0]):
        makedirs(split(log_path)[0])
    if not exists(save_path):
        makedirs(save_path)

    f_log = open(log_path, 'w', encoding = 'utf-8')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask, labels = labels)
            loss = outputs[0].mean()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            print('Epoch {:02d} Batch {:04d} Mean Loss {:.4f} Loss {:.4f}'.format(
                epoch + 1, i + 1, epoch_loss / (i + 1), loss.item()))
            f_log.write('Epoch {:02d} Batch {:04d} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                epoch + 1, i + 1, epoch_loss / (i + 1), loss.item()))

        model.eval()
        y_true = []
        y_pred = []
        val_loss = []
        for batch in tqdm(val_loader, total = len(val_dataset) // batch_size):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            y_true += labels.cpu().numpy().tolist()
            outputs = model(input_ids, attention_mask = attention_mask)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(outputs['logits'], labels)
            val_loss.append(loss.item())
            pred = outputs['logits'].argmax(1)
            y_pred += pred.cpu().numpy().tolist()
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = 'binary')
        acc = accuracy_score(y_true, y_pred)
        print('Val Loss {:.4f} P {:.4f} R {:.4f} F {:.4f} Acc {:.4f}'.format(
            np.mean(val_loss), p, r, f, acc))
        f_log.write('Val Loss {:.4f} P {:.4f} R {:.4f} F {:.4f} Acc {:.4f}\n'.format(
            np.mean(val_loss), p, r, f, acc))

        model.train()
        checkpoint_path = join(save_path, 'epoch_{:02d}.bin'.format(epoch + 1))
        torch.save(model.state_dict(), checkpoint_path)
        print('Model saved to {}\n'.format(checkpoint_path))
        f_log.write('Model saved to {}\n\n'.format(checkpoint_path))

    f_log.close()
