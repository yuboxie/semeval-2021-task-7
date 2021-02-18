import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from os.path import join, exists, split
from os import makedirs
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

batch_size = 1
model_name = 'microsoft/deberta-large'
restore_path = 'checkpoints/deberta_large_1a_5e-6'
restore_epoch = 3

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

def create_val_dataset(data_path, tokenizer):
    print('Creating the validation dataset...')
    df_val = pd.read_csv(join(data_path, 'dev.csv'))
    def encode(df):
        texts = df['text'].tolist()
        labels = df['is_humor'].values.astype(np.int32).tolist()
        encodings = tokenizer(texts, truncation = True, padding = True)
        dataset = HumorDataset(encodings, labels)
        return dataset
    val_dataset = encode(df_val)
    return val_dataset

def create_test_dataset(data_path, tokenizer):
    print('Creating the test dataset...')
    df_test = pd.read_csv(join(data_path, 'gold-test-27446.csv'))
    def encode(df):
        texts = df['text'].tolist()
        labels = df['is_humor'].values.astype(np.int32).tolist()
        encodings = tokenizer(texts, truncation = True, padding = True)
        dataset = HumorDataset(encodings, labels)
        return dataset
    test_dataset = encode(df_test)
    return test_dataset

if __name__ == '__main__':
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    val_dataset = create_val_dataset('../data', tokenizer)
    test_dataset = create_test_dataset('../data', tokenizer)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model = DebertaForSequenceClassification.from_pretrained(model_name)
    state_dict = torch.load('{}/epoch_{:02d}.bin'.format(restore_path, restore_epoch))
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    def predict(loader):
        y_pred = []
        for batch in tqdm(loader, total = len(val_dataset) // batch_size):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask)
            pred = outputs['logits'].argmax(1)
            y_pred += pred.cpu().numpy().tolist()            
        return y_pred

    y_pred_val = predict(val_loader)
    y_pred_test = predict(test_loader)

    df_val = pd.read_csv('../data/dev.csv')
    y_true_val = df_val['is_humor'].values.astype(np.int32).tolist()
    p, r, f, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average = 'binary')
    acc = accuracy_score(y_true_val, y_pred_val)
    print('Val Num {} P {:.4f} R {:.4f} F {:.4f} Acc {:.4f}'.format(len(y_pred_val), p, r, f, acc))

    df_test = pd.read_csv('../data/gold-test-27446.csv')
    y_true_test = df_test['is_humor'].values.astype(np.int32).tolist()
    p, r, f, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average = 'binary')
    acc = accuracy_score(y_true_test, y_pred_test)
    print('Test Num {} P {:.4f} R {:.4f} F {:.4f} Acc {:.4f}'.format(len(y_pred_test), p, r, f, acc))

    y_true = np.array(y_true_test)
    y_pred = np.array(y_pred_test)
    pp = y_pred[y_true == 1].sum()
    np = y_pred[y_true == 1].shape[0] - pp
    pn = y_pred[y_true == 0].sum()
    nn = y_pred[y_true == 0].shape[0] - pn
    print('pp = {}, np = {}, pn = {}, nn = {}'.format(pp, np, pn, nn))

    # if not exists('test_results'):
    #     makedirs('test_results')
    # df_test = pd.read_csv('../data/gold-test-27446.csv')
    # ids_test = df_test['id'].tolist()
    # pd.DataFrame({'id': ids_test, 'text': df_test['text'], 'gold': df_test['is_humor'], 'is_humor': y_pred_test}).to_csv(
    #     'test_results/1a_{:02d}_gold.csv'.format(restore_epoch), index = False)
