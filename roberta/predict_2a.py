import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from os.path import join, exists, split
from os import makedirs
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

batch_size = 1
model_name = 'roberta-large'
restore_path = 'checkpoints/roberta_large_2a_5e-6'
optimal_epochs = {'roberta-base': 10, 'roberta-large': 10}
restore_epoch = optimal_epochs[model_name]

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
        labels = df['offense_rating'].tolist()
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
        labels = df['offense_rating'].tolist()
        encodings = tokenizer(texts, truncation = True, padding = True)
        dataset = HumorDataset(encodings, labels)
        return dataset
    test_dataset = encode(df_test)
    return test_dataset

if __name__ == '__main__':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    val_dataset = create_val_dataset('../data', tokenizer)
    test_dataset = create_test_dataset('../data', tokenizer)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels = 1)
    state_dict = torch.load('{}/epoch_{:02d}.bin'.format(restore_path, restore_epoch))
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    def predict(loader, total_len):
        y_pred = []
        for batch in tqdm(loader, total = total_len // batch_size):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs['logits'].view(-1)
            y_pred += torch.clamp(logits, min = 0.0, max = 5.0).cpu().detach().numpy().tolist()
        return y_pred

    y_pred_val = predict(val_loader, len(val_dataset))
    y_pred_test = predict(test_loader, len(test_dataset))

    df_val = pd.read_csv('../data/dev.csv')
    y_true_val = df_val['offense_rating'].tolist()
    rmse = mean_squared_error(y_true_val, y_pred_val, squared = False)
    print('Val Num {} RMSE {:.4f}'.format(len(y_pred_val), rmse))

    df_test = pd.read_csv('../data/gold-test-27446.csv')
    y_true_test = df_test['offense_rating'].tolist()
    rmse = mean_squared_error(y_true_test, y_pred_test, squared = False)
    print('Test Num {} RMSE {:.4f}'.format(len(y_pred_test), rmse))

    # if not exists('test_results'):
    #     makedirs('test_results')
    # df_test = pd.read_csv('../data/public_test.csv')
    # ids_test = df_test['id'].tolist()
    # pd.DataFrame({'id': ids_test, 'offense_rating': y_pred_test}).to_csv(
    #     'test_results/2a_{:02d}.csv'.format(restore_epoch), index = False)
