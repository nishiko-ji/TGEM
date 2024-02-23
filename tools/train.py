""" TGEMの学習コード
    TGEMの学習
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import time
import pprint

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import GeisterDataset
from src.data.tokenizer import GeisterTokenizer
from src.model.tgem import TGEM


def get_data(path):
    texts = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            data = line.rstrip('\n').split(' ')
            red_pos0 = data[0]
            texts.append(f'{red_pos0},{data[2]}')
            red_pos1 = data[1]
            labels.append(red_pos1)

    return texts, labels


def train(d_model, nhead, num_layers, batch_size, learning_rate):
    vocab = GeisterTokenizer()
    vocab_size = vocab.get_vocab_size()
    num_classes = 8 # 8ラベル分類
    num_epochs = 2    # epoch数
    # num_epochs = 20    # epoch数
    max_seq_length = 206    # 最大入力長
    data_path = './data/Naotti_Naotti_train.txt'
    data_paths = [
        './data/Naotti_Naotti_train.txt',
        './data/Naotti_hayazashi_train.txt',
        './data/hayazashi_Naotti_train.txt',
    ]
    checkpoint_dir = './checkpoints/'

    texts, labels = get_data(data_path)

    # データローダーのセットアップ
    train_texts = texts[:2500]
    train_labels = labels[:2500]
    # pprint.pprint(f'texts: {train_texts[0:10]}')
    # pprint.pprint(f'labels: {train_labels[0:10]}')
    train_dataset = GeisterDataset(train_texts, train_labels, vocab_size, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # for i in range(10):
    #     print(train_dataset[i])
    # モデルの設定
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('cuDNNの有効状態：', torch.backends.cudnn.enabled)


    model = TGEM(vocab_size, d_model, nhead, num_layers, num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_interval = 1

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch_input, batch_labels in train_loader:
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        average_train_loss = train_loss / num_batches
        print(f'Epoch {epoch +1}, Average Training Loss: {average_train_loss}')

        if (epoch + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(checkpoint_dir, f'ckpt_dmodel_{d_model}_nhead_{nhead}_numlayers_{num_layers}_batch_{batch_size}_rate_{str(learning_rate)[2:]}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_filename)
            print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_filename}')
        end_time = time.time()

        with open(checkpoint_dir+'trainloss.csv', 'a') as f:
            print(f'{d_model},{nhead},{num_layers},{batch_size},{learning_rate},{epoch+1},{average_train_loss},{end_time-start_time}', file=f)


def main():
    d_model = 256
    nhead = 8
    num_layers = 6
    batch_size = 16
    learning_rate = 0.0001
    train(d_model, nhead, num_layers, batch_size, learning_rate)


if __name__ == '__main__':
    main()
