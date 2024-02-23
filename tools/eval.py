import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, multilabel_confusion_matrix

import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.tokenizer import GeisterTokenizer
from src.data.dataset import GeisterDataset
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


def eval(d_model, nhead, num_layers, batch_size, learning_rate, epoch, data_path):
    print(d_model, nhead, num_layers, batch_size, learning_rate, epoch)
    vocab = GeisterTokenizer()
    vocab_size = vocab.get_vocab_size()
    num_classes = 8 # 8ラベル分類
    #max_seq_length = 206    # 最大入力長
    max_seq_length = 220    # new
    checkpoint_dir = './checkpoints/'
    eval_dir = './eval/Naotti_Naotti/'

    texts, labels = get_data(data_path)
    eval_texts = texts[2500:]
    eval_labels = labels[2500:]

    # データローダーのセットアップ
    eval_dataset = GeisterDataset(eval_texts, eval_labels, vocab_size, max_seq_length)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # モデルの設定
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('cuDNNの有効状態：', torch.backends.cudnn.enabled)

    model = TGEM(vocab_size, d_model, nhead, num_layers, num_classes).to(device)

    checkpoint_path = checkpoint_dir + f'ckpt_dmodel_{d_model}_nhead_{nhead}_numlayers_{num_layers}_batch_{batch_size}_rate_{str(learning_rate)[2:]}_epoch_{epoch}.pt'
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    # 予測と正解のリスト
    all_predictions = []
    all_ground_truth = []
    all_outputs = []

    with torch.no_grad():
        for batch_input, batch_labels in eval_loader:
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            # モデルの順伝播（forward）で予測を取得
            outputs = model(batch_input)
            all_outputs.extend(outputs)
            predictions = (outputs > 0.5).long()  # しきい値0.5を使用してバイナリ予測に変換
            all_predictions.extend(predictions.cpu().numpy())
            all_ground_truth.extend(batch_labels.cpu().numpy())

    for prediction, ground_truth in zip(all_outputs, all_ground_truth):
        p = prediction.to('cpu').detach().numpy().copy()
        pre = np.zeros(8)
        for red in p.argsort()[:3:-1]:
            pre[red] = 1

    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # 多ラベル混同行列
    conf_matrix = multilabel_confusion_matrix(np.array(all_ground_truth, dtype=np.int64), np.array(all_predictions))

    # 適合率、再現率、F1値、サポート数
    report = classification_report(np.array(all_ground_truth, dtype=np.int64), np.array(all_predictions), target_names=class_names)

    # ハミング損失
    hamming_loss_value = hamming_loss(np.array(all_ground_truth, dtype=np.int64), np.array(all_predictions))

    # ジャッカード類似度
    jaccard_similarity = jaccard_score(np.array(all_ground_truth, dtype=np.int64), np.array(all_predictions), average='samples')

    print("多ラベル混同行列:\n", conf_matrix)
    print("分類レポート:\n", report)
    print("ハミング損失:", hamming_loss_value)
    print("ジャッカード類似度:", jaccard_similarity)

    # 多ラベル混同行列
    conf_matrix_dfs = [pd.DataFrame(matrix.reshape(1, -1), columns=['Predicted 0', 'Predicted 1', 'Actual 0', 'Actual 1']) for matrix in conf_matrix]
    # 各DataFrameを連結
    conf_matrix_df = pd.concat(conf_matrix_dfs, keys=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
    conf_matrix_df.to_csv(eval_dir + f'confusion_matrix_{d_model}_{nhead}_{num_layers}_{batch_size}_{str(learning_rate)[2:]}_{epoch}.csv', index=False)

    # 分類レポート
    report_df = pd.DataFrame.from_dict(classification_report(np.array(all_ground_truth, dtype=np.int64), np.array(all_predictions), target_names=class_names, output_dict=True))
    report_df.to_csv(eval_dir + f'classification_report_{d_model}_{nhead}_{num_layers}_{batch_size}_{str(learning_rate)[2:]}_{epoch}.csv')

    # ハミング損失とジャッカード類似度
    metrics_df = pd.DataFrame({'Hamming Loss': [hamming_loss_value], 'Jaccard Similarity': [jaccard_similarity]})
    metrics_df.to_csv(eval_dir + f'metrics_{d_model}_{nhead}_{num_layers}_{batch_size}_{str(learning_rate)[2:]}_{epoch}.csv', index=False)


def main():
    data_path = './data/Naotti_hayazashi_eval.txt'
    eval(256, 8, 6, 16, 0.0001, 1, data_path)


if __name__ == '__main__':
    main()
