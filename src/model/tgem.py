""" TGEM (Transformer-based Geister Piece Estimation Model)

Transformer-Encoderを用いて，自駒手と現在までの指し手から，相手駒が赤駒である確率を出力するモデル．

"""


import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TGEM(nn.Module):
    """ TGEM (Transformer-based Geister Piece Estimation Model) クラス

    Transformer-Encoderを用いて，自駒手と現在までの指し手から，相手駒が赤駒である確率を出力するモデル．

    Attributes:
        embedding (): 埋め込み層
        encoder_layer (): Transformer-Encoderレイヤ
        transformer_encoder (): Transformer-Encoder層
        fc (): 出力層

    """

    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int, num_layers: int, num_classes: int):
        """ コンストラクタ

        Args:
            vocab_size (int): 語彙数
            hidden_dim (int): 隠れ層の次元数
            num_heads (int): Attentionのヘッド数
            num_layers (int): Traansformerの層数
            num_classes (int): ラベル数，相手駒数8でするのが良さそう

        Examples:
            >>> vocab_size = 72
            >>> hidden_dim = 256
            >>> num_heads = 8
            >>> num_layers = 6
            >>> num_classes = 8
            >>> model = TGEM(vocab_size, hidden_dim, num_heads, num_layers, num_classes)

        """
        super(TGEM, self).__init__()

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Transformerエンコーダ層
        self.encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)

        # 出力層
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """ 順伝搬

        Args:
            x (): よくわかってない

        Returns:
            torch.sigmoid: 出力層の出力をSigmoid関数を通して返す

        """
        # 埋め込み層
        x = self.embedding(x)

        # Transformerエンコーダ
        x = self.transformer_encoder(x)

        # シーケンスの平均
        x = x.mean(dim=1)

        # 出力層
        logits = self.fc(x)

        # Sigmoid関数を利用
        return torch.sigmoid(logits)
