import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SmilesEncoder(layers.Layer):
    def __init__(self, vocab_size: int, hidden_dim: int, emb_dim: int = 64, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.emb_dim = int(emb_dim)
        self.dropout = float(dropout)

        self.embedding = layers.Embedding(self.vocab_size, self.emb_dim, mask_zero=True)
        rnn_dim = max(8, self.hidden_dim // 2)
        self.encoder = layers.Bidirectional(layers.GRU(rnn_dim, dropout=self.dropout, recurrent_dropout=0.0))
        self.proj = layers.Dense(self.hidden_dim, activation="relu")

    def call(self, token_ids):
        x = tf.cast(token_ids, tf.int32)
        x = self.embedding(x)
        x = self.encoder(x)
        return self.proj(x)
