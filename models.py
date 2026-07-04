import keras
from keras import layers
from keras import ops


class TransformerBlock(layers.Layer):
    """
    Один блок Transformer Encoder.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):

        attention = self.attention(inputs, inputs)

        attention = self.dropout1(
            attention,
            training=training
        )

        out1 = self.layernorm1(inputs + attention)

        ffn = self.ffn(out1)

        ffn = self.dropout2(
            ffn,
            training=training
        )

        return self.layernorm2(out1 + ffn)


class TokenAndPositionEmbedding(layers.Layer):
    """
    Слой токенных и позиционных эмбеддингов.
    """

    def __init__(self, max_length, vocab_size, embed_dim):
        super().__init__()

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )

    def call(self, inputs):

        sequence_length = ops.shape(inputs)[-1]

        positions = ops.arange(
            start=0,
            stop=sequence_length,
            step=1
        )

        token_embeddings = self.token_embedding(inputs)

        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings


class NERModel(keras.Model):

    def __init__(
            self,
            num_tags,
            vocab_size,
            max_length,
            embed_dim,
            num_heads,
            ff_dim,
            dropout_rate=0.1
    ):

        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            max_length,
            vocab_size,
            embed_dim
        )

        self.transformer = TransformerBlock(
            embed_dim,
            num_heads,
            ff_dim,
            dropout_rate
        )

        self.dropout1 = layers.Dropout(dropout_rate)

        self.hidden = layers.Dense(
            ff_dim,
            activation="relu"
        )

        self.dropout2 = layers.Dropout(dropout_rate)

        self.classifier = layers.Dense(
            num_tags,
            activation="softmax"
        )

    def call(self, inputs, training=False):

        x = self.embedding(inputs)

        x = self.transformer(
            x,
            training=training
        )

        x = self.dropout1(
            x,
            training=training
        )

        x = self.hidden(x)

        x = self.dropout2(
            x,
            training=training
        )

        return self.classifier(x)


def create_model(
        num_tags,
        vocab_size,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        max_length=128,
        dropout_rate=0.1
):
    """
    Создает модель NER.
    """

    return NERModel(
        num_tags=num_tags,
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate
    )