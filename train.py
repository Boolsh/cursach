import os

import keras
from keras import ops

from models import create_model


class CustomNonPaddingTokenLoss(keras.losses.Loss):
    """
    Функция потерь без учета PAD-токенов.
    """

    def __init__(self):
        super().__init__(name="custom_ner_loss")

    def call(self, y_true, y_pred):

        loss_function = keras.losses.SparseCategoricalCrossentropy(
            reduction=None
        )

        loss = loss_function(
            y_true,
            y_pred
        )

        mask = ops.cast(
            y_true > 0,
            dtype="float32"
        )

        loss *= mask

        return ops.sum(loss) / ops.sum(mask)

class MaskedAccuracy(keras.metrics.Metric):

    def __init__(self, name="masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = ops.argmax(y_pred, axis=-1)

        mask = ops.cast(y_true > 0, "float32")

        matches = ops.cast(
            ops.equal(y_true, y_pred),
            "float32"
        )

        matches *= mask

        self.correct.assign_add(
            ops.sum(matches)
        )

        self.total.assign_add(
            ops.sum(mask)
        )

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.)
        self.total.assign(0.)


def train_model(
        data,
        epochs=5,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        learning_rate=0.001,
        save_model=True,
        model_name="model.keras"
):
    """
    Обучает модель и возвращает:
        model
        history
    """

    model = create_model(
        num_tags=data["num_tags"],
        vocab_size=data["vocab_size"],
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=CustomNonPaddingTokenLoss(),
        metrics=[MaskedAccuracy()]
    )

    print("\n==============================")
    print("Начало обучения")
    print("==============================")

    print(f"Epochs     : {epochs}")
    print(f"Embed dim  : {embed_dim}")
    print(f"Heads       : {num_heads}")
    print(f"FF dim      : {ff_dim}")

    history = model.fit(
        data["train"],
        validation_data=data["validation"],
        epochs=epochs,
        verbose=1
    )

    if save_model:

        os.makedirs(
            "results/models",
            exist_ok=True
        )

        model.save_weights(
            os.path.join(
                "results/models",
                model_name.replace(".keras", ".weights.h5")
            )
        )

    return model, history