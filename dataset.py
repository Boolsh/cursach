from collections import Counter

import keras
import tensorflow as tf
from datasets import load_dataset


def make_tag_lookup():
    return {
        0: "[PAD]",
        1: "O",
        2: "B-PER",
        3: "I-PER",
        4: "B-ORG",
        5: "I-ORG",
        6: "B-LOC",
        7: "I-LOC",
        8: "B-MISC",
        9: "I-MISC",
    }


def convert_dataset(split):
    tokens = []
    tags = []

    for sample in split:
        tokens.append(sample["tokens"])
        tags.append(sample["ner_tags"])

    return {
        "tokens": tokens,
        "ner_tags": tags
    }


def build_vocabulary(train_data, vocab_size=20000):
    all_tokens = []

    for sentence in train_data["tokens"]:
        all_tokens.extend(sentence)

    counter = Counter(map(str.lower, all_tokens))

    vocabulary = [
        token
        for token, _
        in counter.most_common(vocab_size - 2)
    ]

    lookup = keras.layers.StringLookup(
        vocabulary=vocabulary,
        mask_token=None,
        oov_token="[UNK]"
    )

    return vocabulary, lookup


def build_dataset(data, lookup_layer, batch_size=32):
    """
    Создает tf.data.Dataset напрямую из словаря,
    без промежуточных txt-файлов.
    """

    tokens = [
        lookup_layer(
            tf.strings.lower(sentence)
        )
        for sentence in data["tokens"]
    ]

    tags = [
        tf.constant(tag_sequence, dtype=tf.int64) + 1
        for tag_sequence in data["ner_tags"]
    ]

    dataset = tf.data.Dataset.from_generator(
        lambda: zip(tokens, tags),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        )
    )

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None]),
        padding_values=(
            tf.constant(0, dtype=tf.int64),
            tf.constant(0, dtype=tf.int64),
        ),
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_data(batch_size=32):

    dataset = load_dataset("conll2003")

    train = convert_dataset(dataset["train"])
    validation = convert_dataset(dataset["validation"])
    test = convert_dataset(dataset["test"])

    vocabulary, lookup = build_vocabulary(train)

    train_dataset = build_dataset(
        train,
        lookup,
        batch_size
    )

    validation_dataset = build_dataset(
        validation,
        lookup,
        batch_size
    )

    test_dataset = build_dataset(
        test,
        lookup,
        batch_size
    )

    print(f"Train: {len(train['tokens'])}")
    print(f"Validation: {len(validation['tokens'])}")
    print(f"Test: {len(test['tokens'])}")

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"NER tags: {make_tag_lookup()}")

    return {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
        "vocabulary": vocabulary,
        "lookup": lookup,
        "num_tags": 10,
        "tag_mapping": make_tag_lookup(),
        "vocab_size": len(vocabulary) + 2,
        "train_raw": train,
        "validation_raw": validation,
        "test_raw": test,
    }