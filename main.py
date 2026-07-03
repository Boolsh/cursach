import os

# Отключаем варнинги oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Отключаем предупреждения о symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Отключаем все предупреждения TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Подавляем предупреждения
import warnings

warnings.filterwarnings('ignore')

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops
import numpy as np
import tensorflow as tf
from keras import layers
from collections import Counter
import requests
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report,
)

import matplotlib.pyplot as plt

# Скрываем вывод oneDNN
tf.get_logger().setLevel('ERROR')

EPOCHS = 5
EMBED_DIM = 32
NUM_HEADS = 4
FF_DIM = 64
BATCH_SIZE = 32

# def download_conll_manually():
#     """Скачивает CoNLL 2003 датасет вручную из рабочего источника"""
#
#     # Создаем директорию для данных
#     os.makedirs("data/conll2003", exist_ok=True)
#
#     # Используем зеркало с GitHub, где есть эти файлы
#     base_url = "https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/"
#
#     files = {
#         "train.txt": "eng.train",
#         "valid.txt": "eng.testa",
#         "test.txt": "eng.testb"
#     }
#
#     for local_name, remote_name in files.items():
#         url = base_url + remote_name
#         print(f"Скачиваю {remote_name}...")
#
#         try:
#             response = requests.get(url, timeout=30)
#             response.raise_for_status()
#
#             with open(f"data/conll2003/{local_name}", 'w', encoding='utf-8') as f:
#                 f.write(response.text)
#             print(f"  {remote_name} успешно скачан")
#
#         except requests.exceptions.RequestException as e:
#             print(f"  Ошибка при скачивании {remote_name}: {e}")
#             raise
#
#     print("Датасет успешно скачан!")
#     return "data/conll2003"
#
#
# # print("Скачивание датасета...")

conll_path = "data/conll2003"  # <- ИСПРАВЛЕНО: добавлены скобки


# Теперь нужно прочитать файлы в нужном формате
def parse_conll_file(filepath):
    """Парсит файл в формате CoNLL"""
    tokens = []
    ner_tags = []
    current_tokens = []
    current_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_tokens:
                    tokens.append(current_tokens)
                    ner_tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
            else:
                parts = line.split()
                if len(parts) >= 4:
                    current_tokens.append(parts[0])
                    # Конвертируем NER тег в число (по стандарту CoNLL)
                    tag = parts[3]
                    tag_map = {
                        'O': 0,
                        'B-PER': 1, 'I-PER': 2,
                        'B-ORG': 3, 'I-ORG': 4,
                        'B-LOC': 5, 'I-LOC': 6,
                        'B-MISC': 7, 'I-MISC': 8
                    }
                    current_tags.append(tag_map.get(tag, 0))

        # Добавляем последний пример
        if current_tokens:
            tokens.append(current_tokens)
            ner_tags.append(current_tags)

    return {'tokens': tokens, 'ner_tags': ner_tags}


# Загружаем данные
print("Парсинг файлов...")
train_data_parsed = parse_conll_file("data/conll2003/train.txt")
valid_data_parsed = parse_conll_file("data/conll2003/valid.txt")
test_data_parsed = parse_conll_file("data/conll2003/test.txt")


# Создаем структуру, похожую на datasets
class ConllDataset:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test


conll_data = ConllDataset(train_data_parsed, valid_data_parsed, test_data_parsed)


# 2. Определение слоев трансформера
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = ops.shape(inputs)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings


# 3. Основная модель NER
class NERModel(keras.Model):
    def __init__(
            self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x


# Вспомогательная функция для сохранения в файл
def export_to_file(export_file_path, data):
    with open(export_file_path, "w") as f:
        for tokens, ner_tags in zip(data["tokens"], data["ner_tags"]):
            if len(tokens) > 0:
                f.write(
                    str(len(tokens))
                    + "\t"
                    + "\t".join(tokens)
                    + "\t"
                    + "\t".join(map(str, ner_tags))
                    + "\n"
                )


# Создаем папку и сохраняем train/val
os.makedirs("data", exist_ok=True)
print("Сохранение данных в файлы...")
export_to_file("./data/conll_train.txt", conll_data.train)
export_to_file("./data/conll_val.txt", conll_data.validation)
export_to_file("./data/conll_test.txt", conll_data.test)
# отладка проверка датасета
print(f"Train: {len(conll_data.train['tokens'])} примеров")
print(f"Validation: {len(conll_data.validation['tokens'])} примеров")
print(f"Test: {len(conll_data.test['tokens'])} примеров")


# 5. Создание словаря меток NER (IOB формат)
def make_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels)), all_labels))


mapping = make_tag_lookup_table()
num_tags = len(mapping)
print("Словарь меток:", mapping)
print("Количество тегов:", num_tags)

# 6. Создание словаря токенов (Vocabulary)
all_tokens = []
for tokens in conll_data.train["tokens"]:
    all_tokens.extend(tokens)

all_tokens_array = np.array(list(map(str.lower, all_tokens)))
counter = Counter(all_tokens_array)

vocab_size = 20000
vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]
lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)

print("Размер словаря:", len(vocabulary))

# 7. Создание tf.data.Dataset
train_data = tf.data.TextLineDataset("./data/conll_train.txt")
val_data = tf.data.TextLineDataset("./data/conll_val.txt")
test_data = tf.data.TextLineDataset("./data/conll_test.txt")


def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1: length + 1]
    tags = record[length + 1:]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1  # Сдвигаем, чтобы 0 был для паддинга
    return tokens, tags


def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)



train_dataset = (
    train_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(BATCH_SIZE)
)
val_dataset = (
    val_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(BATCH_SIZE)
)

test_dataset = (
    test_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(BATCH_SIZE)
)

# 8. Инициализация модели и кастомная функция потерь
ner_model = NERModel(num_tags, len(vocabulary) + 2, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)


class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=None
        )
        loss = loss_fn(y_true, y_pred)
        mask = ops.cast((y_true > 0), dtype="float32")
        loss = loss * mask
        return ops.sum(loss) / ops.sum(mask)


# 9. Компиляция и обучение
ner_model.compile(
    optimizer="adam",
    loss=CustomNonPaddingTokenLoss(),
    metrics=["accuracy"]
)

print("Начинаем обучение...")
history = ner_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1
)


os.makedirs("results", exist_ok=True)
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")


plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig("results/loss.png", dpi=300)

plt.close()



print("\nОценка модели на тестовой выборке...")

y_true = []
y_pred = []

for batch_x, batch_y in test_dataset:

    predictions = ner_model.predict(batch_x, verbose=0)

    predictions = np.argmax(predictions, axis=-1)

    batch_y = batch_y.numpy()

    for true_seq, pred_seq in zip(batch_y, predictions):

        for true_tag, pred_tag in zip(true_seq, pred_seq):

            # пропускаем PAD
            if true_tag != 0:

                y_true.append(true_tag)
                y_pred.append(pred_tag)

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# 10. Пример инференса (предсказания)
def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)

sample_input = tokenize_and_convert_to_ids(
    "eu rejects german call to boycott british lamb"
)

# Вариант 1: Используем tf.reshape
sample_input = tf.reshape(sample_input, [1, -1])

# Вариант 2: Или используем numpy (если tf не работает)
# sample_input = np.reshape(sample_input, (1, -1))

print("Входные данные (токены):", sample_input)

output = ner_model.predict(sample_input, verbose=0)
prediction = np.argmax(output, axis=-1)[0]
predicted_tags = [mapping[i] for i in prediction]

print("Предсказанные теги:", predicted_tags)

# Для красивого вывода токен-тег
tokens = "eu rejects german call to boycott british lamb".split()
print("\nРезультат NER:")
for token, tag in zip(tokens, predicted_tags):
    print(f"{token:15} -> {tag}")



