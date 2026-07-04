# import os
# import warnings
#
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#
# warnings.filterwarnings("ignore")
#
# import tensorflow as tf
#
# tf.get_logger().setLevel("ERROR")
#
# from experiments import run_experiment_series
#
# from dataset import prepare_data
#
# data = prepare_data()
#
# def main():
#
#     ###########################################################
#     # Исследование количества эпох
#     ###########################################################
#
#     run_experiment_series(
#         data=data,
#         parameter_name="epochs",
#         values=[1, 3, 5, 7, 10]
#     )
#
#     ###########################################################
#     # Исследование размера эмбеддингов
#     ###########################################################
#
#     run_experiment_series(
#         data=data,
#         parameter_name="embed_dim",
#         values=[16, 32, 64, 128]
#     )
#
#     ###########################################################
#     # Исследование количества голов внимания
#     ###########################################################
#
#     run_experiment_series(
#         data=data,
#         parameter_name="num_heads",
#         values=[1, 2, 4, 8]
#     )
#
#
# if __name__ == "__main__":
#     main()


import os
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from dataset import prepare_data
from train import train_model
from evaluate import evaluate_model


def main():

    print("=" * 60)
    print("Проверка проекта")
    print("=" * 60)

    data = prepare_data()

    model, history = train_model(
        data=data,
        epochs=3,
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        model_name="test_model.keras"
    )

    metrics = evaluate_model(
        model=model,
        history=history,
        test_dataset=data["test"],
        tag_mapping=data["tag_mapping"],
        experiment_name="test_run"
    )

    print("\nПолученные метрики")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()