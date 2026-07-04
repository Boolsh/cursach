import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(
        model,
        history,
        test_dataset,
        tag_mapping,
        experiment_name="experiment"
):

    result_folder = os.path.join("results", experiment_name)
    os.makedirs(result_folder, exist_ok=True)

    ##########################################################
    # Предсказания
    ##########################################################

    y_true = []
    y_pred = []

    for batch_x, batch_y in test_dataset:

        predictions = model.predict(batch_x, verbose=0)
        predictions = np.argmax(predictions, axis=-1)

        batch_y = batch_y.numpy()

        for true_seq, pred_seq in zip(batch_y, predictions):

            for true_tag, pred_tag in zip(true_seq, pred_seq):

                # PAD пропускаем
                if true_tag == 0:
                    continue

                y_true.append(int(true_tag))
                y_pred.append(int(pred_tag))

    ##########################################################
    # Отладочная информация
    ##########################################################

    print("\n========== TEST RESULTS ==========")

    print(f"Количество токенов: {len(y_true)}")
    print(f"Истинные классы     : {sorted(set(y_true))}")
    print(f"Предсказанные классы: {sorted(set(y_pred))}")

    from collections import Counter

    print("\nРаспределение истинных классов:")

    counter = Counter(y_true)

    for key in sorted(counter.keys()):
        print(key, tag_mapping[key], counter[key])

    ##########################################################
    # Метрики
    ##########################################################

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

    ##########################################################
    # Classification report
    ##########################################################

    labels = list(range(1, len(tag_mapping)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[tag_mapping[i] for i in labels],
        zero_division=0
    )

    print("\n")
    print(report)

    with open(
            os.path.join(result_folder, "classification_report.txt"),
            "w",
            encoding="utf8"
    ) as f:
        f.write(report)

    ##########################################################
    # Метрики в txt
    ##########################################################

    with open(
            os.path.join(result_folder, "metrics.txt"),
            "w",
            encoding="utf8"
    ) as f:

        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")

    ##########################################################
    # Метрики в csv
    ##########################################################

    pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }]).to_csv(
        os.path.join(result_folder, "metrics.csv"),
        index=False
    )

    ##########################################################
    # Loss
    ##########################################################

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(result_folder, "loss.png"),
        dpi=300
    )

    plt.close()

    ##########################################################
    # Accuracy (обычная или Masked)
    ##########################################################

    train_metric = None
    val_metric = None

    if "masked_accuracy" in history.history:

        train_metric = "masked_accuracy"
        val_metric = "val_masked_accuracy"

    elif "accuracy" in history.history:

        train_metric = "accuracy"
        val_metric = "val_accuracy"

    if train_metric is not None:

        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history[train_metric],
            label="Train"
        )

        plt.plot(
            history.history[val_metric],
            label="Validation"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")

        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(result_folder, "accuracy.png"),
            dpi=300
        )

        plt.close()

    ##########################################################
    # Confusion Matrix
    ##########################################################

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[tag_mapping[i] for i in labels]
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(result_folder, "confusion_matrix.png"),
        dpi=300
    )

    plt.close()

    ##########################################################
    # Возвращаем результаты
    ##########################################################

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }