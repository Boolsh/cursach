import os

import matplotlib.pyplot as plt
import pandas as pd

from train import train_model
from evaluate import evaluate_model
from datetime import datetime

def run_experiment_series(
        data,
        parameter_name,
        values,
        epochs=5,
        embed_dim=32,
        num_heads=4,
        ff_dim=64
):
    """
    Запуск серии экспериментов
    по одному гиперпараметру.
    """

    print("=" * 60)
    print(f"Исследование параметра: {parameter_name}")
    print("=" * 60)


    results = []



    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Запуск эксперимента {experiment_name}")

    for value in values:

        current_epochs = epochs
        current_embed = embed_dim
        current_heads = num_heads

        if parameter_name == "epochs":
            current_epochs = value

        elif parameter_name == "embed_dim":
            current_embed = value

        elif parameter_name == "num_heads":
            current_heads = value

        experiment_name = f"{parameter_name}_{value}"

        print("\n")
        print("=" * 50)
        print(experiment_name)
        print("=" * 50)

        model, history = train_model(
            data=data,
            epochs=current_epochs,
            embed_dim=current_embed,
            num_heads=current_heads,
            ff_dim=ff_dim,
            model_name=f"{experiment_name}.keras"
        )

        metrics = evaluate_model(
            model=model,
            history=history,
            test_dataset=data["test"],
            tag_mapping=data["tag_mapping"],
            experiment_name=experiment_name
        )

        metrics[parameter_name] = value

        results.append(metrics)

    results = pd.DataFrame(results)

    folder = os.path.join(
        "results",
        parameter_name
    )

    os.makedirs(folder, exist_ok=True)

    results.to_csv(
        os.path.join(folder, "summary.csv"),
        index=False
    )

    ###################################################
    # Общий график F1
    ###################################################

    plt.figure(figsize=(8,5))

    plt.plot(
        results[parameter_name],
        results["f1"],
        marker="o"
    )

    plt.xlabel(parameter_name)

    plt.ylabel("F1-score")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(folder,"f1.png"),
        dpi=300
    )

    plt.close()

    ###################################################
    # Accuracy
    ###################################################

    plt.figure(figsize=(8,5))

    plt.plot(
        results[parameter_name],
        results["accuracy"],
        marker="o"
    )

    plt.xlabel(parameter_name)

    plt.ylabel("Accuracy")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(folder,"accuracy.png"),
        dpi=300
    )

    plt.close()

    ###################################################
    # Precision
    ###################################################

    plt.figure(figsize=(8,5))

    plt.plot(
        results[parameter_name],
        results["precision"],
        marker="o"
    )

    plt.xlabel(parameter_name)

    plt.ylabel("Precision")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(folder,"precision.png"),
        dpi=300
    )

    plt.close()

    ###################################################
    # Recall
    ###################################################

    plt.figure(figsize=(8,5))

    plt.plot(
        results[parameter_name],
        results["recall"],
        marker="o"
    )

    plt.xlabel(parameter_name)

    plt.ylabel("Recall")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        os.path.join(folder,"recall.png"),
        dpi=300
    )

    plt.close()

    print("\n")
    print(results)

    return results