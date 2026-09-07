from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score, recall_score
from mltrainer import Trainer, TrainerSettings, ReportTypes

import time

from functools import partial

from src import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def run_experiment(
    model_function,
    experiment_name,
    trainstreamer,
    teststreamer,
    loss_fn,
    class_weights=False,
    parallel=False,
    runs=3,
    epochs=5,
    learning_rate=0.001,
    weight_decay=0.0,
    filename="results.csv",
):

    all_confusion_matrices = []

    for run in range(1, runs + 1):
        print(f"\n{experiment_name} - run {run}")

        model = model_function()

        optimizer_function = partial(
            torch.optim.Adam,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        settings = TrainerSettings(
            epochs=epochs,
            metrics=[
                metrics.Accuracy(),
                metrics.F1Score(average="micro"),
                metrics.F1Score(average="macro"),
                metrics.Precision("micro"),
                metrics.Recall("macro"),
            ],
            logdir=f"logs/{experiment_name}/run_{run}",
            train_steps=len(trainstreamer) // 5,
            valid_steps=len(teststreamer) // 5,
            reporttypes=[ReportTypes.TENSORBOARD],
            earlystop_kwargs=None,
        )

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_function,
            traindataloader=trainstreamer.stream(),
            validdataloader=teststreamer.stream(),
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            Early_stopping=True,
            patience=5
        )

        start_time = time.perf_counter()
        
        trainer.loop()

        training_time = time.perf_counter() - start_time

        # Model testen
        model.eval()

        y_true = []
        y_pred = []

        testdata = teststreamer.stream()

        with torch.no_grad():
            for _ in range(len(teststreamer)):
                X, y = next(testdata)

                predictions = model(X).argmax(dim=1)

                y_true.extend(y.numpy())
                y_pred.extend(predictions.numpy())

        recalls = recall_score(
            y_true,
            y_pred,
            average=None,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            normalize="true",
            )

        all_confusion_matrices.append(cm)

        result = pd.DataFrame({
            "Experiment": [experiment_name],
            "Run": [run],
            "Class Weights": [class_weights],
            "Parallel": [parallel],
            "Accuracy": [
                accuracy_score(y_true, y_pred)
            ],
            "Recall_N": [recalls[0]],
            "Recall_S": [recalls[1]],
            "Recall_V": [recalls[2]],
            "Recall_F": [recalls[3]],
            "Recall_Q": [recalls[4]],
        })

        result_file = Path(filename)

        # Zorg dat de map bestaat
        result_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        if result_file.exists() and result_file.stat().st_size > 0:
            old_results = pd.read_csv(result_file)

            # Dezelfde experiment-run vervangen
            old_results = old_results[
                ~(
                    (old_results["Experiment"] == experiment_name)
                    & (old_results["Run"] == run)
                )
            ]

            result = pd.concat(
                [old_results, result],
                ignore_index=True,
            )

        result.to_csv(
            result_file,
            index=False,
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(y_true, y_pred):.3f}"
        )

        print(
            "Recall:",
            recalls.round(3),
        )

        print(
            f"Trainingstijd: {training_time:.1f} seconden"
        )


    plot_mean_confusion_matrix(
    confusion_matrices=all_confusion_matrices,
    experiment_name=experiment_name,
)

    


def plot_mean_confusion_matrix(
    confusion_matrices,
    experiment_name,
    output_dir="figures",
):
    """
    Berekent, toont en bewaart de gemiddelde confusion matrix
    van meerdere trainingsruns.
    """

    mean_cm = np.mean(
        confusion_matrices,
        axis=0,
    )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        mean_cm,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=["N", "S", "V", "F", "Q"],
        yticklabels=["N", "S", "V", "F", "Q"],
    )

    plt.xlabel("Voorspelde klasse")
    plt.ylabel("Werkelijke klasse")
    plt.title(
        f"Gemiddelde confusion matrix – {experiment_name}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path
        / f"{safe_name}_mean_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    return mean_cm