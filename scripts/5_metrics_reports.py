from argparse import ArgumentParser

import numpy as np
import wandb
from config import (
    val_data_artifact,
    wandb_entity,
    wandb_project,
)
from datasets import load_from_disk
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
)
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)


def evaluate(model_artifact):
    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="evaluate")

    artifact = run.use_artifact(val_data_artifact, type="dataset")
    eval_dataset = load_from_disk(artifact.download())

    artifact = run.use_artifact(model_artifact, type="model")
    model_path = artifact.download()
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=eval_dataset.features["label"].num_classes,
        id2label={
            i: label for i, label in enumerate(eval_dataset.features["label"].names)
        },
        label2id={
            label: i for i, label in enumerate(eval_dataset.features["label"].names)
        },
        ignore_mismatched_sizes=True,
    )

    # Preprocess/tokenize data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, remove_columns="text"
    )
    eval_dataset = eval_dataset.rename_column("label", "labels")

    outputs = [model(**collator([sample])) for sample in tqdm(eval_dataset)]

    probas = [softmax(out.logits).detach().numpy() for out in outputs]
    pred_labels = [probs.argmax().item() for probs in probas]
    true_labels = eval_dataset["labels"]

    conf_matrix = confusion_matrix(true_labels, pred_labels)
    class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    run.log(
        {
            "Class Accuracy": wandb.Table(
                data=[
                    [lab, acc]
                    for lab, acc in zip(
                        eval_dataset.features["labels"].names, class_acc
                    )
                ],
                columns=["class", "accuracy"],
            )
        }
    )

    xs = []
    ys = []

    for class_idx in range(eval_dataset.features["labels"].num_classes):
        mask = np.array(true_labels) == class_idx

        precision, recall, thresholds = precision_recall_curve(
            mask, np.array(probas).squeeze()[:, class_idx]
        )
        xs.append(precision)
        ys.append(recall)

    run.log(
        {
            "Precision Recall": wandb.plot.line_series(
                xs=xs,
                ys=ys,
                keys=eval_dataset.features["labels"].names,
                xname="precision",
            )
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a model from a previous job.")
    parser.add_argument("model_artifact", help="Model artifact path to evaluate")
    args = parser.parse_args()

    evaluate(args.model_artifact)
