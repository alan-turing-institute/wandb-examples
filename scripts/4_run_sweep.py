from argparse import ArgumentParser

import wandb
from config import (
    model_name,
    save_dir,
    train_data_artifact,
    val_data_artifact,
    wandb_entity,
    wandb_project,
)
from datasets import load_from_disk
from transformers import (
    DataCollatorWithPadding,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
)


def train():
    # wandb globally holds the relevant parameter sweep config for this run in
    # wandb.config
    run = wandb.init(project=wandb_project, entity=wandb_entity)

    # Load the dataset and model
    # - Mark the run as using the relevant artifacts as inputs
    # - "Download" the artifacts (in this case actually just copy from file so we could
    #    just load them directly, but you might have them remote/want to put them on a
    #    faster disk etc.)
    # - Load them with the relevant datasets/transformers classes
    artifact = run.use_artifact(train_data_artifact, type="dataset")
    train_dataset = load_from_disk(artifact.download())

    artifact = run.use_artifact(val_data_artifact, type="dataset")
    eval_dataset = load_from_disk(artifact.download())

    # artifact = run.use_artifact(model_artifact, type="model")
    # model_path = artifact.download()
    model_path = f"{save_dir}/{model_name}"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=train_dataset.features["label"].num_classes,
        id2label={
            i: label for i, label in enumerate(train_dataset.features["label"].names)
        },
        label2id={
            label: i for i, label in enumerate(train_dataset.features["label"].names)
        },
        ignore_mismatched_sizes=True,
    )

    # Preprocess/tokenize data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set Training Arguments. Some are hard-coded here, and others are kwargs from the
    # sweep stored in wandb.config (in which we named parameters appropriately for
    # compatibility with TrainingArguments)
    training_args = TrainingArguments(
        "tmp",
        report_to="wandb",
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=0.005,
        use_mps_device=True,
        **wandb.config,
    )

    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save and log final fine-tuned model as an artifact
    ft_model_name = f"{run.name}-model"
    trainer.save_model(f"{save_dir}/{ft_model_name}")
    artifact = wandb.Artifact(ft_model_name, type="model")
    artifact.add_reference(f"file://{save_dir}/{ft_model_name}")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = ArgumentParser(description="Launch a training job from the sweep")
    parser.add_argument("sweep_id", help="ID of sweep to launch a job from")
    parser.add_argument(
        "--count", default=1, type=int, help="Number of training jobs to run"
    )
    args = parser.parse_args()

    wandb.agent(
        sweep_id=args.sweep_id,
        entity=wandb_entity,
        project=wandb_project,
        function=train,
        count=args.count,
    )
