from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
)

dataset_path = "sonos-nlu-benchmark/snips_built_in_intents"
model_path = "microsoft/deberta-v3-xsmall"


def train():
    dataset = load_dataset(dataset_path, split="train")
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=dataset.features["label"].num_classes,
        id2label={i: label for i, label in enumerate(dataset.features["label"].names)},
        label2id={label: i for i, label in enumerate(dataset.features["label"].names)},
    )

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = dataset.map(preprocess_function, batched=True)
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        "tmp", report_to="none", eval_strategy="epoch", use_mps_device=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()


if __name__ == "__main__":
    train()
