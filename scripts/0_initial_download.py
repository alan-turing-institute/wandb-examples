import os

from config import dataset_name, hf_dataset_path, hf_model_path, model_name, save_dir
from datasets import load_dataset
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Tokenizer,
)


def download_data_and_models():
    dataset = load_dataset(hf_dataset_path, split="train")
    tokenizer = DebertaV2Tokenizer.from_pretrained(hf_model_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(hf_model_path)

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(f"{save_dir}/{model_name}")
    tokenizer.save_pretrained(f"{save_dir}/{model_name}")
    dataset.save_to_disk(f"{save_dir}/{dataset_name}")


if __name__ == "__main__":
    download_data_and_models()
