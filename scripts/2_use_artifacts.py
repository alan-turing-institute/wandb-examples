import wandb
from config import (
    full_data_artifact,
    save_dir,
    train_dataset_name,
    val_dataset_name,
    wandb_entity,
    wandb_project,
)
from datasets import Dataset, load_from_disk
from wandb.sdk.wandb_run import Run


def save_split(split: Dataset, name: str, run: Run):
    split.save_to_disk(f"{save_dir}/{name}")

    artifact = wandb.Artifact(name, type="dataset")
    artifact.add_reference(f"file://{save_dir}/{name}")
    run.log_artifact(artifact)


def make_data_splits():
    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="preprocess")

    artifact = run.use_artifact(full_data_artifact, type="dataset")
    artifact_dir = artifact.download()

    dataset = load_from_disk(artifact_dir)

    splits = dataset.train_test_split(test_size=0.2, seed=42)

    save_split(splits["train"], train_dataset_name, run)
    save_split(splits["test"], val_dataset_name, run)


if __name__ == "__main__":
    make_data_splits()
