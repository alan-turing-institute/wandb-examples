import wandb
from config import dataset_name, model_name, save_dir, wandb_entity, wandb_project


def make_artifacts():
    run = wandb.init(project=wandb_project, entity=wandb_entity, job_type="artifacts")

    artifact = wandb.Artifact(dataset_name, type="dataset")
    artifact.add_reference(f"file://{save_dir}/{dataset_name}")
    run.log_artifact(artifact)

    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_reference(f"file://{save_dir}/{model_name}")
    run.log_artifact(artifact)


if __name__ == "__main__":
    make_artifacts()
