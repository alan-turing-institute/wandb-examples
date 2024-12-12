save_dir = "/tmp/wandb-examples"

hf_dataset_path = "sonos-nlu-benchmark/snips_built_in_intents"
dataset_name = hf_dataset_path.split("/")[-1]
train_dataset_name = f"{dataset_name}-train"
val_dataset_name = f"{dataset_name}-val"

hf_model_path = "microsoft/deberta-v3-xsmall"
model_name = hf_model_path.split("/")[-1]

wandb_entity = "turing-arc"
wandb_project = "christmas-wandb"

full_data_artifact = f"{wandb_entity}/{wandb_project}/{dataset_name}:latest"
train_data_artifact = f"{wandb_entity}/{wandb_project}/{train_dataset_name}:latest"
val_data_artifact = f"{wandb_entity}/{wandb_project}/{val_dataset_name}:latest"
model_artifact = f"{wandb_entity}/{wandb_project}/{model_name}:latest"
