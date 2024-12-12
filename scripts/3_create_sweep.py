# Or use the wandb CLI:
# wandb sweep --entity entity_name --project project_name sweep_config.yaml
import wandb
from config import wandb_entity, wandb_project

sweep_config = {
    "name": "Example Training Sweep",
    "method": "random",
    "metric": {"goal": "minimize", "name": "eval/loss"},
    "parameters": {
        "per_device_train_batch_size": {
            "distribution": "int_uniform",
            "min": 4,
            "max": 32,
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        },
        "weight_decay": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.1,
        },
        "num_train_epochs": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 10,
        },
        "lr_scheduler_type": {
            "values": ["linear", "cosine", "constant"],
        },
        "warmup_ratio": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.25,
        },
        "optim": {
            "values": ["adamw_torch", "sgd", "adafactor", "lion_32bit"],
        },
    },
}

if __name__ == "__main__":
    wandb.sweep(sweep=sweep_config, entity=wandb_entity, project=wandb_project)
