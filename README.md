# WandB Examples

## Setup

### Python environment

```bash
git clone https://github.com/alan-turing-institute/wandb-examples.git
python -m venv .venv
source .venv/bin/activate
pip install .
```

### WandB Login

```bash
wandb login
```

If not logged in already this should prompt you for an API key, which you can get from https://wandb.ai/authorize

### Initial model and data download

```bash
python scripts/0_initial_download.py
```