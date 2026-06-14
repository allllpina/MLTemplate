# Machine Learning Experiments Template
This template based on PyTorch Lightning and designed to simplify experiments with different Neural Network architectures and data processing pipelines

## Quick start

<!-- ### Clone project using `Copier`
```
``` -->

### Setup `dvc` and `MLflow`
```
# Initialize dvc
make dvc_init

# Create .env file
make dotenv_init
```
<u>Be sure to fill all configuration files before you start to work</u>

### File structure
```
.
├── .bash_scripts
│   ├── env_init.sh
│   └── .gitkeep
├── conf
│   ├── data
│   │   ├── cifar10_example.yaml
│   │   └── README.md
│   ├── example_config.yaml
│   ├── model
│   │   ├── convkan_example.yaml
│   │   └── README.md
│   └── README.md
├── data
│   ├── .gitkeep
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       ├── cifar-10-batches-py
│       │   ├── batches.meta
│       │   ├── data_batch_1
│       │   ├── data_batch_2
│       │   ├── data_batch_3
│       │   ├── data_batch_4
│       │   ├── data_batch_5
│       │   ├── readme.html
│       │   └── test_batch
│       ├── cifar-10-python.tar.gz
│       └── .gitkeep
├── .env
├── .env.example
├── flake.lock
├── flake.nix
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
├── scripts # Made for additional scripts e.g. for creating dvc.yaml data pipelines 
│   └── .gitkeep
├── src
│   ├── cli.py
│   ├── data
│   │   ├── cifar10_example.py
│   │   └── README.md
│   └── models
│       ├── architectures
│       │   ├── convkan_example.py
│       │   └── README.md
│       ├── classification_task_example.py
│       └── README.md
└── uv.lock
```
`About certain directories`
- [/conf/](./conf/README.md)
- [/conf/data/](./conf/data/README.md)
- [/conf/model/](./conf/model/README.md)
- [/src/data/](./src/data/README.md)
- [/src/models/](./src/models/README.md)
- [/src/models/architectures](./src/models/architectures/README.md)

### Run the first experiment
#### Train model
```
python -m src.cli train -c <path to config file>
```
#### Test model
```
python -m src.cli test <path to model weights*> -c <path to config file>
```
`* - path to model weights could be either local or web`
