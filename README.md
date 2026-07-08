# MLFramework Template

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Environment](https://img.shields.io/badge/environment-Nix%20%7C%20uv-success.svg)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-lightgrey.svg)

A professional, strictly typed, and reproducible boilerplate for advanced Machine Learning research and MLOps pipelines. 

This template is designed to provide a robust infrastructure for developing and evaluating complex neural network architectures. By abstracting away environment provisioning, hyperparameter management, and data versioning, it allows researchers to focus entirely on model logic and experimental iterations.

## Core Features

* **Deterministic Environments:** Powered by `Nix` and `uv` to ensure identical system-level dependencies and Python packages across all machines.
* **Decoupled Architecture:** Pure PyTorch models are strictly separated from PyTorch Lightning training loops.
* **Dynamic Configuration:** Hierarchical parameter management using `Hydra`, eliminating hardcoded values.
* **Experiment Tracking:** Out-of-the-box integration with `MLflow` for logging metrics and model checkpoints.
* **Data Versioning:** Pre-configured `DVC` setup for managing heavy datasets and binary artifacts outside of version control.
* **Code Quality:** Automated linting, formatting, and strict type checking enforced via `Makefile` (`Ruff` and `Mypy`).

## Documentation

Comprehensive documentation, including installation steps, architectural decisions, and usage guides, is maintained in the project Wiki.

👉 **[Go to the MLFramework Wiki](https://github.com/allllpina/MLTemplate/wiki)**

### Quick Links:
* [Quickstart Guide](https://github.com/allllpina/MLTemplate/wiki/Quickstart)
* [Configuration Management](https://github.com/allllpina/MLTemplate/wiki/Configuration-Management)
* [Data Versioning](https://github.com/allllpina/MLTemplate/wiki/Data-Versioning)

## Citation

If you utilize this framework template in your research or engineering workflows, please consider citing it to support ongoing development:

```bibtex
@software{MLFramework_Template_2026,
  author = {Your Name},
  title = {MLFramework Template: A Reproducible MLOps Environment},
  year = {2026},
  url = {[https://github.com/your-username/your-repository](https://github.com/your-username/your-repository)}
}
```
## License
This project is licensed under the PolyForm Noncommercial 1.0.0 License. See the LICENSE file for full details.