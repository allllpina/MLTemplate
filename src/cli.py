import typer
from hydra import initialize, compose
from hydra.utils import instantiate
import pytorch_lightning as pl
from rich.console import Console
from rich.traceback import install
import torch

install(show_locals=True, word_wrap=True)

app = typer.Typer(help="Custom CLI for neural networks experiments")
console = Console()

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    config_name: str = typer.Option("config", "--config", "-c", help="Name of the main configuration file"),
    overrides: list[str] = typer.Argument(None, help="Parametres overrides (e.g. trainer.max_epochs=50)")
):
    """
    Starts training cycle of the model based on declared config
    """

    console.print(f"[bold green]Initialization of the pipeline with config:[/bold green] {config_name}")
    # 1. Configuration loading via Hydra
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    # 2. Dynamic object creation
    console.print("[bold blue]Compilation of DataModule...[/bold blue]")
    datamodule = instantiate(cfg.data)

    console.print("[bold blue]Architecture compilation...[/bold blue]")
    architecture = instantiate(cfg.model)

    console.print("[bold blue]Task compilation (LightningModule)...[/bold blue]")
    # Dynamic creation of a Task by passing it an initialized architecture
    task = instantiate(cfg.task, model=architecture)

    # 3. Trainer setup
    console.print("[bold blue]PyTorch Lightning Trainer initialization...[/bold blue]")
    
    trainer_kwargs = dict(cfg.trainer)
    # Dynamic initialization of logger, if it is present in the configuration
    if 'logger' in trainer_kwargs:
        trainer_kwargs['logger'] = instantiate(trainer_kwargs['logger'])

    # Tensor cores optimisation for GPU computation acceleration
    torch.set_float32_matmul_precision('medium') 
    trainer = pl.Trainer(**trainer_kwargs)

    # 4. Training cycle start
    console.print("[bold yellow]Starting the training...[/bold yellow]")
    trainer.fit(model=task, datamodule=datamodule)
    
    console.print("[bold green]Training completed successfully.[/bold green]")

if __name__ == "__main__":
    app()