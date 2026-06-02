import os
import mlflow
from mlflow.tracking import MlflowClient
import typer
from hydra import initialize, compose
from hydra.utils import instantiate, get_class
import pytorch_lightning as pl
from rich.console import Console
from rich.traceback import install
import torch

install(show_locals=True, word_wrap=True)

app = typer.Typer(help="Custom CLI for neural networks experiments")
console = Console()

@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def train(
    ctx: typer.Context,
    config_name: str = typer.Option("config", "--config", "-c", help="Name of the main configuration file"),
):
    """
    Starts training cycle of the model based on declared config
    """

    console.print(f"[bold green]Initialization of the pipeline with config:[/bold green] {config_name}")

    overrides = [arg for arg in ctx.args if "=" in arg]
    
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

def resolve_checkpoint_path(checkpoint_ref: str, tracking_uri: str) -> str:
    """
    Looks for checkpoing:
    1. If it is real local path -> returns it.
    2. If it is path in MLflow format (runs:/...) -> downloads from MLflow.
    3. If nothing found -> raises an exception
    """
    # Scenario 1: Local file on PC
    if os.path.exists(checkpoint_ref):
        console.print(f"[bold green]Local checkpoing has been found:[/bold green] {checkpoint_ref}")
        return checkpoint_ref

    # Scenario 2: Attempt to download from MLflow
    mlflow_prefixes = ("runs:/", "models:/", "mlflow-artifacts:/")
    
    if checkpoint_ref.startswith(mlflow_prefixes):
        console.print(f"[bold yellow]Attempt to download an artefact form MLflow/DagsHub:[/bold yellow] {checkpoint_ref}")
        
        if not tracking_uri:
            console.print("[bold red]Error: MLFLOW_TRACKING_URI does not exist.[/bold red]")
            raise FileNotFoundError("Check MLFLOW_TRACKING_URI configuration.")
            
        mlflow.set_tracking_uri(tracking_uri)
        
        try:
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=checkpoint_ref)
            console.print(f"[bold green]Successfully saved in cache:[/bold green] {local_path}")
            return local_path
        except Exception as e:
            console.print(f"[bold red]Failed to download:[/bold red] {e}")
            raise FileNotFoundError(f"Failed to download from MLflow: {checkpoint_ref}")

    console.print(f"[bold red]Error: Checkpoint not found:[/bold red] {checkpoint_ref}")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_ref}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test(
    ctx: typer.Context,
    checkpoint_ref: str = typer.Argument(..., help="Local path (e.g. best_model.ckpt) or MLflow URI (e.g. runs:/<run_id>/checkpoints/best_model.ckpt)"),
    config_name: str = typer.Option("config", "--config", "-c", help="Name of the name config file"),
):
    """
    Tests trained model on train dataset using saved weights
    """
    overrides = [arg for arg in ctx.args if "=" in arg]
    
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name=config_name, overrides=overrides)

    # 1. Smart checkpoint search    
    tracking_uri = cfg.trainer.get("logger", {}).get("tracking_uri", os.environ.get("MLFLOW_TRACKING_URI"))    
    actual_checkpoint_path = resolve_checkpoint_path(checkpoint_ref, tracking_uri)
        
    console.print("[bold blue]Compiling DataModule...[/bold blue]")
    datamodule = instantiate(cfg.data)
    
    console.print("[bold blue]Compiling Architecture...[/bold blue]")
    architecture = instantiate(cfg.model)
    
    console.print("[bold blue]Task weights loading...[/bold blue]")
    TaskClass = get_class(cfg.task._target_)
    
    #2. Task class initialization
    task = TaskClass.load_from_checkpoint(
        actual_checkpoint_path, 
        model=architecture
    )
    
    console.print("[bold blue]Initialization of PyTorch Lightning Trainer...[/bold blue]")
    trainer_kwargs = dict(cfg.trainer)
    if 'logger' in trainer_kwargs:
        trainer_kwargs['logger'] = instantiate(trainer_kwargs['logger'])
        
    torch.set_float32_matmul_precision('medium') 
    trainer = pl.Trainer(**trainer_kwargs)
    
    console.print("[bold yellow]Beginning of testing...[/bold yellow]")
    trainer.test(model=task, datamodule=datamodule)
    
    console.print("[bold green]Testing completed.[/bold green]")


if __name__ == "__main__":
    app()

