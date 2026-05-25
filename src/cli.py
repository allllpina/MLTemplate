import typer
from hydra import initialize, compose
from hydra.utils import instantiate
import pytorch_lightning as pl
from rich.console import Console
import torch

from src.models.task import ClassifierTask

app = typer.Typer(help="🔥 Кастомний MLOps CLI для експериментів з нейронками")
console = Console()

@app.command()
def train(
    config_name: str = typer.Option("config", "--config", "-c", help="Ім'я головного файлу конфігурації"),
    overrides: list[str] = typer.Argument(None, help="Перевизначення параметрів (напр. trainer.max_epochs=50)")
):
    """
    Запускає цикл навчання моделі на основі декларативного конфігу.
    """
    console.print(f"[bold green]🚀 Ініціалізація пайплайну з конфігом:[/bold green] {config_name}")
    
    # 1. Завантажуємо конфіги через Hydra
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        
    # 2. Магія Hydra: автоматичне створення об'єктів з yaml
    console.print("[bold blue]📦 Збирання DataModule...[/bold blue]")
    datamodule = instantiate(cfg.data)
    
    console.print("[bold blue]🧠 Збирання Архітектури (CNN + KAN)...[/bold blue]")
    architecture = instantiate(cfg.model)
    
    # Обгортаємо архітектуру в наш LightningTask
    task = ClassifierTask(
        model=architecture, 
        lr=1e-3, 
        num_classes=cfg.model.num_classes
    )
    
    # 3. Налаштування Trainer
    console.print("[bold blue]⚙️ Ініціалізація PyTorch Lightning Trainer...[/bold blue]")
    
    trainer_kwargs = dict(cfg.trainer)
    
    # Якщо в конфігу є логер, створюємо його як об'єкт
    if 'logger' in trainer_kwargs:
        trainer_kwargs['logger'] = instantiate(trainer_kwargs['logger'])
        
    torch.set_float32_matmul_precision('medium') 
    trainer = pl.Trainer(**trainer_kwargs)
    # Налаштовуємо оптимізацію тензорних ядер для твоєї RTX 5070
    torch.set_float32_matmul_precision('medium') 
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # 4. Запуск
    console.print("[bold yellow]🔥 Починаємо тренування![/bold yellow]")
    trainer.fit(model=task, datamodule=datamodule)
    
    console.print("[bold green]✅ Тренування завершено![/bold green]")


@app.command()
def test(
    config_name: str = typer.Option("config", "--config", "-c"),
    checkpoint_path: str = typer.Argument(..., help="Шлях до .ckpt файлу з вагами")
):
    """
    Тестує навчену модель на відкладеній вибірці.
    """
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name=config_name)
        
    datamodule = instantiate(cfg.data)
    architecture = instantiate(cfg.model)
    
    # Завантажуємо ваги з чекпоінту
    task = ClassifierTask.load_from_checkpoint(
        checkpoint_path, 
        model=architecture
    )
    
    trainer = pl.Trainer(**dict(cfg.trainer))
    console.print(f"[bold yellow]📊 Запуск тестування для чекпоінту:[/bold yellow] {checkpoint_path}")
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    app()