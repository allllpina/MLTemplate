import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

class ClassifierTask(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, num_classes: int = 10):
            super().__init__()
            self.save_hyperparameters(ignore=['model'])
            self.model = model
            
            self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        
        # Логування
        acc = self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)

        acc = self.val_acc(logits, y)
        f1 = self.val_f1(logits, y)

        self.log(
            'val_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            'val_acc',
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            'val_f1',
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

    def configure_optimizers(self):
        # Класичний AdamW
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        
        # Scheduler для поступового зменшення learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }