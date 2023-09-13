import os
import torch # PyTorch
from torch import nn # PyTorch model builder
from torchvision import transforms # transforms for the data
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split # dataloader
import pytorch_lightning as pl # lightning
from torchmetrics import Accuracy, MaxMetric, MeanMetric

# TO DELETE
#from src.models.components.Resnet50 import Resnet50

class FoodVisionMini(pl.LightningModule):
    def __init__(self,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    out_classes: int
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['net'], logger=False) # net already in model

        # print(self.parameters())
        self.out_classes = out_classes
        self.net = net
        self.criterion = nn.CrossEntropyLoss() # loss

        ## METRICS -----------------------------------------------------------
        # Averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.pred_loss = MeanMetric()

        # Calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=out_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=out_classes)
        self.pred_acc = Accuracy(task="multiclass", num_classes=out_classes)

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        ## -------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def model_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss # return loss for backprop

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # as a value (.compute()) and not a metric so that it's not reseted every step
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        return preds, targets

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        return optimizer

if __name__ == "__main__":
    _ = FoodVisionMini()
