import torch
import lightning as L
from data import CIFARDataModule
from model import CIFARLitModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def train(from_load=None, to_load="./nn.pth"):

    wandb_logger = L.pytorch.loggers.WandbLogger(log_model="all")

    dm = CIFARDataModule()
    model = CIFARLitModel(*dm.dims, dm.num_classes, hidden_size=256)
    checkpoint_callback = ModelCheckpoint(dirpath='model-chkp/')
    early_stopping = EarlyStopping('val_loss')

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")
    if from_load:
        model.load_state_dict(torch.load(from_load))
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping]
    )
    trainer.fit(model, dm)
    if to_load:
        torch.save(model.state_dict(), to_load)

# train(to_load="nn2.pth")
if __name__ == "__main__":
    train(to_load="nn.pth")