 
###########################################################
# Ce script constitue le benchmark du marché cadre pour le
# domaine IA
##############################################################
# Le calculateur du domaine IA à proposer pour le marché cadre
# devra permettre de faire tourner 500 fois ce script en
# même temps en moins de 15 minutes sur un jeu de données
# propre à chaque exemplaire du script
##############################################################
# Il est issu du travail publié en 2009 par Alex Krizhevsky
# Pour utiliser ce script, il faut avoir téléchargé le jeu
# de données depuis https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# (163 Mo)
# Pour obtenir la performance de ce benchmark, il faut
# mesurer son temps d'exécution "elapse" (par exemple
# avec la commande time)
# Le dimensionnement du calculateur pour le niveau de
# performance demandé est à déduire via une règle de trois
# à partir du temps mesuré pour ce script sur le serveur
# de test
# Le détail de la configuration du serveur de test utilisé, le
# temps mesuré pour le script et la déduction de la taille
# du calculateur proposé devra figurer dans le mémoire
# technique
#pip3 install --quiet "torch>=1.6, <1.9" "torchmetrics>=0.3" "lightning-bolts" "pytorch-lightning>=1.3" "torchvision"
#python3 ./run_IA.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from  xpu import XPUAccelerator
import intel_extension_for_pytorch
#import ipex

seed_everything(0)
PATH_DATASETS = os.environ.get("PATH_DATASETS", "data/cifar-10-batches-p/y")
AVAIL_GPUS = min(1, torch.xpu.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS#,
    #train_transforms=train_transforms,
    #test_transforms=test_transforms,
    #val_transforms=test_transforms,
)

cifar10_dm.train_transforms=train_transforms
cifar10_dm.test_transforms=test_transforms
cifar10_dm.val_transforms=test_transforms

def create_model():
    model = torchvision.models.resnet152(pretrained=False, num_classes=10)
    #model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #model.maxpool = nn.Identity()
    return model
class LightningResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model()
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss
        
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,momentum=0.9,weight_decay=5e-4,
                )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
                "scheduler": OneCycleLR(
                    optimizer,
                    0.1,
                    epochs=self.trainer.max_epochs,
                    steps_per_epoch=steps_per_epoch,
                    ),
                "interval": "step",
                }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        
def main():
    model = LightningResnet(lr=0.05)
    model.datamodule = cifar10_dm
    model.to("xpu")
    print("xpu pvc setting ")
    
    accelerator = XPUAccelerator()
    #trainer = Trainer(accelerator=accelerator, devices=1)
    trainer = Trainer(
            accelerator=accelerator,
            #progress_bar_refresh_rate=10,
            max_epochs=30,
            gpus=1,
            logger=TensorBoardLogger("lightning_logs/", name="resnet"),
                callbacks=[LearningRateMonitor(logging_interval="step")],
            )
    print("HEllo World")
    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)


if __name__ == '__main__':
    print("I AM HERE !")
    main()
