from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.pylogger.get_pylogger(__name__)

def train(cfg: DictConfig):
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    # Set seed if given
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule:
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model:
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # NOT INDISPENSABLE:
    #---------------------------------------------------------------------------
    # Instantiate callbacks:
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiators.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate logger:
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiators.instantiate_loggers(cfg.get("loggers"))
    #---------------------------------------------------------------------------

    # Instantiate trainer:
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger) #callbacks=callbacks,

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.logging_utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("test"):
        trainer.test(model=model, datamodule=datamodule)

    # I will do a predict here just to see the classes that it outputs:
    if cfg.get("predict"):
        preds_targets = trainer.predict(model=model, datamodule=datamodule)
        preds = torch.cat([preds_targets[i][0] for i in range(len(preds_targets))])
        targets = torch.cat([preds_targets[i][1] for i in range(len(preds_targets))])
        #print(f"Preds:\n{preds.tolist()}")
        #print(f"Targets:\n{targets.tolist()}")
        print(f"Unique predictions: {preds.unique().tolist()}")
    return

from omegaconf import OmegaConf

@hydra.main(version_base="1.3", config_path="../configs", config_name="train_NLP.yaml")
def main(cfg: DictConfig) -> None:

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    #print(OmegaConf.to_yaml(cfg))
    train(cfg)
    return

if __name__ == "__main__":
    main()
