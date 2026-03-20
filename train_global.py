import random

import torch
from data.ps import PSCountingDataset
from data.gf import GFCountingDataset
from data.spot import SPOTCountingDataset
# from models.dm_count_unbalanced import Trainer
from models.dm_count import Trainer
from models.backbones import *
import hydra
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
import os
import datetime
import wandb
import tqdm
from sklearn.metrics import r2_score
import numpy as np
from eval import game_batched
from torch.utils.data import DataLoader, random_split, ConcatDataset
from itertools import cycle
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

def get_preds(loader, model, device):
    device = torch.device(device)
    preds = []
    targets = []
    with torch.no_grad():
        for inp, tgt in loader:
            inputs = inp.to(device)
            valid = inp[:, [-1, ]].to(device)
            with torch.no_grad():
                outputs = nn.functional.relu(model(inputs)) * valid
            preds.extend(outputs.cpu())
            targets.extend(tgt.cpu())
    return torch.cat(preds), torch.cat(targets)


def evaluate(preds, gts):
    # patch-level
    pred_counts = preds.view(preds.shape[0], -1).sum(dim=1)
    tgt_counts = gts.view(gts.shape[0], -1).sum(dim=1)
    nmae = (torch.abs(tgt_counts - pred_counts) / tgt_counts.mean()).mean().item()
    rmse = torch.sqrt(((tgt_counts - pred_counts) ** 2).mean()).item()

    return nmae


@hydra.main(version_base=None, config_path="conf", config_name="train_all")
def train(cfg):
    device = cfg.train.device

    # instantiate dataset
    train_datasets = {
        "spot": SPOTCountingDataset(imsize=64, split="train"),
        "ps": PSCountingDataset(imsize=64, split="train"),
        "gf": GFCountingDataset(imsize=64, split="train")
    }
    train_dataset = ConcatDataset(train_datasets.values())

    test_datasets = {
        "spot": SPOTCountingDataset(imsize=64, split="test"),
        "ps": PSCountingDataset(imsize=64, split="test"),
        "gf": GFCountingDataset(imsize=64, split="train")
    }
    test_dataset = ConcatDataset(test_datasets.values())

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

    # instantiate backbone
    backbone = hydra.utils.instantiate(cfg.backbone)

    # instantiate model (within trainer)
    trainer = hydra.utils.instantiate(cfg.model)
    trainer.setup(backbone)

    logger = wandb.init(
        project="treedensity",
        config=OmegaConf.to_container(cfg),
        reinit="create_new"
    )

    logdir = os.path.join(cfg.train.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(logdir, exist_ok=True)
    # write conf to logdir
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    best_nmae = 1e8
    best_metrics = {}
    log_metrics = ["nmae", "rmse"]
    for epoch in range(cfg.train.nepoch):
        trainer.train()  # Set model to training mode
        for step, (inputs, gt_discrete) in enumerate(train_loader):
            trainer.train_step(inputs, gt_discrete, logger)

        if (epoch + 1) % cfg.train.val_freq == 0:
            trainer.eval()
            with torch.no_grad():
                # test
                test_pred, test_target = get_preds(test_loader, trainer, device)
                test_metrics = evaluate(test_pred, test_target)
                logger.log({"test/"+k: v for k, v in test_metrics.items() if k in log_metrics})

                if test_metrics["rmse"] < best_nmae:
                    best_nmae = test_metrics["rmse"]
                    best_metrics = test_metrics
                    print(f"found best metrics ")
                    print(best_metrics)
                    logger.summary.update(best_metrics)
                    torch.save(backbone.state_dict(), os.path.join(logdir, "best_model.pth"))

            if hasattr(trainer, "scheduler"):
                trainer.scheduler.step()

    logger.finish()


def evaluate_global():
    from data.gf import GFCountingDataset
    from data.ps import PSCountingDataset
    from data.spot import SPOTCountingDataset
    from models.backbones import UNetR50
    from omegaconf import OmegaConf
    from itertools import product
    from torch.utils.data import DataLoader
    from matplotlib.patches import Circle


    datasets = {
        "gf": GFCountingDataset(imsize=64, split="test"),
        "ps": PSCountingDataset(imsize=64, split="test"),
        "spot": SPOTCountingDataset(imsize=64, split="test")
    }
    datasets["global"] = ConcatDataset(datasets.values())

    device = torch.device("cuda:1")

    ckpts = OmegaConf.load("conf/ckpt.yaml")

    models = {
        "gf_udm": UNetR50(in_channels=4),
        "spot_udm": UNetR50(in_channels=4),
        "ps_udm": UNetR50(in_channels=4),
        "global_udm": UNetR50(in_channels=4)
    }

    # load models
    for m in models:
        ckpt_path = ckpts[f"{m}"]
        print(ckpt_path)
        ckpt = torch.load(ckpt_path)
        models[m].load_state_dict(ckpt)
        models[m].to(device)
        models[m].eval()

    for d_train, d_eval in product(["ps", "gf", "spot", "global"], ["ps", "gf", "spot", "global"]):

        # create loader
        dataset = datasets[d_eval]
        loader = DataLoader(dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=True)

        with torch.no_grad():
            test_pred, test_target = get_preds(loader, models[f"{d_train}_udm"], device)
            nmae = evaluate(test_pred, test_target)

        print(f"training on {d_train} and evaluating on {d_eval}: nmae = {nmae}")


if __name__ == "__main__":
    # train()
    evaluate_global()