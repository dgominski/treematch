import random
from models.backbones import *
import hydra
from omegaconf import OmegaConf
import os
import datetime
import wandb
from sklearn.metrics import r2_score
import numpy as np
from torch.utils.data import DataLoader, random_split
from itertools import cycle


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')


def get_preds(loader, trainer, device):
    device = torch.device(device)
    trainer.backbone.eval()
    preds = []
    valids = []
    targets = []
    with torch.no_grad():
        for inp, valid, tgt in loader:
            output = trainer.predict(inp.to(device))
            preds.extend(output.cpu())
            valids.append(valid.cpu())
            targets.extend(tgt.cpu())
    return torch.cat(preds), torch.cat(targets), torch.cat(valids)


def evaluate(preds, gts, gsd_m, masks):
    # Apply validity mask — zero out ignored pixels before any metric computation
    masks = masks.float()
    preds = preds * masks
    gts = gts * masks

    # Count only labeled pixels per sample for normalisation
    pixel_counts = masks.view(masks.shape[0], -1).sum(dim=1).clamp(min=1)

    # patch-level counts (sum over labeled pixels only)
    pred_counts = preds.view(preds.shape[0], -1).sum(dim=1)
    tgt_counts = gts.view(gts.shape[0], -1).sum(dim=1)

    r2 = r2_score(tgt_counts.numpy(), pred_counts.numpy())
    mae = torch.abs(tgt_counts - pred_counts).mean().item()
    nmae = mae / (tgt_counts.mean().item() + 1e-8)

    # RMSE in objects/ha: scale counts by (10_000 / patch_area_m2)
    patch_area_m2 = pixel_counts * (gsd_m ** 2)          # m² of labeled pixels per sample
    counts_to_ha = 10_000 / patch_area_m2                 # scalar to convert raw count → objects/ha
    pred_ha = pred_counts * counts_to_ha
    tgt_ha  = tgt_counts  * counts_to_ha
    rmse = torch.sqrt(((tgt_ha - pred_ha) ** 2).mean()).item()

    metrics = {
        "r2": r2,
        "mae": mae,
        "nmae": nmae,
        "rmse": rmse,
    }
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg):
    device = cfg.train.device

    # instantiate dataset
    train_dataset = hydra.utils.instantiate(cfg.dataset, split="train")
    test_dataset = hydra.utils.instantiate(cfg.dataset, split="test")

    if cfg.train.clean_ratio < 1.0:
        train_dataset_noisy = hydra.utils.instantiate(cfg.dataset_noisy)
        clean_batch_size = int(cfg.train.batch_size * cfg.train.clean_ratio)
        noisy_batch_size = cfg.train.batch_size - clean_batch_size
        noisy_loader = cycle(DataLoader(train_dataset_noisy, batch_size=noisy_batch_size, shuffle=True, num_workers=cfg.train.num_workers))
    else:
        clean_batch_size = cfg.train.batch_size
        noisy_batch_size = 0

    if cfg.model.name in ["p2p", "centernet"]:
        # keep a small validation split for hparam tuning
        train_dataset, val_dataset = random_split(train_dataset, lengths=[0.9, 0.1])
        val_loader = DataLoader(val_dataset, batch_size=clean_batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=clean_batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
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
    log_metrics = ["nmae", "rmse"]
    for epoch in range(cfg.train.nepoch):
        trainer.train()  # Set model to training mode
        for step, (inputs, gt_discrete) in enumerate(train_loader):
            if noisy_batch_size > 0:
                inputs_noisy, gt_discrete_noisy = next(noisy_loader)
                # concatenate
                inputs = torch.cat([inputs, inputs_noisy], dim=0)
                gt_discrete = torch.cat([gt_discrete, gt_discrete_noisy], dim=0)
            trainer.train_step(inputs, gt_discrete, logger)

        if (epoch + 1) % cfg.train.val_freq == 0:
            trainer.eval()
            if cfg.model.name in ["p2p", "centernet"]:
                # run hparam sweep
                trainer.hparam_sweep(val_loader)

            with torch.no_grad():
                # test
                test_pred, test_target = get_preds(test_loader, trainer, device)
                test_metrics = evaluate(test_pred, test_target)
                logger.log({"test/"+k: v for k, v in test_metrics.items() if k in log_metrics})

                if test_metrics["nmae"] < best_nmae:
                    best_nmae = test_metrics["nmae"]
                    best_metrics = test_metrics
                    print(f"found best metrics ")
                    print(best_metrics)
                    logger.summary.update(best_metrics)
                    torch.save(backbone.state_dict(), os.path.join(logdir, "best_model.pth"))

            if hasattr(trainer, "scheduler"):
                trainer.scheduler.step()

    logger.finish()


if __name__ == "__main__":
    train()
