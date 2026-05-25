"""SA-2: Diffusion Policy training with checkpointing at locked epochs.

`train_run` is backend-agnostic: it takes a dataset provider that returns
(train_ds, val_ds, info), so the same loop drives the synthetic smoke test and
the real LIBERO/Robomimic runs. Per-epoch train loss and validation L1 are
logged; checkpoints are written only at the locked milestone epochs.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from drc import config
from drc.data.dataset import collate
from drc.policy.diffusion_policy import DiffusionUnetImagePolicy
from drc.seeds import seed_everything
from drc.utils import ckpt_path, ensure_dir, get_logger, path, save_json

log = get_logger("train")


def build_policy(info: dict, train_cfg: dict, backbone: str = "smallcnn") -> DiffusionUnetImagePolicy:
    p = train_cfg.get("policy", {})
    return DiffusionUnetImagePolicy(
        action_dim=info["action_dim"],
        image_shape=info["image_shape"],
        proprio_dim=info["proprio_dim"],
        horizon=p.get("horizon", 16),
        n_action_steps=p.get("n_action_steps", 8),
        n_obs_steps=p.get("n_obs_steps", 2),
        num_train_timesteps=p.get("num_train_timesteps", 100),
        num_inference_steps=p.get("num_inference_steps", 16),
        crop_shape=info.get("crop_shape", tuple(p.get("crop_shape", (76, 76)))),
        crop_pad=train_cfg.get("augmentation", {}).get("random_crop_pad", 4),
        backbone=backbone,
    )


@torch.no_grad()
def validation_l1(policy, val_loader, device, K: int) -> float:
    """Mean per-dim L1 over the prediction horizon — the M1 quantity (PRD 8.1)."""
    policy.eval()
    errs = []
    for batch in val_loader:
        obs = {k: v.to(device) for k, v in batch["obs"].items()}
        expert = batch["action"].to(device)
        pred = policy.predict_deterministic(obs, K=K)
        errs.append((pred - expert).abs().mean().item())
    return float(np.mean(errs)) if errs else float("nan")


def train_run(
    task: str,
    seed: int,
    dataset_provider,
    train_cfg: dict | None = None,
    backbone: str = "smallcnn",
    device: str = "cpu",
    epochs: int | None = None,
    checkpoint_epochs=None,
    num_workers: int = 0,
    val_K: int = 10,
    log_every: int = 1,
) -> dict:
    train_cfg = train_cfg or config.load_train_cfg()
    epochs = epochs or train_cfg.get("num_epochs", config.NUM_EPOCHS)
    checkpoint_epochs = list(checkpoint_epochs or train_cfg.get("checkpoint_epochs", config.CHECKPOINT_EPOCHS))
    bs = train_cfg.get("batch_size", 64)
    opt_cfg = train_cfg.get("optimizer", {})

    seed_everything(seed)
    device = torch.device(device)

    train_ds, val_ds, info = dataset_provider()
    policy = build_policy(info, train_cfg, backbone=backbone).to(device)
    policy.normalizer.fit(train_ds.all_actions_flat().to(device))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=opt_cfg.get("lr", 1e-4), weight_decay=opt_cfg.get("weight_decay", 1e-6)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA weights for the deployed checkpoint (DP recipe).
    ema_cfg = train_cfg.get("ema", {"enabled": True, "decay": 0.9999})
    ema_model = copy.deepcopy(policy) if ema_cfg.get("enabled", True) else None
    ema_decay = ema_cfg.get("decay", 0.9999)

    history = []
    for epoch in range(1, epochs + 1):
        policy.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            obs = {k: v.to(device) for k, v in batch["obs"].items()}
            batch = {"obs": obs, "action": batch["action"].to(device)}
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if ema_model is not None:
                with torch.no_grad():
                    for ep, p in zip(ema_model.parameters(), policy.parameters()):
                        ep.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                    for eb, b in zip(ema_model.buffers(), policy.buffers()):
                        eb.copy_(b)
        scheduler.step()

        eval_model = ema_model if ema_model is not None else policy
        val_l1 = validation_l1(eval_model, val_loader, device, K=val_K)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_l1": val_l1, "wallclock_s": time.time() - t0}
        )
        if epoch % log_every == 0:
            log.info(f"{task} s{seed} ep{epoch}/{epochs} loss={train_loss:.4f} val_l1={val_l1:.4f}")

        if epoch in checkpoint_epochs:
            cp = ckpt_path(task, seed, epoch)
            ensure_dir(path("checkpoints", task, str(seed)))
            torch.save(
                {
                    "model_state": eval_model.state_dict(),
                    "info": info,
                    "policy_cfg": train_cfg.get("policy", {}),
                    "backbone": backbone,
                    "task": task,
                    "seed": seed,
                    "epoch": epoch,
                    "val_l1": val_l1,
                },
                cp,
            )

    log_path = path("checkpoints", task, str(seed), "train_log.json")
    save_json({"task": task, "seed": seed, "history": history}, log_path)
    return {"history": history, "info": info}


def load_policy(ckpt_file: str, train_cfg: dict | None = None, device: str = "cpu"):
    """Reconstruct a policy from a checkpoint for SA-3 / SA-4."""
    train_cfg = train_cfg or config.load_train_cfg()
    cp = torch.load(ckpt_file, map_location=device, weights_only=False)
    policy = build_policy(cp["info"], train_cfg, backbone=cp.get("backbone", "smallcnn")).to(device)
    policy.load_state_dict(cp["model_state"])
    policy.eval()
    return policy, cp
