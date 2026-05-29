#!/usr/bin/env python3

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segformer_potsdam_common import (
    CLASS_IDS,
    IGNORE_INDEX,
    NUM_CLASSES,
    PotsdamPatchDataset,
    build_image_ids,
    compute_metrics_from_confusion_matrix,
    id2label,
    label2id,
    load_yaml_config,
    prepare_output_dir,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config):
    device_config = str(config.get("device", {}).get("type", "auto")).lower()
    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device.type=cuda but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device_config == "cpu":
        return torch.device("cpu")
    if device_config != "auto":
        raise ValueError(f"Unknown device.type={device_config!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_segformer_model(config):
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("transformers is required for SegFormer training") from exc

    model_config = config.get("model", {})
    model_name = model_config.get(
        "pretrained_model_name_or_path",
        "nvidia/segformer-b0-finetuned-ade-512-512",
    )
    local_files_only = bool(model_config.get("local_files_only", False))

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=id2label(),
        label2id=label2id(),
        ignore_mismatched_sizes=bool(model_config.get("ignore_mismatched_sizes", True)),
        local_files_only=local_files_only,
    )
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    return model


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "image_id": [item["image_id"] for item in batch],
    }


def confusion_from_batch(pred, labels):
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    valid = labels != IGNORE_INDEX
    pred_flat = pred[valid].reshape(-1)
    gt_flat = labels[valid].reshape(-1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    if pred_flat.size > 0:
        valid_pred = (pred_flat >= 0) & (pred_flat < NUM_CLASSES)
        np.add.at(
            cm,
            (gt_flat[valid_pred].astype(np.int64), pred_flat[valid_pred].astype(np.int64)),
            1,
        )
    return cm


@torch.no_grad()
def validate(model, loader, device, amp_enabled=False):
    model.eval()
    dataset_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc="Validate", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with autocast_context:
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        total_loss += float(outputs.loss.detach().cpu())
        total_batches += 1
        pred = upsampled_logits.argmax(dim=1)
        dataset_cm += confusion_from_batch(pred, labels)

    metrics = compute_metrics_from_confusion_matrix(dataset_cm)
    metrics["loss"] = total_loss / max(total_batches, 1)
    metrics["confusion_matrix"] = dataset_cm.tolist()
    return metrics


def save_checkpoint(model, output_dir, name, epoch, metrics, optimizer=None):
    checkpoint_dir = Path(output_dir) / "checkpoints" / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    state = {
        "epoch": int(epoch),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, checkpoint_dir / "training_state.pt")
    with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer on Potsdam semantic segmentation")
    parser.add_argument("--config", default="segformer_potsdam_rgb.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = prepare_output_dir(config, args.config)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = resolve_device(config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_config = config.get("data", {})
    train_config = config.get("training", {})
    split_config = config.get("splits", {})
    image_processing = config.get("image_processing", {})
    mean = image_processing.get("mean", [0.485, 0.456, 0.406])
    std = image_processing.get("std", [0.229, 0.224, 0.225])
    patch_size = int(image_processing.get("patch_size", 512))
    train_stride = int(image_processing.get("train_stride", patch_size))
    val_stride = int(image_processing.get("val_stride", patch_size))
    min_valid_ratio = float(image_processing.get("min_valid_ratio", 0.05))

    train_ids = build_image_ids(split_config, "train")
    val_ids = build_image_ids(split_config, "val")
    split_overlap = sorted(set(train_ids) & set(val_ids))
    if split_overlap:
        raise ValueError(f"Train/val split overlap detected: {split_overlap}")

    train_dataset = PotsdamPatchDataset(
        config,
        train_ids,
        patch_size=patch_size,
        stride=train_stride,
        mean=mean,
        std=std,
        min_valid_ratio=min_valid_ratio,
        augment=bool(data_config.get("augment", True)),
        cache_tiles=bool(data_config.get("cache_tiles", False)),
    )
    val_dataset = PotsdamPatchDataset(
        config,
        val_ids,
        patch_size=patch_size,
        stride=val_stride,
        mean=mean,
        std=std,
        min_valid_ratio=min_valid_ratio,
        augment=False,
        cache_tiles=bool(data_config.get("cache_tiles", False)),
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset has no patches")
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset has no patches")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=bool(train_config.get("drop_last", False)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_config.get("eval_batch_size", train_config.get("batch_size", 4))),
        shuffle=False,
        num_workers=int(train_config.get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = load_segformer_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 6e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    epochs = int(train_config.get("epochs", 40))
    grad_accum = int(train_config.get("gradient_accumulation_steps", 1))
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    run_context = {
        "config_path": args.config,
        "output_dir": str(output_dir),
        "device": str(device),
        "train_images": train_ids,
        "val_images": val_ids,
        "num_train_patches": len(train_dataset),
        "num_val_patches": len(val_dataset),
        "class_ids": CLASS_IDS,
        "ignore_index": IGNORE_INDEX,
    }
    with open(output_dir / "metrics" / "run_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2, ensure_ascii=False)

    history = []
    best_miou = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        num_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                if train_config.get("max_grad_norm") is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(train_config.get("max_grad_norm")),
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach().cpu()) * grad_accum
            num_steps += 1
            progress.set_postfix(loss=running_loss / max(num_steps, 1))

        train_loss = running_loss / max(num_steps, 1)
        val_metrics = validate(model, val_loader, device, amp_enabled=amp_enabled)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
        }
        history.append(epoch_record)
        with open(output_dir / "metrics" / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_mIoU={val_metrics['mean_iou']:.4f}, "
            f"val_OA={val_metrics['overall_accuracy']:.4f}"
        )
        save_checkpoint(model, output_dir, "latest", epoch, epoch_record, optimizer=optimizer)
        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            best_epoch = epoch
            save_checkpoint(model, output_dir, "best", epoch, epoch_record, optimizer=optimizer)

    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_miou,
        "num_epochs": epochs,
        "run_context": run_context,
    }
    with open(output_dir / "metrics" / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Training complete. Best epoch={best_epoch}, best val mIoU={best_miou:.4f}")


if __name__ == "__main__":
    main()
