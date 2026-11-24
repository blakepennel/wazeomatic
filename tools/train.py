"""Fine-tune a YOLO11 model on the merged hazard dataset.

    python tools/train.py --name hazard_v3n                      # defaults: yolo11n, 640 px, 40 epochs, batch 32
    python tools/train.py --model yolo11s.pt --batch 24 --name hazard_v3s

Put the dataset on an SSD (WAZEOMATIC_DATASETS or configs/local.yaml): from a hard disk the data
loader starves the GPU.

Expects the dataset produced by tools/prepare_datasets.py (data yaml path below).
Checkpoints land in runs/train/<name>/weights/{best,last}.pt; export them with
tools/export_onnx.py.

--dw-box-head makes the Detect head's box branch depthwise-separable (a 3x3 depthwise and a 1x1 convolution in place
of each 3x3 convolution, as its class branch already is). Those two 3x3 convolutions of 64 channels per scale are
about a quarter of the nano's work at 640 x 384; separable they are an eighth of that. Every other weight comes from
--model; the new layers start untrained, so train the box branch alone first (--freeze 23 --freeze-cls), then the
whole model:

    python tools/train.py --data <datasets>/hazard_v4.yaml --model runs/train/hazard_v4n/weights/best.pt
        --dw-box-head --freeze 23 --freeze-cls --epochs 6 --close-mosaic 0 --name hazard_v4dw_head
    python tools/train.py --data <datasets>/hazard_v4.yaml --model runs/train/hazard_v4dw_head/weights/best.pt
        --dw-box-head --epochs 20 --lr0 0.002 --warmup 1 --name hazard_v4dw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running from the repo without installing
from wazeomatic.paths import datasets_root, merged_dataset  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = merged_dataset() / "data_train.yaml"


def dw_box_head(model) -> None:
    """Replace each scale's box branch Conv(x, c2, 3), Conv(c2, c2, 3) by DWConv(x, 3) + Conv(x, c2, 1),
    DWConv(c2, 3) + Conv(c2, c2, 1); the last 1x1 convolution is kept."""
    import torch.nn as nn
    from ultralytics.nn.modules import Conv, DWConv

    det = model.model[-1]
    for i, seq in enumerate(det.cv2):
        if isinstance(seq[0], nn.Sequential):          # already separable (a checkpoint of this kind)
            continue
        x, c2 = seq[0].conv.in_channels, seq[0].conv.out_channels
        det.cv2[i] = nn.Sequential(nn.Sequential(DWConv(x, x, 3), Conv(x, c2, 1)),
                                   nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1)), seq[2])


def load_matching(model, weights) -> None:
    """Copy every tensor of ``weights`` whose name and shape this model has (DetectionModel.load assumes the
    standard box branch)."""
    src = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
    csd, own = src.float().state_dict(), model.state_dict()
    keep = {k: v for k, v in csd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    print(f"dw-box-head: {len(keep)} of {len(own)} tensors from the checkpoint")


def dw_trainer():
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DWBoxTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            import copy

            import torch.nn as nn

            src = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
            if src is None or not isinstance(src.model[-1].cv2[0][0], nn.Sequential):
                # a standard checkpoint: DetectionModel.load also rebuilds the class branch at the checkpoint's
                # width (80 hidden channels in a model fine-tuned from the COCO weights, 64 when built for 7
                # classes), then the box branch is swapped
                model = super().get_model(cfg, weights, verbose)
                dw_box_head(model)
            else:
                # a checkpoint that already has the separable box branch: take its head's layout, then its weights
                model = super().get_model(cfg, None, verbose)
                model.model[-1].cv3 = copy.deepcopy(src.model[-1].cv3)
                dw_box_head(model)
                load_matching(model, src)
            return model

    return DWBoxTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--model", default="yolo11n.pt", help="starting weights (COCO pretrained)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="hazard")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache", default="ram", help="ultralytics cache: ram, disk or false (ram avoids slow disks)")
    ap.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (use ~0.002 to fine-tune a finished run)")
    ap.add_argument("--warmup", type=float, default=3.0, help="warm-up epochs")
    ap.add_argument("--close-mosaic", type=int, default=10, help="turn mosaic off for the last N epochs")
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (23: train the head alone)")
    ap.add_argument("--dw-box-head", action="store_true", help="depthwise-separable box branch (see above)")
    ap.add_argument("--freeze-cls", action="store_true",
                    help="with --freeze 23: keep the head's class branch frozen too, so only the box branch trains "
                         "(while the boxes are poor the class targets, weighted by box overlap, are near zero)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        resume=args.resume,
        cache=False if str(args.cache).lower() == "false" else args.cache,
        # dashcam-friendly augmentation: no vertical flips (potholes do not hang from the sky),
        # mild colour jitter for exposure differences, mosaic off for the last epochs
        fliplr=0.5, flipud=0.0, degrees=2.0, translate=0.1, scale=0.4, shear=1.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=args.close_mosaic, mixup=0.05,
        cos_lr=True, lr0=args.lr0, warmup_epochs=args.warmup, plots=True, verbose=True,
        **({"freeze": list(range(args.freeze)) + (["23.cv3"] if args.freeze_cls else [])} if args.freeze else {}),
        **({"trainer": dw_trainer()} if args.dw_box_head else {}),
    )
    print("best weights:", ROOT / "runs" / "train" / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
