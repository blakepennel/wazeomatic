"""Quantization-aware fine-tuning of a YOLO detector for its int8 (QDQ) phone copy.

    python tools/quantize_onnx.py models/coco_yolo11n_dyn.onnx --static --calib <folder> ... \
        --calib-cache runs/qat/coco_ranges.json --out runs/qat/coco_ptq.qdq.onnx
    python tools/qat_onnx.py --weights yolo11n.pt --onnx models/coco_yolo11n_dyn.onnx --ranges runs/qat/coco_ranges.json \
        --images <folder> --images <folder> ... --shapes 256x416,352x416 --out models/coco_yolo11n_dyn.qat.qdq.onnx

(the full command is in models/README.md). It also writes <stem>.float.onnx (the tuned float weights) and
<out>.qat.json (arguments and the per-epoch log). --int8-classes tunes a model whose class branch is int8 too; for the
hazard model that halved the class-score error but still added false Construction clips (docs/model-research.md).

Post-training int8 (tools/quantize_onnx.py --static) rounds the finished float model once. Here the float model is
fine-tuned with that rounding simulated in its forward pass: per-channel symmetric int8 weights, and int8 activations
after every convolution, sigmoid and SiLU product at the ranges ONNX Runtime's calibration picked for those tensors
(--ranges, the cache quantize_onnx.py --calib-cache writes). The weights learn to sit where int8 rounding costs least.

The target is the float model itself (distillation), not the labels: on unlabelled frames the student's class scores
are pulled towards the float teacher's (binary cross-entropy against the teacher's sigmoid scores, everywhere, so
background scores that int8 would inflate are pulled down too) and its box distributions towards the teacher's
(KL divergence, weighted by the teacher's best class score). The int8 copy keeps the float model's behaviour instead
of becoming a new model with its own mistakes.

The tuned weights go back into the float ONNX graph (same node and tensor names: the initialisers are named after
the PyTorch modules), which is then quantized with the same cached ranges, so the int8 file rounds exactly the
tensors the training simulated at the same scales. Frames are cropped at the shapes the app feeds the model
(--shapes, height x width) from the --images folders, --per-source images from each.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

IMG_EXT = {".jpg", ".jpeg", ".png"}


# --- fake quantization, matching onnxruntime.quantization (QInt8 activations, symmetric per-channel QInt8 weights) ---

def act_qparams(lo: float, hi: float) -> tuple[float, int]:
    """Scale and zero point ORT's compute_scale_zp gives an asymmetric int8 tensor with range [lo, hi]."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = (hi - lo) / 255.0
    if scale < np.finfo(np.float32).tiny:
        return 1.0, 0
    return float(np.float32(scale)), int(np.clip(np.round(-128 - lo / scale), -128, 127))


def build_qat_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ActFQ(nn.Module):
        def __init__(self, lo: float, hi: float):
            super().__init__()
            self.scale, self.zp = act_qparams(lo, hi)
            self.enabled = True

        def forward(self, x):
            if not self.enabled:
                return x
            return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zp, -128, 127)

    def weight_fq(w):
        s = (w.detach().abs().amax(dim=(1, 2, 3)) / 127.0).clamp(min=1e-12)
        return torch.fake_quantize_per_channel_affine(w, s, torch.zeros_like(s, dtype=torch.int32), 0, -127, 127)

    class QConv2d(nn.Module):
        """A Conv2d whose weight and output are rounded to int8 the way the QDQ graph rounds them."""

        def __init__(self, conv: nn.Conv2d, out_fq: ActFQ, in_fq: ActFQ | None = None):
            super().__init__()
            self.conv, self.out_fq, self.in_fq, self.enabled = conv, out_fq, in_fq, True

        def forward(self, x):
            c = self.conv
            if self.in_fq is not None:
                x = self.in_fq(x)
            w = weight_fq(c.weight) if self.enabled else c.weight
            return self.out_fq(F.conv2d(x, w, c.bias, c.stride, c.padding, c.dilation, c.groups))

    class QSiLU(nn.Module):
        """x * sigmoid(x) as the graph computes it: Sigmoid and Mul each with a rounded output."""

        def __init__(self, sig_fq: ActFQ, mul_fq: ActFQ):
            super().__init__()
            self.sig_fq, self.mul_fq = sig_fq, mul_fq

        def forward(self, x):
            return self.mul_fq(x * self.sig_fq(torch.sigmoid(x)))

    return ActFQ, QConv2d, QSiLU


def load_ranges(path: Path) -> dict[str, tuple[float, float]]:
    from onnxruntime.quantization.calibrate import load_tensors_data

    data = load_tensors_data(path).data
    out = {}
    for name, td in data.items():
        lo, hi = (td.range_value if hasattr(td, "range_value") else (td.lowest, td.highest))
        out[name] = (float(np.asarray(lo).min()), float(np.asarray(hi).max()))
    return out


def insert_fake_quant(model, graph, ranges: dict, int8_classes: bool) -> dict:
    """Swap the fused model's convolutions and SiLUs for fake-quantized ones where the QDQ graph is int8.
    Returns {module name: ONNX conv node} for every convolution (quantized or not), for writing the weights back."""
    import torch.nn as nn
    from ultralytics.nn.modules import Conv

    from quantize_onnx import float_nodes

    ActFQ, QConv2d, QSiLU = build_qat_modules()
    keep_float = set(float_nodes(graph, int8_classes))
    by_weight = {n.input[1]: n for n in graph.node if n.op_type == "Conv"}
    producer = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    inits = {i.name: i for i in graph.initializer}
    from onnx import numpy_helper

    convs, n_q, n_act, n_in, worst = {}, 0, 0, 0, 0.0
    modules = dict(model.named_modules())
    for name, mod in list(modules.items()):
        if not isinstance(mod, nn.Conv2d) or ".dfl." in name:      # DFL: a fixed float projection
            continue
        node = by_weight.get(f"{name}.weight")
        if node is None:
            raise SystemExit(f"no ONNX Conv with weight {name}.weight: the ONNX is not an export of these weights")
        w = numpy_helper.to_array(inits[f"{name}.weight"])
        worst = max(worst, float(np.abs(w - mod.weight.detach().cpu().numpy()).max()))
        convs[name] = node
        if node.name in keep_float:
            continue
        t_conv = node.output[0]
        parent_name, _, attr = name.rpartition(".")
        # a concatenation or sum feeding the convolution is rounded too (e.g. the C3k2 / SPPF concat, the
        # attention's sum); other ops in between (Resize, MaxPool, Split) keep their input's scale
        src = producer.get(node.input[0])
        in_fq = None
        if src is not None and src.op_type in ("Concat", "Add") and src.name not in keep_float                 and node.input[0] in ranges:
            in_fq = ActFQ(*ranges[node.input[0]])
            n_in += 1
        setattr(modules[parent_name], attr, QConv2d(mod, ActFQ(*ranges[t_conv]), in_fq))
        n_q += 1
        owner = modules.get(parent_name)
        if isinstance(owner, Conv) and isinstance(owner.act, nn.SiLU):
            sig = next((c for c in consumers.get(t_conv, []) if c.op_type == "Sigmoid"), None)
            mul = next((c for c in consumers.get(t_conv, []) if c.op_type == "Mul"), None)
            if sig is None or mul is None:
                raise SystemExit(f"{name}: no Sigmoid / Mul after {t_conv} in the graph")
            owner.act = QSiLU(ActFQ(*ranges[sig.output[0]]), ActFQ(*ranges[mul.output[0]]))
            n_act += 1
    if worst > 1e-4:
        raise SystemExit(f"weights differ from the ONNX by up to {worst:g}: export the ONNX from these weights")
    print(f"fake quantization: {n_q} of {len(convs)} convolutions int8, {n_act} SiLUs, {n_in} concat / sum inputs "
          f"(weights match the ONNX "
          f"to {worst:.1e})")
    return convs


def prefetch(fn, jobs, ex, depth: int = 6):
    """``fn`` over ``jobs`` on the thread pool, at most ``depth`` results ahead of the consumer."""
    from collections import deque

    q, it = deque(), iter(jobs)
    for job in it:
        q.append(ex.submit(fn, job))
        if len(q) >= depth:
            break
    while q:
        r = q.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            q.append(ex.submit(fn, nxt))
        yield r


def set_fake_quant(model, enabled: bool) -> None:
    for m in model.modules():
        if hasattr(m, "enabled"):
            m.enabled = enabled


# --- frames ---

def pick_images(folders: list[str], per_source: int, seed: int) -> list[Path]:
    rng, files = random.Random(seed), []
    for d in folders:
        found = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not found:
            raise SystemExit(f"no images under {d}")
        rng.shuffle(found)
        files += found[:per_source]
        print(f"  {len(found[:per_source]):5d} of {len(found):6d} images from {d}")
    return files


def build_pool(files: list[Path], pool: Path, max_side: int = 960) -> list[Path]:
    """Copy the chosen frames to a local folder at most ``max_side`` px (the sources may be on a slow disk)."""
    pool.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(files):
        dst = pool / f"{i:06d}_{p.stem[:60]}.jpg"
        if not dst.exists():
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            r = max_side / max(h, w)
            if r < 1:
                img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append(dst)
        if i % 1000 == 999:
            print(f"  pool {i + 1} / {len(files)}", flush=True)
    return out


def crop_frame(img: np.ndarray, shape: tuple[int, int], rng: random.Random, train: bool) -> np.ndarray:
    """A crop of the frame at the aspect of ``shape`` (height, width), resized to it: the whole width (centre) for
    validation, a random 45-100 % of the largest such crop at a random place for training."""
    oh, ow = shape
    h, w = img.shape[:2]
    cw = min(w, h * ow / oh)
    if train:
        cw *= rng.uniform(0.45, 1.0)
    ch = cw * oh / ow
    x0 = rng.uniform(0, w - cw) if train else (w - cw) / 2
    y0 = rng.uniform(0, h - ch) if train else (h - ch) / 2
    crop = img[int(y0): int(y0 + ch), int(x0): int(x0 + cw)]
    out = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA if cw > ow else cv2.INTER_LINEAR)
    if train and rng.random() < 0.5:
        out = out[:, ::-1]
    return out


class Frames:
    """Batches of crops; every batch has one of the shapes."""

    def __init__(self, files, shapes, batch, train, seed):
        self.files, self.shapes, self.batch, self.train, self.seed = files, shapes, batch, train, seed
        self.epoch = 0

    def batches(self):
        rng = random.Random(self.seed + 1000 * self.epoch)
        idx = list(range(len(self.files)))
        if self.train:
            rng.shuffle(idx)
        out = []
        for b in range(0, len(idx) - (self.batch - 1 if self.train else 0), self.batch):
            shape = rng.choice(self.shapes) if self.train else self.shapes[(b // self.batch) % len(self.shapes)]
            out.append((idx[b: b + self.batch], shape, rng.randrange(1 << 30)))
        return out

    def load(self, job):
        import torch

        idx, shape, seed = job
        rng = random.Random(seed)
        ims = []
        for i in idx:
            img = cv2.imread(str(self.files[i]))
            ims.append(crop_frame(img, shape, rng, self.train)[:, :, ::-1])
        x = np.ascontiguousarray(np.stack(ims).transpose(0, 3, 1, 2))
        return torch.from_numpy(x)


# --- distillation loss ---

def distill_loss(s: dict, t: dict, reg_max: int = 16, cls_focus: float = 0.0):
    import torch
    import torch.nn.functional as F

    pt = t["scores"].sigmoid()
    # cross-entropy minus the teacher's own entropy: 0 when the scores agree
    kl_cls = (F.binary_cross_entropy_with_logits(s["scores"], pt, reduction="none")
              - F.binary_cross_entropy_with_logits(t["scores"], pt, reduction="none")).sum(1)     # (b, n)
    if cls_focus > 0:
        # weight the anchors where either model scores something (a box the teacher keeps, or one int8 would
        # add): nearly every anchor is plain background, which otherwise swamps the mean
        w = torch.maximum(pt.max(1).values, s["scores"].detach().sigmoid().max(1).values) + 0.01
        cls = cls_focus * (kl_cls * w).sum() / w.sum()
    else:
        cls = kl_cls.mean()
    b, _, n = s["boxes"].shape
    ls = s["boxes"].view(b, 4, reg_max, n).log_softmax(2)
    lt = t["boxes"].view(b, 4, reg_max, n).log_softmax(2)
    kl = (lt.exp() * (lt - ls)).sum(2).sum(1)            # (b, n)
    wgt = pt.max(1).values
    box = (kl * wgt).sum() / wgt.sum().clamp(min=1e-6)
    return cls, box


def score_gap(s: dict, t: dict):
    """Largest and mean absolute class-score difference over anchors where either scores >= 0.1."""
    ps, pt = s["scores"].sigmoid(), t["scores"].sigmoid()
    m = (ps.max(1).values >= 0.1) | (pt.max(1).values >= 0.1)
    d = (ps - pt).abs().max(1).values[m]
    return (float(d.max()) if d.numel() else 0.0), (float(d.mean()) if d.numel() else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="the .pt checkpoint the float ONNX was exported from")
    ap.add_argument("--onnx", required=True, help="its float ONNX export (export_onnx.py --dynamic)")
    ap.add_argument("--ranges", required=True, help="calibration cache from quantize_onnx.py --calib-cache")
    ap.add_argument("--out", required=True, help="the int8 QDQ model to write (also <stem>.float.onnx)")
    ap.add_argument("--images", action="append", required=True, help="folder of training frames (repeatable)")
    ap.add_argument("--per-source", type=int, default=3000, help="frames drawn from each --images folder")
    ap.add_argument("--val", type=int, default=400, help="frames held out to measure the int8 gap")
    ap.add_argument("--pool", help="local folder for the resized frames (default: next to --out)")
    ap.add_argument("--shapes", default="384x640,288x640", help="input shapes, height x width (the app's passes)")
    ap.add_argument("--int8-classes", action="store_true", help="the Detect head's class branch (cv3) goes int8 too")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--box-weight", type=float, default=1.0)
    ap.add_argument("--cls-focus", type=float, default=0.0,
                    help="> 0: class loss weighted towards anchors either model scores, times this factor")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import onnx
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from onnx import numpy_helper
    from ultralytics import YOLO

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    float_out = out.with_name(out.name.removesuffix(".onnx").removesuffix(".qdq") + ".float.onnx")

    print("frames:")
    files = pick_images(args.images, args.per_source, args.seed)
    pool = Path(args.pool) if args.pool else out.parent / (out.stem + "_frames")
    files = build_pool(files, pool)
    random.Random(args.seed).shuffle(files)
    val_files, train_files = files[: args.val], files[args.val:]
    print(f"{len(train_files)} training frames, {len(val_files)} held out, shapes {shapes}")

    graph = onnx.load(args.onnx).graph
    ranges = load_ranges(Path(args.ranges))
    dev = torch.device(args.device)

    def fused(path):
        m = YOLO(path).model.float()
        m.fuse(verbose=False)
        return m.to(dev)

    teacher = fused(args.weights).train()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = fused(args.weights)
    convs = insert_fake_quant(student, graph, ranges, args.int8_classes)
    student.to(dev).train()
    params = [p for n, p in student.named_parameters() if ".dfl." not in n]
    for p in params:
        p.requires_grad_(True)

    train = Frames(train_files, shapes, args.batch, True, args.seed)
    val = Frames(val_files, shapes, args.batch, False, args.seed)
    steps = args.epochs * len(train.batches())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps), pct_start=0.05,
                                                div_factor=10, final_div_factor=10)
    pool_ex = ThreadPoolExecutor(args.workers)

    def run_val(fq=True):
        set_fake_quant(student, fq)
        tot = np.zeros(4)
        n = 0
        with torch.no_grad():
            for x in prefetch(val.load, val.batches(), pool_ex):
                x = x.to(dev).float() / 255
                s, t = student(x), teacher(x)
                cls, box = distill_loss(s, t, cls_focus=args.cls_focus)
                gmax, gmean = score_gap(s, t)
                tot += [float(cls), float(box), gmax, gmean]
                n += 1
        set_fake_quant(student, True)
        tot /= max(n, 1)
        return f"cls {tot[0]:.5f} box {tot[1]:.4f} score gap max {tot[2]:.3f} mean {tot[3]:.4f}"

    print("held out, int8 simulated, before:", run_val(), flush=True)
    log = []
    for ep in range(args.epochs):
        train.epoch = ep
        t0, acc, n = time.time(), np.zeros(2), 0
        for x in prefetch(train.load, train.batches(), pool_ex):
            x = x.to(dev, non_blocking=True).float() / 255
            with torch.no_grad():
                t = teacher(x)
            s = student(x)
            cls, box = distill_loss(s, t, cls_focus=args.cls_focus)
            loss = cls + args.box_weight * box
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            acc += [float(cls), float(box)]
            n += 1
        v = run_val()
        line = f"epoch {ep + 1}/{args.epochs}: train cls {acc[0] / n:.5f} box {acc[1] / n:.4f} | held out {v} | {time.time() - t0:.0f} s"
        print(line, flush=True)
        log.append(line)

    # the tuned weights back into the float graph, then int8 with the same ranges
    model = onnx.load(args.onnx)
    inits = {i.name: i for i in model.graph.initializer}
    modules = dict(student.named_modules())
    for name in convs:
        conv = modules[name].conv if hasattr(modules[name], "conv") and not hasattr(modules[name], "weight") else modules[name]
        for kind in ("weight", "bias"):
            if getattr(conv, kind) is None:
                continue
            init = inits[f"{name}.{kind}"]
            new = getattr(conv, kind).detach().cpu().numpy().astype(np.float32)
            if tuple(init.dims) != new.shape:
                raise SystemExit(f"{name}.{kind}: shape {tuple(init.dims)} in the ONNX, {new.shape} trained")
            init.CopyFrom(numpy_helper.from_array(new, init.name))
    onnx.save(model, str(float_out))
    labels = Path(args.onnx).with_suffix(".labels.txt")
    if labels.exists():
        float_out.with_suffix(".labels.txt").write_text(labels.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {float_out}")
    cmd = [sys.executable, str(ROOT / "tools" / "quantize_onnx.py"), str(float_out), "--static",
           "--calib-cache", str(args.ranges), "--out", str(out)] + (["--int8-classes"] if args.int8_classes else [])
    subprocess.run(cmd, check=True)
    (out.with_suffix(".qat.json")).write_text(json.dumps({"args": vars(args), "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
