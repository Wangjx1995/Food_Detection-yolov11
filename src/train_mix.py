#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-epoch data mixing driver for Ultralytics YOLOv11:
- Keeps 70% real + 30% synthetic for train/val/test each epoch
- Regenerates synthetic split-wise every epoch
- No change to YOLO source code required

Requires: pip install ultralytics pillow numpy
"""
import os, sys, json, math, shutil, random
from pathlib import Path
from typing import List, Tuple

# User settings (EDIT THESE)
REAL_ROOT = Path("real")          # your real YOLO dataset root
ASSETS_DIR = Path("assets")       # generator assets root
OUT_BASE  = Path("out_epoch")     # synthetic output base
MODEL_WEIGHTS = "yolo11n.pt"      # or your checkpoint
EPOCHS = 20
IMGSZ = 640
BATCH = 16
DEVICE = 0                        # -1 for CPU; "0,1" for multi-GPU

REAL_FRACTION = 0.70
SYN_FRACTION  = 0.30

GEN_SCRIPT = Path("synth_yolo_generator_plus.py")
IMAGE_SIZE = (1280, 720)
MIN_OBJS, MAX_OBJS = 1, 4
CLASS_RATIOS = {}
PER_CLASS_MINMAX = {}
ALLOW_OVERLAP = True

def list_pairs(root: Path, split: str) -> List[Path]:
    img_dir = root/"images"/split
    lbl_dir = root/"labels"/split
    imgs = {}
    for p in img_dir.glob("*.*"):
        imgs.setdefault(p.stem, p)
    pairs = []
    for l in sorted(lbl_dir.glob("*.txt")):
        if l.stem in imgs:
            pairs.append(imgs[l.stem])
    return sorted(pairs)

def ensure_import(gen_script: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("synth_gen_plus", str(gen_script.resolve()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["synth_gen_plus"] = mod
    spec.loader.exec_module(mod)
    return mod

def synth_needed(n_real: int) -> int:
    return max(0, int(round(n_real * SYN_FRACTION / REAL_FRACTION)))

def write_list(paths, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p.resolve()).replace("\\","/") + "\n")

def load_names(real_root: Path):
    y = real_root/"dataset.yaml"
    if y.exists():
        import re
        txt = y.read_text(encoding="utf-8")
        m = re.search(r"names:\s*(\[.*?\])", txt, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    cj = real_root/"classes.json"
    if cj.exists():
        mp = json.loads(cj.read_text(encoding="utf-8"))
        keys = sorted(map(int, mp.keys()))
        return [mp[str(k)] if str(k) in mp else mp[k] for k in keys]
    raise RuntimeError("Cannot determine class names from real dataset.")

def main():
    from ultralytics import YOLO
    names = load_names(REAL_ROOT)
    gen = ensure_import(GEN_SCRIPT)

    splits = ["train", "val", "test"]
    real_lists = {sp: list_pairs(REAL_ROOT, sp) for sp in splits}
    for sp in splits:
        if not real_lists[sp]:
            raise RuntimeError(f"No real data found in {REAL_ROOT}/images/{sp}")

    model = YOLO(MODEL_WEIGHTS)
    workdir = Path("epoch_work"); workdir.mkdir(exist_ok=True)

    for ep in range(EPOCHS):
        print(f"\n========== Epoch {ep+1}/{EPOCHS} ==========")
        ep_out = OUT_BASE / f"ep_{ep:03d}"

        mix = {}
        for sp in splits:
            R = len(real_lists[sp])
            S = synth_needed(R)
            print(f"[{sp}] real={R} -> synth={S}")
            cfg = gen.GenConfig(
                assets_dir=str(ASSETS_DIR),
                out_dir=str(ep_out),
                image_size=IMAGE_SIZE,
                train_count=S if sp=='train' else 0,
                val_count=S if sp=='val' else 0,
                test_count=S if sp=='test' else 0,
                min_objects_per_image=MIN_OBJS,
                max_objects_per_image=MAX_OBJS,
                class_ratios=CLASS_RATIOS,
                per_class_min_max=PER_CLASS_MINMAX,
                allow_overlap=ALLOW_OVERLAP,
                yaml_abs=True,
                seed=ep + 2025,   # change seed each epoch
            )
            gen.generate_dataset(cfg)
            synth_dir = ep_out/"images"/sp
            synth_imgs = sorted(synth_dir.glob("*.*")) if synth_dir.exists() else []
            mixed = list(real_lists[sp]) + synth_imgs
            mix[sp] = mixed

        # Write file lists & yaml
        tl = workdir/f"train_ep{ep:03d}.txt"
        vl = workdir/f"val_ep{ep:03d}.txt"
        te = workdir/f"test_ep{ep:03d}.txt"
        write_list(mix["train"], tl); write_list(mix["val"], vl); write_list(mix["test"], te)

        yaml = workdir/f"dataset_ep{ep:03d}.yaml"
        yaml.write_text(
            "train: "+str(tl.resolve()).replace("\\","/")+"\n"+
            "val: "+str(vl.resolve()).replace("\\","/")+"\n"+
            "test: "+str(te.resolve()).replace("\\","/")+"\n"+
            f"nc: {len(names)}\n"+
            f"names: {json.dumps(names, ensure_ascii=False)}\n",
            encoding="utf-8"
        )

        model.train(
            data=str(yaml),
            epochs=1,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            resume=True if ep>0 else False,
            verbose=True,
        )

    print("Done. Check runs/detect/train for weights and metrics.")

if __name__=="__main__":
    main()
