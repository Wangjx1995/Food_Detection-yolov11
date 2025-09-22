#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, subprocess, argparse
from pathlib import Path

DRIVE_MOUNT = "/content/drive"
REPO_URL    = "https://github.com/Wangjx1995/Object_Detection_Tutorial.git"
REPO_DIR    = "/content/Object_Detection_Tutorial"

# ===== CLI =====
p = argparse.ArgumentParser(description="Colab starter for Object_Detection_Tutorial")
p.add_argument("--mode", choices=["real", "mix"], default="real",
               help="训练模式：real=全真实数据；mix=真实+生成数据（调用 train_mix.py）")
p.add_argument("--dataset_yaml", default="/content/Object_Detection_Tutorial/configs/dataset.yaml",
               help="仅在 --mode real 时使用：真实数据集的 dataset.yaml 路径")
p.add_argument("--skip_drive", action="store_true",
               help="不挂载 Google Drive（在纯本地/容器环境下用）")
p.add_argument("--branch", default=None, help="可选：git 指定分支")
p.add_argument("--no_requirements", action="store_true",
               help="跳过安装 requirements.txt")
args = p.parse_args()

def run(cmd, check=True, cwd=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)


Path("/content").mkdir(exist_ok=True)
os.chdir("/content")


if not args.skip_drive:
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT, force_remount=False)
    except Exception:
        print("ℹ️ 若在子进程中运行：请先在单元格执行 drive.mount('/content/drive')")


run(f"rm -rf '{REPO_DIR}'", check=False)
clone_cmd = f"git clone -vv {REPO_URL} '{REPO_DIR}'"
if args.branch:
    clone_cmd = f"git clone -vv --branch {args.branch} {REPO_URL} '{REPO_DIR}'"
run(clone_cmd)


run("python -m pip install -U pip")
if not args.no_requirements:
    run(f"python -m pip install -r '{REPO_DIR}/requirements.txt'")

run("python -m pip install -U ultralytics pillow numpy", check=False)


if args.mode == "real":
    DATASET_YAML = args.dataset_yaml
    TRAIN_CMD    = f"python src/train.py --data '{DATASET_YAML}'"
    run(TRAIN_CMD, cwd=REPO_DIR)

elif args.mode == "mix":
    MIX_CMD = "python train_mix.py"
    run(MIX_CMD, cwd=REPO_DIR)

print("\n✅ All done.")
