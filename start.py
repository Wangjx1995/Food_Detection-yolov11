import os, subprocess
from pathlib import Path

DRIVE_MOUNT = "/content/drive"
REPO_URL    = "https://github.com/Wangjx1995/Object_Detection_Tutorial.git"
REPO_DIR    = "/content/Object_Detection_Tutorial"


DATASET_YAML = "/content/Object_Detection_Tutorial/configs/dataset.yaml"

TRAIN_CMD    = f"python src/train.py --data '{DATASET_YAML}'"

def run(cmd, check=True, cwd=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)


Path("/content").mkdir(exist_ok=True)
os.chdir("/content")


try:
    from google.colab import drive
    drive.mount(DRIVE_MOUNT, force_remount=False)
except Exception:
    print("ℹ️ 若在子进程中运行：请先在单元格执行 drive.mount('/content/drive')")


run(f"rm -rf '{REPO_DIR}'", check=False)


run(f"git clone -vv {REPO_URL} '{REPO_DIR}'")


run("python -m pip install -U pip")
run(f"python -m pip install -r '{REPO_DIR}/requirements.txt'")


run(TRAIN_CMD, cwd=REPO_DIR)

print("\n✅ All done.")