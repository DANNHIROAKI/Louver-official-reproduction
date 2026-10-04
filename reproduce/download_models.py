#!/usr/bin/env python3
"""Download Louver paper weights (safetensors only) from ModelScope via aria2."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path("/home/dhy/PhD/weights/Louver")
ROOT.mkdir(parents=True, exist_ok=True)

# (modelscope id, local directory name)
MODELS = [
    ("LLM-Research/Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct"),
    ("LLM-Research/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    ("LLM-Research/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B"),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-14B"),
]

SKIP_NAMES = {
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "USE_POLICY.md",
    ".gitattributes",
    "configuration.json",
}


def list_files(model_id: str) -> list[dict]:
    url = (
        "https://www.modelscope.cn/api/v1/models/"
        f"{model_id}/repo/files?Revision=master&Recursive=true"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    return data["Data"]["Files"]


def keep(entry: dict) -> bool:
    path = entry["Path"]
    name = path.split("/")[-1]
    if entry.get("Type") != "blob":
        return False
    if path == "original" or path.startswith("original/") or name in SKIP_NAMES:
        return False
    if name.endswith((".pth", ".bin", ".msgpack", ".ot", ".h5")):
        return False
    return True


def download_file(model_id: str, dest_dir: Path, path: str, size: int) -> None:
    dest = dest_dir / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  skip {path}", flush=True)
        return
    url = f"https://www.modelscope.cn/models/{model_id}/resolve/master/{path}"
    print(f"  get {path} ({size / 1e9:.2f} GB)", flush=True)
    cmd = [
        "aria2c",
        "-c",
        "-x", "16",
        "-s", "16",
        "-k", "8M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--max-tries=30",
        "--retry-wait=5",
        "--timeout=120",
        "--connect-timeout=30",
        "--summary-interval=30",
        "-d", str(dest.parent),
        "-o", dest.name,
        url,
    ]
    subprocess.run(cmd, check=True)
    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"size mismatch {path}: got {got} expected {size}")


def update_weights_md() -> None:
    md = Path("/home/dhy/Desktop/权重存放路径.md")
    text = md.read_text() if md.exists() else "# 权重存放路径\n"
    marker = "\n## Louver\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    lines = [
        "",
        "## Louver",
        "",
        "机械盘 `/hdd`（`/dev/sda`）当前账号不能新建目录，权重放在根分区。",
        "来源：ModelScope（Llama 在 Hugging Face 上是 gated，本机 token 无权限）。只下载 safetensors 与 tokenizer，不下载 `original/*.pth`。",
        "",
        "| 模型 | 路径 | 说明 |",
        "|---|---|---|",
    ]
    for _mid, name in MODELS:
        dest = ROOT / name
        state = "已完成" if (dest / "READY").exists() else "未完成"
        lines.append(f"| {name} | `{dest}` | Louver 复现，{state} |")
    lines.append("")
    md.write_text(text.rstrip() + "\n" + "\n".join(lines))


def download_model(model_id: str, local_name: str) -> None:
    dest = ROOT / local_name
    ready = dest / "READY"
    if ready.exists():
        print(f"READY {local_name}", flush=True)
        update_weights_md()
        return
    print(f"START {model_id}", flush=True)
    files = [f for f in list_files(model_id) if keep(f)]
    for f in files:
        download_file(model_id, dest, f["Path"], int(f["Size"]))
    ready.write_text(model_id + "\n")
    update_weights_md()
    print(f"DONE {local_name}", flush=True)


def download_datasets() -> None:
    marker = ROOT / "DATASETS_READY"
    if marker.exists():
        print("DATASETS_READY", flush=True)
        return
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault(
        "HF_HOME",
        "/home/dhy/PhD/weights/Louver/hf-home",
    )
    from datasets import load_dataset

    print("START datasets", flush=True)
    for task in [
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
    ]:
        print(f"  LongBench {task}", flush=True)
        load_dataset("THUDM/LongBench", task, split="test")
    print("  AIME 2024", flush=True)
    load_dataset("Maxwell-Jia/AIME_2024", split="train")
    print("  MATH-500", flush=True)
    load_dataset("HuggingFaceH4/MATH-500", split="test")
    marker.write_text("ok\n")
    print("DONE datasets", flush=True)


def main() -> None:
    for item in MODELS:
        download_model(*item)
    download_datasets()
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
