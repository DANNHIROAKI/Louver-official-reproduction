"""Load paper benchmarks from local files when the Hub is unreachable."""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

ROOT = Path("/home/dhy/PhD/weights/Louver/datasets")


def load_longbench(task: str):
    path = ROOT / "longbench" / "data" / f"{task}.jsonl"
    if path.exists():
        return load_dataset("json", data_files=str(path), split="train")
    return load_dataset("THUDM/LongBench", task, split="test")


def load_math500():
    path = ROOT / "MATH-500" / "test.jsonl"
    if path.exists():
        return load_dataset("json", data_files=str(path), split="train")
    return load_dataset("HuggingFaceH4/MATH-500", split="test")


def load_aime():
    path = ROOT / "AIME_2024" / "aime_2024_problems.parquet"
    if path.exists():
        return load_dataset("parquet", data_files=str(path), split="train")
    return load_dataset("Maxwell-Jia/AIME_2024", split="train")
