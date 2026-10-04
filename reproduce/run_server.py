#!/usr/bin/env python3
"""Schedule Louver paper experiments onto idle L40 GPUs.

Sample sizes follow benchmark_area/experiments/accuracy/run_all.sh
(the commands the authors marked as the run), with AIME expanded to the
full 30-problem 2024 set.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/dhy/PhD/Louver")
W = Path("/home/dhy/PhD/weights/Louver")
LOG = REPO / "reproduce" / "logs"
RES = REPO / "reproduce" / "results"
PY = "/home/dhy/anaconda3/envs/louver/bin/python"
ACC = REPO / "benchmark_area" / "experiments" / "accuracy"
LAT = REPO / "benchmark_area" / "experiments" / "latency"
FN = REPO / "benchmark_area" / "experiments" / "false_negativity"

LB_TASKS = "narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique"
RULER_TASKS = "niah_single,niah_multi,vt"
OFF_TASKS = "hotpotqa,2wikimqa,musique,qasper,narrativeqa"

ENV_HEADER = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/dhy/PhD/weights/Louver/hf-home
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
"""


def model_path(name: str) -> str:
    return str(W / name)


def sh_job(name: str, body: str, needs: list[str], needs_jobs: list[str] | None = None,
           group: str | None = None, gpu: bool = True) -> dict:
    return {
        "name": name,
        "body": ENV_HEADER + "\n" + body + "\necho JOB_OK\n",
        "needs": needs,
        "needs_jobs": needs_jobs or [],
        "group": group,
        "gpu": gpu,
    }


def fn_job(model: str) -> dict:
    tag = model.replace(".", "_")
    m = model_path(model)
    body = f"""
cd {FN}
M={m}
TAG={tag}
run_exp() {{
  local script="$1" out="$2"
  if [ -s "reports/${{out}}__${{TAG}}.csv" ]; then
    echo "skip $out"
    return
  fi
  {PY} "$script" --model "$M"
  cp -f "reports/${{out}}.csv" "reports/${{out}}__${{TAG}}.csv"
}}
run_exp exp1_relevant_vs_irrelevant.py exp1_relevant_vs_irrelevant
run_exp exp2_fixed_k.py exp2_fixed_k
run_exp exp3_variable_list.py exp3_variable_list
run_exp exp4_fixed_sum_threshold.py exp4_fixed_sum_threshold
"""
    return sh_job(f"fn_{tag}", body, [model], group="fn")


def longbench_job(method: str, kind: str) -> dict:
    m = model_path("Llama-3.1-8B-Instruct")
    out = RES / "longbench"
    if kind == "dense":
        cmd = f'{PY} eval/longbench.py --model "{m}" --method dense_sdpa --output_dir "{out}" --max_input_length 32768 --tasks {LB_TASKS} --max_samples 100'
    elif kind == "louver":
        cmd = (
            f'{PY} eval/longbench.py --model "{m}" --method louver_ta --threshold_mode budget '
            f'--budget_fraction 0.1 --sample_size 512 --output_dir "{out}" '
            f'--max_input_length 32768 --tasks {LB_TASKS} --max_samples 100'
        )
    else:
        extra = "--top_p 0.85" if method == "twilight" else "--budget_fraction 0.1"
        cmd = (
            f'{PY} eval/longbench_baselines.py --model "{m}" --method {method} {extra} '
            f'--output_dir "{out}" --max_input_length 32768 --tasks {LB_TASKS} --max_samples 100'
        )
    body = f"cd {ACC}\nmkdir -p {out}\n{cmd}\n"
    return sh_job(f"lb_{method}", body, ["Llama-3.1-8B-Instruct", "DATASETS"])


def ruler_job(method: str, kind: str) -> dict:
    m = model_path("Llama-3.1-8B-Instruct")
    out = RES / "ruler"
    if kind == "dense":
        extra = "--method dense_sdpa"
    elif kind == "louver":
        extra = "--method louver_ta --threshold_mode budget --budget_fraction 0.1 --sample_size 512"
    elif method == "twilight":
        extra = "--method twilight --top_p 0.85"
    else:
        extra = f"--method {method} --budget_fraction 0.1"
    cmd = (
        f'{PY} eval/ruler.py --model "{m}" {extra} --output_dir "{out}" '
        f'--tasks {RULER_TASKS} --seq_len 32768 --n_samples 50'
    )
    body = f"cd {ACC}\nmkdir -p {out}\n{cmd}\n"
    return sh_job(f"ruler_{method}", body, ["Llama-3.1-8B-Instruct", "DATASETS"])


def aime_job(method: str, kind: str) -> dict:
    m = model_path("DeepSeek-R1-Distill-Llama-8B")
    out = RES / "aime"
    if kind == "dense":
        extra = "--method dense_sdpa"
    elif kind == "louver":
        extra = "--method louver_ta --threshold_mode budget --budget_fraction 0.15 --sample_size 512"
    elif method == "twilight":
        extra = "--method twilight --top_p 0.85"
    else:
        extra = "--method h2o --budget_tokens 512"
    cmd = f'{PY} eval/aime.py --model "{m}" {extra} --output_dir "{out}" --max_problems 30 --max_new_tokens 8192'
    body = f"cd {ACC}\nmkdir -p {out}\n{cmd}\n"
    return sh_job(f"aime_{method}", body, ["DeepSeek-R1-Distill-Llama-8B", "DATASETS"])


def math_job(method: str, kind: str) -> dict:
    m = model_path("DeepSeek-R1-Distill-Llama-8B")
    out = RES / "math500"
    if kind == "dense":
        extra = "--method dense_sdpa"
    elif kind == "louver":
        extra = "--method louver_ta --threshold_mode budget --budget_fraction 0.15 --sample_size 512"
    elif method == "twilight":
        extra = "--method twilight --top_p 0.85"
    else:
        extra = "--method h2o --budget_tokens 512"
    # run_all.sh uses 50; full MATH-500 x 4 methods is multi-day on one GPU.
    cmd = f'{PY} eval/math500.py --model "{m}" {extra} --output_dir "{out}" --max_samples 50 --max_new_tokens 4096'
    body = f"cd {ACC}\nmkdir -p {out}\n{cmd}\n"
    return sh_job(f"math_{method}", body, ["DeepSeek-R1-Distill-Llama-8B", "DATASETS"])


def offload_job(method: str) -> dict:
    script = REPO / "benchmark_area" / "experiments" / "offload" / "run_longbench_offload.py"
    m = model_path("Llama-3.1-8B-Instruct")
    out = RES / "offload"
    cmd = (
        f'{PY} {script} --model "{m}" --method {method} --tasks {OFF_TASKS} '
        f'--max_samples 10 --max_input_length 4096 --output_dir "{out}"'
    )
    body = f"mkdir -p {out}\n{cmd}\n"
    return sh_job(f"offload_{method}", body, ["Llama-3.1-8B-Instruct", "DATASETS"])


def latency_job(model: str, layer: int, tag: str) -> dict:
    m = model_path(model)
    cap = LAT / "captures"
    pt = cap / f"{tag}_layer{layer}.pt"
    body = f"""
cd {LAT}
mkdir -p {cap} {LAT / "reports"} {RES}
PT="{pt}"
if [ ! -s "$PT" ]; then
  {PY} capture_aime.py --model "{m}" --layer {layer} --max-tokens 40000 --problem-idx 0 --output-dir "{cap}"
  FOUND=$(ls -1t {cap}/*layer{layer}_N*.pt | head -1)
  cp -f "$FOUND" "$PT"
fi
echo "CAPTURE $PT"
{PY} gpu_bench.py --input-qkv "$PT" --output-csv "{LAT / "reports" / f"gpu_bench_{tag}.csv"}"
{PY} cpu_bench.py --input-qkv "$PT" --output-csv "{LAT / "reports" / f"cpu_bench_{tag}.csv"}"
echo "$PT" > {RES}/capture_{tag}.txt
"""
    return sh_job(f"lat_{tag}", body, [model])


def recall_job() -> dict:
    script = REPO / "benchmark_area" / "experiments" / "recall" / "recall_bench.py"
    body = f"""
PT1=$(cat {RES}/capture_llama3b.txt)
PT2=$(cat {RES}/capture_qwen7b.txt)
PT3=$(cat {RES}/capture_qwen14b.txt)
{PY} {script} --n-samples 100 --max-n 8000 --seed 42 --input-qkv "$PT1" "$PT2" "$PT3"
"""
    return sh_job("recall", body, [], needs_jobs=["lat_llama3b", "lat_qwen7b", "lat_qwen14b"])


def threshold_job() -> dict:
    d = REPO / "benchmark_area" / "experiments" / "threshold_oracle"
    body = f"""
{PY} {d}/bench.py --captures-dir {LAT}/captures --n-steps 200 --n-steps-ts 2000 --sample-size 256 --top-frac 0.10 --output-dir {d}/results
{PY} {d}/plot_oracle_oscillation.py --results-dir {d}/results --out-dir {d}/results/figs --smooth 7 || true
"""
    return sh_job("threshold_oracle", body, [], needs_jobs=["lat_llama3b", "lat_qwen7b", "lat_qwen14b"])


def pruning_job() -> dict:
    qp = REPO / "benchmark_area" / "quick_pruning"
    bench = REPO / "benchmark_area" / "experiments" / "pruning" / "bench.py"
    cap = qp / "capture_qkv_8000_Llama-3.2-3B-Instruct.pt"
    m = model_path("Llama-3.2-3B-Instruct")
    out = REPO / "benchmark_area" / "experiments" / "pruning" / "results"
    loops = "\n".join(
        f'{PY} {bench} --input-qkv "{cap}" --mode louver --strategy {s} '
        f'--S-values 2,4,8,16 --r-values 2,4,8,16 --n-queries 30 --topk 20 --output-dir "{out}"'
        for s in ("contiguous", "interleaved", "random", "pca")
    )
    body = f"""
cd {qp}
mkdir -p {out}
{PY} capture.py --model "{m}" --n-tokens 8000 --output-qkv "{cap}"
{loops}
{PY} {bench} --input-qkv "{cap}" --mode standard \
  --clusterings kcenter,kmeans,pq_subspace,batch_nn \
  --enclosings ball_centroid,aabb,span_ball \
  --r-values 4,8,16 --n-queries 30 --topk 20 --output-dir "{out}"
"""
    return sh_job("pruning", body, ["Llama-3.2-3B-Instruct"])


def key_norms_job() -> dict:
    d = REPO / "benchmark_area" / "experiments" / "key_norms"
    m = model_path("Qwen2.5-7B-Instruct")
    snap = d / "snapshots" / "snap_repro.pt"
    body = f"""
cd {d}
mkdir -p snapshots reports
{PY} capture.py --model "{m}" --max_new_tokens 128 --out "{snap}"
{PY} analyze.py --snap "{snap}" --out_dir "{d}"
"""
    return sh_job("key_norms", body, ["Qwen2.5-7B-Instruct"])


def score_job(model: str, tokens: int, tag: str) -> dict:
    d = REPO / "benchmark_area" / "experiments" / "score_distribution"
    m = model_path(model)
    snap = d / "snapshots" / f"snap_{tag}.pt"
    metrics = d / "reports" / f"tail_metrics_{tag}.csv"
    body = f"""
cd {d}
mkdir -p snapshots reports figs
{PY} capture.py --model "{m}" --max_new_tokens {tokens} --out "{snap}"
{PY} tail_metrics.py --snap "{snap}" --out "{metrics}"
{PY} analyze.py --snap "{snap}" --metrics "{metrics}" --out_dir "{d}/figs/{tag}" --report "{d}/reports/windows_{tag}.md" || true
"""
    return sh_job(f"score_{tag}", body, [model])


def build_jobs() -> list[dict]:
    jobs = [
        fn_job("Llama-3.2-1B-Instruct"),
        fn_job("Qwen2.5-3B-Instruct"),
        fn_job("Llama-3.2-3B-Instruct"),
        fn_job("Qwen2.5-7B-Instruct"),
        fn_job("Llama-3.1-8B-Instruct"),
        key_norms_job(),
        pruning_job(),
        latency_job("Llama-3.2-3B-Instruct", 14, "llama3b"),
        latency_job("Qwen2.5-7B-Instruct", 14, "qwen7b"),
        latency_job("Qwen2.5-14B-Instruct", 24, "qwen14b"),
        recall_job(),
        threshold_job(),
        score_job("Qwen2.5-7B-Instruct", 2000, "qwen7b"),
        score_job("DeepSeek-R1-Distill-Qwen-14B", 2000, "dsr14b"),
    ]
    for method, kind in [
        ("dense_sdpa", "dense"),
        ("louver_ta", "louver"),
        ("h2o", "base"),
        ("quest", "base"),
        ("streaming_llm", "base"),
        ("twilight", "base"),
    ]:
        jobs.append(longbench_job(method, kind))
        jobs.append(ruler_job(method, kind))
    for method, kind in [
        ("dense_sdpa", "dense"),
        ("louver_ta", "louver"),
        ("h2o", "base"),
        ("twilight", "base"),
    ]:
        jobs.append(aime_job(method, kind))
        jobs.append(math_job(method, kind))
    for method in ("louver_offload", "hnsw_offload", "ivf_offload", "lsh_offload"):
        jobs.append(offload_job(method))
    return jobs


def ready(name: str) -> bool:
    if name == "DATASETS":
        return (W / "DATASETS_READY").exists()
    return (W / name / "READY").exists()


def gpu_stats() -> list[tuple[int, int, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, mem, util = [int(x.strip()) for x in line.split(",")]
        rows.append((idx, mem, util))
    return rows


def pick_gpu(busy: set[int]) -> int | None:
    free = [(mem, util, idx) for idx, mem, util in gpu_stats() if idx not in busy and mem < 1500]
    if not free:
        return None
    free.sort()
    return free[0][2]


def done_marker(name: str) -> Path:
    return LOG / f"{name}.ok"


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    running: dict[str, tuple[subprocess.Popen, int | None]] = {}
    busy_gpu: set[int] = set()
    cpu_running = False

    print(f"queued {len(jobs)} jobs", flush=True)
    while True:
        finished = []
        for name, (proc, gpu) in running.items():
            ret = proc.poll()
            if ret is None:
                continue
            finished.append(name)
            if gpu is not None:
                busy_gpu.discard(gpu)
            else:
                cpu_running = False
            if ret == 0:
                done_marker(name).write_text("ok\n")
                print(f"OK {name}", flush=True)
            else:
                print(f"FAIL {name} exit={ret}", flush=True)
                (LOG / f"{name}.fail").write_text(str(ret))
        for name in finished:
            running.pop(name)

        launched = False
        for job in jobs:
            name = job["name"]
            if done_marker(name).exists() or (LOG / f"{name}.fail").exists() or name in running:
                continue
            if any(not ready(n) for n in job["needs"]):
                continue
            if any(not done_marker(n).exists() for n in job["needs_jobs"]):
                continue
            if job["group"] and any(j["group"] == job["group"] for j in jobs if j["name"] in running):
                continue
            gpu = None
            env = os.environ.copy()
            if job["gpu"]:
                gpu = pick_gpu(busy_gpu)
                if gpu is None:
                    continue
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            else:
                if cpu_running:
                    continue
                env["CUDA_VISIBLE_DEVICES"] = ""
            script = LOG / f"{name}.sh"
            script.write_text(job["body"])
            logf = open(LOG / f"{name}.log", "w")
            print(f"START {name} gpu={gpu}", flush=True)
            proc = subprocess.Popen(["bash", str(script)], stdout=logf, stderr=subprocess.STDOUT, env=env)
            running[name] = (proc, gpu)
            if gpu is not None:
                busy_gpu.add(gpu)
            else:
                cpu_running = True
            launched = True
        pending = [
            j["name"] for j in jobs
            if not done_marker(j["name"]).exists() and not (LOG / f"{j['name']}.fail").exists()
        ]
        if not pending and not running:
            print("ALL_JOBS_FINISHED", flush=True)
            return
        if not launched:
            time.sleep(20)


if __name__ == "__main__":
    main()
