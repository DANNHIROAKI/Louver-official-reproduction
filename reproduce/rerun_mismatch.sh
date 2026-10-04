#!/usr/bin/env bash
# Re-run the reproduction pieces that disagreed with the paper protocol.
# One argument: gpu0 | gpu1 | gpu2 | gpu3 | cpu
set -euo pipefail

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=8.9
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/dhy/PhD/weights/Louver/hf-home
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/dhy/anaconda3/envs/louver/bin/python
REPO=/home/dhy/PhD/Louver
ACC="${REPO}/benchmark_area/experiments/accuracy"
LLAMA8=/home/dhy/PhD/weights/Louver/Llama-3.1-8B-Instruct
DS=/home/dhy/PhD/weights/Louver/DeepSeek-R1-Distill-Llama-8B
RULER_OUT="${REPO}/reproduce/results/ruler_bf16"
MATH_OUT="${REPO}/reproduce/results/math500_full"
AIME_OUT="${REPO}/reproduce/results/aime_32k"
CAP="${REPO}/benchmark_area/quick_pruning/capture_qkv_8000_Llama-3.2-3B-Instruct.pt"
BENCH="${REPO}/benchmark_area/experiments/pruning/bench.py"
POUT="${REPO}/benchmark_area/experiments/pruning/results"
LAT="${REPO}/benchmark_area/experiments/latency"

mkdir -p "${RULER_OUT}" "${MATH_OUT}" "${AIME_OUT}" "${POUT}" "${LAT}/reports"

ruler() {
  local method="$1"; shift
  echo "=== RULER bf16 ${method} $(date -Is) ==="
  cd "${ACC}"
  "${PY}" eval/ruler.py --model "${LLAMA8}" --method "${method}" "$@" \
    --output_dir "${RULER_OUT}" --tasks vt --seq_len 32768 --n_samples 50 --seed 42
}

math() {
  local method="$1"; shift
  echo "=== MATH-500 full ${method} $(date -Is) ==="
  cd "${ACC}"
  "${PY}" eval/math500.py --model "${DS}" --method "${method}" "$@" \
    --output_dir "${MATH_OUT}" --max_samples 500 --max_new_tokens 4096
}

aime() {
  local method="$1"; shift
  echo "=== AIME 32768 ${method} $(date -Is) ==="
  cd "${ACC}"
  "${PY}" eval/aime.py --model "${DS}" --method "${method}" "$@" \
    --output_dir "${AIME_OUT}" --max_problems 30 --max_new_tokens 32768
}

prune() {
  echo "=== pruning Table 7 protocol N=4500 layer 15 r=4 $(date -Is) ==="
  cd "${REPO}"
  for strategy in contiguous interleaved random pca; do
    "${PY}" "${BENCH}" \
      --input-qkv "${CAP}" \
      --mode louver \
      --strategy "${strategy}" \
      --S-values 2,4,8,16 \
      --r-values 4 \
      --layer 15 \
      --max-n 4500 \
      --n-queries 30 \
      --topk 20 \
      --output-dir "${POUT}"
  done
}

cpu_probe() {
  echo "=== CPU latency probe 64 threads near 40k $(date -Is) ==="
  cd "${LAT}"
  for spec in "llama3b llama3b_layer14.pt" "qwen7b qwen7b_layer14.pt" "qwen14b qwen14b_layer24.pt"; do
    set -- ${spec}
    tag="$1"; pt="$2"
    "${PY}" cpu_bench.py \
      --input-qkv "${LAT}/captures/${pt}" \
      --prefill-frac 0.995 \
      --n-steps 8 \
      --num-threads 64 \
      --output-csv "${LAT}/reports/cpu_probe64_${tag}.csv"
  done
}

case "${1:-}" in
  gpu0)
    export CUDA_VISIBLE_DEVICES=0
    prune
    ruler quest --budget_fraction 0.1
    math dense_sdpa
    aime dense_sdpa
    ;;
  gpu1)
    export CUDA_VISIBLE_DEVICES=1
    ruler dense_sdpa
    math louver_ta --threshold_mode budget --budget_fraction 0.15 --sample_size 512
    aime louver_ta --threshold_mode budget --budget_fraction 0.15 --sample_size 512
    ;;
  gpu2)
    export CUDA_VISIBLE_DEVICES=2
    ruler h2o --budget_fraction 0.1
    math h2o --budget_tokens 512
    aime twilight --top_p 0.85
    ;;
  gpu3)
    export CUDA_VISIBLE_DEVICES=3
    ruler louver_ta --threshold_mode budget --budget_fraction 0.1 --sample_size 512
    math twilight --top_p 0.85
    ;;
  cpu)
    cpu_probe
    ;;
  *)
    echo "usage: $0 gpu0|gpu1|gpu2|gpu3|cpu" >&2
    exit 2
    ;;
esac
echo "=== DONE ${1} $(date -Is) ==="
