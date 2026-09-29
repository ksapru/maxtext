#!/usr/bin/env bash
# Tiny MaxText training run on CPU with synthetic data and a fake 8-device mesh.
# Extra args are appended, e.g.: ./run_tiny_cpu.sh ici_fsdp_parallelism=4 ici_tensor_parallelism=2
# XLA_FLAGS from the caller is kept as-is (include the device count yourself if you set it).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

export XLA_FLAGS="${XLA_FLAGS:---xla_force_host_platform_device_count=8}"
export JAX_PLATFORMS=cpu
export DECOUPLE_GCLOUD=TRUE

python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  run_name=tiny_cpu \
  base_output_directory=/tmp/maxtext_out \
  hardware=cpu \
  skip_jax_distributed_system=true \
  dataset_type=synthetic \
  steps=5 \
  per_device_batch_size=1 \
  base_emb_dim=64 \
  base_num_decoder_layers=2 \
  base_num_query_heads=4 \
  base_num_kv_heads=4 \
  head_dim=16 \
  base_mlp_dim=128 \
  vocab_size=256 \
  max_target_length=128 \
  attention=dot_product \
  dtype=float32 \
  enable_checkpointing=false \
  enable_tensorboard=false \
  "$@"
