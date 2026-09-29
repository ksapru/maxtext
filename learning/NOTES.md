# MaxText learning notes

Tiny model on a fake 8-device CPU mesh (`run_tiny_cpu.sh`):
D=64, F=128, 4 heads x 16, 2 layers, vocab 256, 128 tokens, synthetic data, 5 steps.
About 115k parameters, 1,024 tokens per step.

## Week 1: setup

- Repo in `~/code/maxtext`, not `~/Desktop` (iCloud evicts files and breaks the venv).
- Venv: uv-managed arm64 Python 3.12, decoupled requirements minus the
  `cuda-bindings`, `cuda-pathfinder` and `nvidia-*` lines, then `uv pip install --no-deps -e .`
- `hardware=cpu` needs `skip_jax_distributed_system=true`.
- Loss 5.987 -> 5.976 over 5 steps. Random tokens, so it stays near ln(256) = 5.55.
- Log line to remember: `85.71% learnable weight flops and 14.29% attention flops`
  (attention scores are halved by the causal mask).

## Week 2: sharding

Dump the compiled HLO:

    XLA_FLAGS="--xla_force_host_platform_device_count=8 --xla_dump_to=/tmp/hlo_<name>" \
      ./run_tiny_cpu.sh <mesh flags>

Count collectives in `*jit_train_step.cpu_after_optimizations.txt`:

    grep -cE ' all-gather(-start)?\(' <file>

| Layout | Mesh flags | all-gather | all-reduce | all-to-all | collective-permute | Step time (steps 2-4) |
|---|---|---|---|---|---|---|
| dp=8 | `ici_data_parallelism=8 ici_fsdp_parallelism=1` | 0 | 3 | 0 | 0 | ~0.010 s |
| fsdp=8 | default | 16 | 5 | 2 | 0 | ~0.015 s |
| fsdp=4, tp=2 | `ici_fsdp_parallelism=4 ici_tensor_parallelism=2` | 33 | 27 | 1 | 2 | ~0.036 s |

Loss was identical (5.987 -> 5.976) in all three.

- dp: full model on every chip; the only big call is one all-reduce of all gradients.
- fsdp: weights stored in 1/8 slices; all-gather weight tables before use
  (shapes like `[64,128]`, `[64,4,16]`), all-reduce gradients. No reduce-scatter
  showed up on the CPU build.
- tp=2 adds all-reduces on activations, shape `[2,128,64]`
  (2 sequences per FSDP group x 128 tokens x 64), inside every layer.
- For this tiny model compute is ~0, so the layout with the fewest collectives wins.
  Fake CPU devices share memory, so timings mostly measure overhead.
  Big models can't use dp (model doesn't fit per chip), so they use fsdp + a little tp.

Fusion example (RMSNorm scale): before optimization it is 5 separate ops
(square, reduce_sum, divide by 64, add 1e-5, rsqrt). After, it is 2 fusions:
square+sum in one, `* 0.015625` + add + rsqrt in the other, and the squared
values are never written to memory. `divide` count went 80 -> 50 overall.

## Next: Week 3

Real text (`dataset_type=hf`, tokenizer), train long enough for loss to drop
well below 5.55, checkpoint and resume.
