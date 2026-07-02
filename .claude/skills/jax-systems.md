# Skill: JAX/XLA Systems

Use this skill when touching model code, training steps, inference, sharding, compilation, or performance timing.

## JAX Discipline

- Keep transformed functions pure.
- Pass PRNG keys explicitly.
- Treat shapes and static arguments as part of the compilation contract.
- Avoid Python objects as jitted inputs unless intentionally static.
- Avoid accidental host-device synchronization in training loops.
- Use `jax.Array` and current sharding APIs.

## Compilation Notes

Document or preserve:

- Which functions are jitted.
- Which arguments are static.
- What causes recompilation.
- Why fixed batch/sequence/cache shapes matter.
- When `block_until_ready()` is required.

## Training Invariants

Maintain tests for:

- Finite loss.
- Parameters change after an update.
- Gradient accumulation approximates the equivalent large batch update.
- Loss normalization is correct across microbatches.
- Dropout is disabled during evaluation.

## Inference Invariants

Maintain tests for:

- Future tokens do not affect earlier logits.
- KV-cached decode matches full-prefix decode within tolerance.
- Cache positions advance correctly.
- Overlong prompts fail with actionable errors.
- Greedy decoding is deterministic.

## Sharding Rules

- Validate mesh dimensions against available devices.
- Place batches and state explicitly.
- Log local/global devices and process count.
- If only one device exists, keep the single-device fallback correct.
- If CPU device simulation is documented, label it as simulation only.

