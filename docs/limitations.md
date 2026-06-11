# Limitations

An honest inventory. Items marked *(roadmap)* are candidate extensions; the
rest are deliberate scope decisions for an educational systems lab.

## Hardware and measurement

- **CPU only.** Development and all recorded results come from a single
  Apple-Silicon CPU. No GPU/TPU numbers exist in this repository; bf16 on
  CPU validates *correctness*, not speed (emulated arithmetic).
- **Multi-device = simulated.** Sharding logic is tested with
  `--xla_force_host_platform_device_count`; these devices share one CPU and
  prove placement/collective correctness, never scaling. Real
  multi-accelerator measurements are marked "unavailable" in
  [results.md](results.md).
- **Multi-host execution is documented, not tested** (single host here);
  `jax.distributed.initialize()` wiring is out of scope.
- Device memory statistics are unsupported on the CPU backend; the memory
  probe reports host RSS with provenance and says "unsupported" for the
  device ([memory.py](../src/jaxscale_lm/benchmark/memory.py)).

## Model and training

- Models are tiny (≈1.8M / ≈29M parameters) by design; text quality is a
  non-goal — the systems behavior (compilation, caching, sharding,
  checkpointing) is the deliverable.
- Tensor/model parallelism: the mesh reserves a `model` axis but no
  parameter sharding ships *(roadmap)*.
- No activation checkpointing / rematerialization *(roadmap)*.
- Data loading is synchronous on the host thread; at this scale it is
  nowhere near the bottleneck, so no prefetch pipeline.
- Exact data-order resumption replays the deterministic batch stream from
  step 0 to the resume point (index arithmetic only — cheap), rather than
  checkpointing loader state.

## Inference and serving

- **Batched generation assumes equal prompt lengths** per batch (the cache
  keeps one shared length scalar). The serving path uses batch size 1, so
  this only constrains programmatic use. Ragged batches / paged KV cache
  *(roadmap)*.
- **Prefill recompiles per new prompt length** (shape-keyed jit cache). The
  decode step compiles once (capacity pinned to the model context). Prompt
  bucketing would bound prefill compiles *(roadmap)*.
- **Requests are serialized**, one at a time, behind a lock. There is no
  dynamic or continuous batching — and the docs/API never claim otherwise
  *(roadmap)*.
- The EOS check costs one host-device sync per generated token; benchmark
  docs quantify the effect.
- The decode loop always attends over the full pinned capacity, so very
  short generations on a large-context model do more work than strictly
  necessary — the standard fixed-shape trade.
- The model registry is a local JSON file with atomic writes — inspired by
  managed-model lifecycles, not a multi-writer service; no auth, no
  concurrent-writer support.

## Benchmarks

- Tiny workloads over-weight fixed overheads; numbers demonstrate
  mechanisms (jit benefit, cache complexity, shape recompilation), not
  production throughput.
- No MFU reporting (no documented hardware peak on CPU; see
  [benchmarking.md](benchmarking.md)).

## Security

- The API has no authentication/authorization or rate limiting; it is a
  local lab service, not an internet-facing deployment.

## Roadmap (optional extensions, in rough order of value)

grouped-query attention (config plumbing exists: `num_key_value_heads`),
prompt-length bucketing, tensor parallelism on the `model` axis, activation
checkpointing, dynamic batching, paged KV cache, speculative decoding,
LoRA fine-tuning, TensorBoard logging, a Grafana dashboard JSON, multi-host
initialization, OpenAI-compatible endpoint.
