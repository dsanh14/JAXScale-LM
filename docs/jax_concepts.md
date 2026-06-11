# JAX Concepts Used in This Project

A practical tour of the JAX machinery JAXScale-LM is built on, with pointers
to where each concept lives in the code.

## Tracing and `jax.jit`

`jax.jit` does not run your Python function on arrays. It runs it once with
*tracers* — abstract values carrying only shape and dtype — records every
JAX operation into a graph (jaxpr), hands the graph to XLA for compilation,
and caches the compiled executable. Subsequent calls with the same
shapes/dtypes skip Python almost entirely.

Consequences that shape this codebase:

- **Python control flow runs at trace time.** `if cache is None:` in
  [attention.py](../src/jaxscale_lm/model/attention.py) selects between the
  training and cached paths *during tracing*; each variant is its own
  compiled program. Data-dependent branching must use `jnp.where` /
  `jax.lax.cond` instead.
- **Side effects don't survive tracing.** Printing, logging, mutating
  Python state inside a jitted function happens once (at trace time) or
  never. All logging and I/O in this project lives outside the jitted
  functions ([trainer.py](../src/jaxscale_lm/training/trainer.py) is the
  host-side orchestrator).
- **Jitted functions here are closed over static values only** (graphdef,
  optimizer, accumulation count) and take everything dynamic as arguments —
  see `make_train_step` in [step.py](../src/jaxscale_lm/training/step.py).

## Static vs dynamic values

Anything that changes a compiled program's *structure* must be static:
shapes, dtypes, Python ints/bools used in control flow, the NNX graph
definition. Anything that's data flows through as a traced array.
`SamplingParams` ([sampling.py](../src/jaxscale_lm/inference/sampling.py))
is a frozen, hashable dataclass precisely so jit can treat it as static;
changing `top_k` recompiles, changing the PRNG key does not.

## Recompilation

A jit cache miss (= recompilation) happens when the *signature* changes:

- different array shapes (a new batch size or sequence length),
- different dtypes,
- different static argument values,
- different pytree structure (e.g. an optimizer state with a new field).

Recompiles are the dominant silent performance killer in JAX programs. This
project counters them three ways:

1. **Fixed shapes by construction**: packed training batches are always
   `[accum, micro, seq]`; the KV cache has a pinned capacity so every decode
   step is one shape ([engine.py](../src/jaxscale_lm/inference/engine.py)).
2. **A compile counter**: `compilation_count` in
   [timing.py](../src/jaxscale_lm/utils/timing.py) listens to JAX's
   monitoring event for backend compilations; the trainer and benchmarks
   warn when the count grows unexpectedly.
3. **Explicit measurement**: the compilation benchmark suite times
   first-call vs steady-state and demonstrates which shape changes
   recompile ([compilation.py](../src/jaxscale_lm/benchmark/compilation.py)).

## Pytrees

A pytree is any nested structure of containers (tuples, lists, dicts,
NamedTuples, registered dataclasses) with arrays at the leaves. JAX
transformations operate leaf-wise. Everything stateful here is a pytree:
`TrainState` (NamedTuple), the NNX parameter `State`, the `KVCache`
(NamedTuple of NamedTuples), and batches. That is what lets one
`jax.device_put` call shard a whole training state, and lets Orbax
checkpoint it without custom serializers.

## Asynchronous dispatch and `block_until_ready`

JAX returns control to Python as soon as work is *enqueued* on the device,
not when it finishes. `result = jitted_fn(x)` followed by
`time.perf_counter()` measures dispatch latency — typically microseconds —
not execution.

Every timed region in this project therefore ends with
`jax.block_until_ready(...)` on the outputs
([timing.py](../src/jaxscale_lm/utils/timing.py)). The flip side: in the
training loop we *don't* synchronize every step; the host runs ahead
enqueueing work while the device computes, and only the periodic
`device_get` for logging forces a sync. Host-device synchronization also
hides in innocent code: `float(x)`, `np.asarray(x)`, `if bool(mask.all())`.
The EOS check in the decode loop
([generate.py](../src/jaxscale_lm/inference/generate.py)) is a deliberate,
documented example.

## `jax.Array` and sharding

`jax.Array` is JAX's unified array type: every array carries a `Sharding`
that says which devices hold which slice. The same type covers a
single-device array, a replicated array, and a batch sharded across eight
devices — which is why the trainer's code path is identical for one device
and many ([trainer.py](../src/jaxscale_lm/training/trainer.py)).

Inside `jax.jit`, XLA's GSPMD partitioner propagates input shardings
through the computation: with batches sharded over the `data` mesh axis and
parameters replicated, the gradient all-reduce is inserted automatically.
We never write collectives by hand. See [sharding.md](sharding.md).

## `jax.grad`, `jax.value_and_grad`

Reverse-mode autodiff over pure functions. The training step differentiates
a loss-*sum* (not mean) so that gradients accumulated across microbatches
add linearly and can be normalized once by the global token count — making
gradient accumulation exactly equivalent to a larger batch
([step.py](../src/jaxscale_lm/training/step.py), verified by a regression
test).

## `jax.lax.scan`

`scan` compiles a loop body once and runs it sequentially over a leading
axis, keeping compile time constant regardless of trip count. Gradient
accumulation scans over stacked microbatches. The alternative — a Python
loop inside jit — would unroll and recompile for every accumulation
setting.

## `jax.vmap`

`vmap` vectorizes a per-example function over a batch axis. This codebase
mostly expresses batching directly with batched matmuls (the natural form
for Transformers), so `vmap` appears only where it genuinely helps; we do
not add transformations for show.

## Explicit randomness

JAX PRNG is splittable and stateless: a `key` deterministically produces
streams via `fold_in`/`split`, never via hidden global state. The root key
lives in `TrainState` and is checkpointed; per-step dropout keys are
derived as `fold_in(root, step)` so resuming at step N reproduces the exact
stream an uninterrupted run would have used
([step.py](../src/jaxscale_lm/training/step.py),
[seed.py](../src/jaxscale_lm/utils/seed.py)).

## Why fixed shapes matter for decode

Autoregressive decoding naively grows the sequence by one token per step —
which under jit would mean a new shape, hence a fresh XLA compilation,
*every step*. The fixed-capacity KV cache
([cache.py](../src/jaxscale_lm/model/cache.py)) turns decoding into a
shape-stable operation: arrays are pre-allocated at capacity, each step
writes one position via `jax.lax.dynamic_update_slice_in_dim` and masks
positions beyond the valid length. One compile serves the entire
generation, at the cost of attention always scanning the full capacity —
the classic compiled-decode trade.
