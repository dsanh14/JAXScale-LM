# Sharding and Distributed Execution

## The logical mesh

JAXScale-LM arranges devices in a 2-D logical mesh with named axes
(`distributed.axis_names`, default `("data", "model")`), built in
[mesh.py](../src/jaxscale_lm/distributed/mesh.py):

```
devices = mesh_utils.create_device_mesh((data_size, model_size))
mesh = Mesh(devices, axis_names=("data", "model"))
```

`data_axis_size: -1` means "all devices not consumed by the model axis".
Mesh dimensions are validated against the actual device count with
actionable errors before any array is placed.

## Named shardings and PartitionSpec

A `PartitionSpec` maps array dimensions to mesh axes; `NamedSharding`
binds a spec to a mesh ([partitioning.py](../src/jaxscale_lm/distributed/partitioning.py)):

| Object | Spec | Meaning |
|---|---|---|
| parameters, optimizer state, RNG key | `P()` | replicated on every device |
| train batch `[accum, micro, seq]` | `P(None, "data", None)` | microbatch rows split across the data axis |
| eval batch | `P()` | replicated (final batch can be ragged) |

Placement is one explicit call: `jax.device_put(tree, sharding)`
([placement.py](../src/jaxscale_lm/distributed/placement.py)). Inside
`jax.jit`, XLA's GSPMD partitioner propagates these input shardings through
the whole step function and inserts the gradient all-reduce automatically —
data parallelism without hand-written collectives.

## Replication vs data parallelism vs tensor parallelism

- **Replication**: every device holds a full copy; no communication, no
  speedup. The correct placement for parameters under pure data
  parallelism, and what a 1×1 mesh degenerates to.
- **Data parallelism**: each device computes forward/backward on its slice
  of the batch; gradients are all-reduced. Scales well when the per-device
  batch is large enough that compute dominates the all-reduce.
- **Tensor/model parallelism**: individual weight matrices are split over
  the `model` axis and activations are exchanged *inside* each layer. Far
  chattier than data parallelism; worthwhile when the model doesn't fit (or
  underutilizes) one device. The mesh reserves the `model` axis for this,
  but the default configuration keeps `model_axis_size: 1`; enabling real
  tensor parallelism (sharded QKV/MLP matrices) is a documented roadmap
  item, not a shipped feature.

## Single-host vs multi-host

Everything in this repository is **single-process, single-host**: one
Python process drives all local devices, and `jax.process_count() == 1`.

On multi-host deployments (TPU pods, GPU clusters), each host runs the same
program; `jax.distributed.initialize()` wires them together, every process
sees only its *addressable* (local) devices, and global arrays span hosts.
The diagnostics module already distinguishes `jax.local_device_count()`
from `jax.device_count()` and prints `addressable_shards`, so the concepts
are visible — but multi-host execution is documented here rather than
tested, because the environment has a single host (see
[limitations.md](limitations.md)).

## Addressable vs global devices

- `jax.devices()` — all devices in the (global) computation.
- `jax.local_devices()` — devices this process can touch.
- `array.addressable_shards` — the pieces of a sharded array this process
  holds; on a single host that is all of them.

`scripts/inspect_devices.py` and
[diagnostics.py](../src/jaxscale_lm/distributed/diagnostics.py) print all
of these, plus mesh shape and a sample array's sharding.

## Limitations of the local test environment

This machine exposes **one CPU device**. Multi-device behavior is exercised
with XLA's host-platform override, set *before* JAX initializes:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python scripts/inspect_devices.py
```

This carves the CPU into 8 XLA devices that share the same silicon. It
validates real sharding logic — mesh construction, placement, shard shapes,
collective insertion, divisibility errors (`tests/unit/test_distributed.py`
runs exactly this in a subprocess) — but it is **not** accelerator scaling:
the 8 "devices" contend for the same cores and memory bandwidth. No
throughput number from simulated devices appears anywhere in
[results.md](results.md), and none should ever be presented as scaling
evidence.

On real multi-device hardware, `configs/train/multi_device.yaml` shards the
batch across all devices with no code changes.

## Why data parallelism only pays off at scale

A data-parallel step costs roughly
`compute(per-device batch) + all_reduce(parameter count)`. The all-reduce
term is independent of batch size. For the tiny models in this lab, the
per-device compute for a CPU-sized batch is microseconds-to-milliseconds —
the same order as dispatch and communication overhead — so multi-device
execution can easily be *slower* than one device. This is expected and is
called out in the benchmark methodology
([benchmarking.md](benchmarking.md)); data parallelism earns its overhead
when per-device compute dominates, i.e. larger models and batches.
