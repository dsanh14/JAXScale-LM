# Benchmark run `20260611_085752_0eb57a`

- platform: **cpu** (cpu), 1 device(s)
- host: macOS-26.5.1-arm64-arm-64bit
- git: `7dfa39040182ebc09189774757adc79b025163a7` (dirty)
- versions: jax 0.10.1, flax 0.12.7, optax 0.2.8, orbax 0.12.0, python 3.12.13
- model: 2L x 64h, vocab 259

## cache

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| cached_p16_g32 | steady_state | ok | 11.24 ms | - | - | 5 | ms_per_token=0.4 |
| naive_p16_g32 | steady_state | ok | 46.58 ms | - | - | 5 | ms_per_token=1.5 |

## compilation

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| forward_seq64 | eager | ok | 3.65 ms | 4.16 ms | 0.32 | 5 |  |
| forward_seq64 | first_call | ok | 93.09 ms | - | - | 1 |  |
| forward_seq64 | steady_state | ok | 0.31 ms | 0.33 ms | 0.01 | 10 | first_call_over_steady=304.5 |
| forward_seq128 | eager | ok | 3.63 ms | 3.90 ms | 0.17 | 5 |  |
| forward_seq128 | first_call | ok | 89.30 ms | - | - | 1 |  |
| forward_seq128 | steady_state | ok | 0.65 ms | 0.67 ms | 0.01 | 10 | first_call_over_steady=136.5 |
| shape_recompilation | first_call_per_shape | ok | 0.34 ms | - | - | 1 |  |
| shape_recompilation | first_call_per_shape | ok | 0.63 ms | - | - | 1 |  |
| shape_recompilation | first_call_per_shape | ok | 100.00 ms | - | - | 1 |  |

## decode

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| decode_b1_ctx16 | steady_state | ok | 0.07 ms | 0.07 ms | 0.01 | 10 | tokens_per_second=14801.2; ms_per_token=0.1 |
| decode_b1_ctx64 | steady_state | ok | 0.07 ms | 0.09 ms | 0.01 | 10 | tokens_per_second=14585.2; ms_per_token=0.1 |
| decode_b2_ctx16 | steady_state | ok | 0.10 ms | 0.11 ms | 0.01 | 10 | tokens_per_second=19386.0; ms_per_token=0.1 |
| decode_b2_ctx64 | steady_state | ok | 0.10 ms | 0.11 ms | 0.01 | 10 | tokens_per_second=19716.6; ms_per_token=0.1 |
| decode_b4_ctx16 | steady_state | ok | 0.19 ms | 0.20 ms | 0.01 | 10 | tokens_per_second=21371.4; ms_per_token=0.2 |
| decode_b4_ctx64 | steady_state | ok | 0.18 ms | 0.18 ms | 0.00 | 10 | tokens_per_second=22066.3; ms_per_token=0.2 |

## e2e

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| generate_p16_g32 | steady_state | ok | 11.24 ms | - | - | 5 |  |
| generate_p64_g32 | steady_state | ok | 11.88 ms | - | - | 5 |  |

## prefill

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| prefill_b1_p16 | steady_state | ok | 0.23 ms | 0.26 ms | 0.03 | 10 |  |
| prefill_b1_p64 | steady_state | ok | 0.52 ms | 0.56 ms | 0.03 | 10 |  |
| prefill_b2_p16 | steady_state | ok | 0.33 ms | 0.33 ms | 0.01 | 10 |  |
| prefill_b2_p64 | steady_state | ok | 0.89 ms | 0.93 ms | 0.02 | 10 |  |
| prefill_b4_p16 | steady_state | ok | 0.52 ms | 0.59 ms | 0.04 | 10 |  |
| prefill_b4_p64 | steady_state | ok | 1.53 ms | 1.55 ms | 0.02 | 10 |  |

## training

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| train_step_b1x1_float32 | steady_state | ok | 1.05 ms | 1.09 ms | 0.02 | 10 | tokens_per_second=60894.4 |
| train_step_b2x1_float32 | steady_state | ok | 1.65 ms | 1.71 ms | 0.04 | 10 | tokens_per_second=77420.3 |
| train_step_b4x1_float32 | steady_state | ok | 2.80 ms | 2.89 ms | 0.12 | 10 | tokens_per_second=91269.7 |
| train_step_accum2_b2 | steady_state | ok | 3.17 ms | 3.27 ms | 0.09 | 10 | tokens_per_second=80681.3 |
