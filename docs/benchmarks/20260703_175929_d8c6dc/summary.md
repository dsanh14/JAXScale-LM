# Benchmark run `20260703_175929_d8c6dc`

- platform: **cpu** (cpu), 1 device(s)
- host: macOS-26.5.1-arm64-arm-64bit
- git: `6bca6212cf2bf4f36d12c0f3e20c38f7bb1c9ff8`
- versions: jax 0.10.1, flax 0.12.7, optax 0.2.8, orbax 0.12.0, python 3.12.13
- model: 2L x 64h, vocab 259

## cache

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| cached_p16_g32 | steady_state | ok | 10.85 ms | - | - | 5 | ms_per_token=0.3 |
| naive_p16_g32 | steady_state | ok | 46.39 ms | - | - | 5 | ms_per_token=1.4 |

## compilation

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| forward_seq64 | eager | ok | 3.94 ms | 4.41 ms | 0.46 | 5 |  |
| forward_seq64 | first_call | ok | 97.70 ms | - | - | 1 |  |
| forward_seq64 | steady_state | ok | 0.31 ms | 0.33 ms | 0.01 | 10 | first_call_over_steady=319.3 |
| forward_seq128 | eager | ok | 3.76 ms | 4.07 ms | 0.32 | 5 |  |
| forward_seq128 | first_call | ok | 88.84 ms | - | - | 1 |  |
| forward_seq128 | steady_state | ok | 0.66 ms | 0.69 ms | 0.02 | 10 | first_call_over_steady=133.9 |
| shape_recompilation | first_call_per_shape | ok | 0.35 ms | - | - | 1 |  |
| shape_recompilation | first_call_per_shape | ok | 0.64 ms | - | - | 1 |  |
| shape_recompilation | first_call_per_shape | ok | 105.70 ms | - | - | 1 |  |

## decode

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| decode_b1_ctx16 | steady_state | ok | 0.07 ms | 0.07 ms | 0.00 | 10 | tokens_per_second=15099.0; ms_per_token=0.1 |
| decode_b1_ctx64 | steady_state | ok | 0.07 ms | 0.07 ms | 0.01 | 10 | tokens_per_second=15345.4; ms_per_token=0.1 |
| decode_b2_ctx16 | steady_state | ok | 0.12 ms | 0.12 ms | 0.00 | 10 | tokens_per_second=17229.0; ms_per_token=0.1 |
| decode_b2_ctx64 | steady_state | ok | 0.10 ms | 0.10 ms | 0.00 | 10 | tokens_per_second=19797.9; ms_per_token=0.1 |
| decode_b4_ctx16 | steady_state | ok | 0.18 ms | 0.19 ms | 0.01 | 10 | tokens_per_second=21932.8; ms_per_token=0.2 |
| decode_b4_ctx64 | steady_state | ok | 0.18 ms | 0.18 ms | 0.00 | 10 | tokens_per_second=22508.8; ms_per_token=0.2 |

## e2e

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| generate_p16_g32 | steady_state | ok | 11.23 ms | - | - | 5 |  |
| generate_p64_g32 | steady_state | ok | 11.74 ms | - | - | 5 |  |

## prefill

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| prefill_b1_p16 | steady_state | ok | 0.20 ms | 0.20 ms | 0.00 | 10 |  |
| prefill_b1_p64 | steady_state | ok | 0.52 ms | 0.54 ms | 0.02 | 10 |  |
| prefill_b2_p16 | steady_state | ok | 0.30 ms | 0.31 ms | 0.00 | 10 |  |
| prefill_b2_p64 | steady_state | ok | 0.89 ms | 0.91 ms | 0.02 | 10 |  |
| prefill_b4_p16 | steady_state | ok | 0.50 ms | 0.52 ms | 0.01 | 10 |  |
| prefill_b4_p64 | steady_state | ok | 1.67 ms | 1.77 ms | 0.08 | 10 |  |

## training

| name | mode | status | median | p90 | std | iters | notes |
|---|---|---|---|---|---|---|---|
| train_step_b1x1_float32 | steady_state | ok | 1.09 ms | 1.18 ms | 0.05 | 10 | tokens_per_second=58508.7 |
| train_step_b2x1_float32 | steady_state | ok | 1.61 ms | 1.66 ms | 0.06 | 10 | tokens_per_second=79748.7 |
| train_step_b4x1_float32 | steady_state | ok | 2.73 ms | 2.83 ms | 0.07 | 10 | tokens_per_second=93844.5 |
| train_step_accum2_b2 | steady_state | ok | 3.12 ms | 3.38 ms | 0.15 | 10 | tokens_per_second=81950.6 |
