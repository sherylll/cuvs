# Symmetric BBQ local-join optimization notes


## Focused NCU report commands

```bash
ncu --kernel-name local_join_kernel_bbq_asymmetric   --metrics launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__registers_per_thread --section Occupancy --section SpeedOfLight  --launch-count 1 ./my_tests/bbq/build/bbq_nn_descent_comparison   --data-file=xxx   2 4t 1
```

## Test on DEEP

The original symmetric 4-bit implementation was approximately 4.5× slower than the WMMA path
(2.98 ms vs. 656.10 µs). Both reports show low DRAM throughput, indicating that the bottleneck is
scalar packed-code computation and shared-memory activity—not DRAM bandwidth.

### Baseline: symmetric 4-bit

```text
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.52
Elapsed Cycles                cycle      7494838
Memory Throughput                 %        73.70
DRAM Throughput                   %         0.25
Duration                         ms         2.98
L1/TEX Cache Throughput           %        74.47
L2 Cache Throughput               %         4.68
SM Active Cycles              cycle   7413044.63
Compute (SM) Throughput           %        73.70
----------------------- ----------- ------------
```

### Reference: WMMA

```text
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.51
Elapsed Cycles                cycle      1645570
Memory Throughput                 %        42.49
DRAM Throughput                   %         2.74
Duration                         us       656.10
L1/TEX Cache Throughput           %        42.94
L2 Cache Throughput               %         35.14
SM Active Cycles              cycle   1626822.01
Compute (SM) Throughput           %        46.06
----------------------- ----------- ------------
```

### Vectorized shared-memory loading

```text
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.51
Elapsed Cycles                cycle      3681050
Memory Throughput                 %        51.32
DRAM Throughput                   %         0.50
Duration                         ms         1.46
L1/TEX Cache Throughput           %        52.08
L2 Cache Throughput               %         9.65
SM Active Cycles              cycle   3624846.13
Compute (SM) Throughput           %        51.32
----------------------- ----------- ------------
```

### `uint32_t` accumulation

No measurable change from vectorized loading alone. But I prefer use of uint32 over float due to higher resolution. Overflow is unlikely given largest bit width of 8.

```text
Section: GPU Speed Of Light Throughput
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.51
Elapsed Cycles                cycle      3669578
Memory Throughput                 %        51.49
DRAM Throughput                   %         0.50
Duration                         ms         1.46
L1/TEX Cache Throughput           %        52.20
L2 Cache Throughput               %         9.71
SM Active Cycles              cycle   3616434.10
Compute (SM) Throughput           %        51.49
----------------------- ----------- ------------
```

### Reduce `BBQ_ROW_BYTES`: 128 → 64

Reducing shared memory increases occupancy; the 128-byte tile exceeded the 32 KiB threshold.

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.50
Elapsed Cycles                cycle      2672371
Memory Throughput                 %        72.38
DRAM Throughput                   %         0.69
Duration                         ms         1.07
L1/TEX Cache Throughput           %        72.93
L2 Cache Throughput               %        13.92
SM Active Cycles              cycle   2649715.17
Compute (SM) Throughput           %        72.38
----------------------- ----------- ------------
```

### 2×1 microkernel

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.50
Elapsed Cycles                cycle      2231535
Memory Throughput                 %        70.56
DRAM Throughput                   %         0.83
Duration                         us          892
L1/TEX Cache Throughput           %        71.34
L2 Cache Throughput               %        17.15
SM Active Cycles              cycle   2206126.67
Compute (SM) Throughput           %        70.55
----------------------- ----------- ------------
```

### Compile-time layout and tile size

Specializing the kernel by layout prevents unused variants from increasing register pressure.
Passing the fixed tile size as a template argument also removes tail handling.

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.48
Elapsed Cycles                cycle      1946122
Memory Throughput                 %        70.92
DRAM Throughput                   %         0.86
Duration                         us       783.26
L1/TEX Cache Throughput           %        71.71
L2 Cache Throughput               %        18.28
SM Active Cycles              cycle   1920570.73
Compute (SM) Throughput           %        70.93
----------------------- ----------- ------------
```

## Test on Falcon, asymmetric 2+4

### Baseline: asymmetric 2+4

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.51
Elapsed Cycles                cycle     10111946
Memory Throughput                 %        84.96
DRAM Throughput                   %         0.56
Duration                         ms         4.02
L1/TEX Cache Throughput           %        85.67
L2 Cache Throughput               %         7.76
SM Active Cycles              cycle  10017490.80
Compute (SM) Throughput           %        84.96
----------------------- ----------- ------------
```

### Reference: WMMA

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.45
Elapsed Cycles                cycle      2826551
Memory Throughput                 %        51.56
DRAM Throughput                   %         5.16
Duration                         ms         1.14
L1/TEX Cache Throughput           %        52.04
L2 Cache Throughput               %         44.14
SM Active Cycles              cycle   2779948.93
Compute (SM) Throughput           %        51.56
----------------------- ----------- ------------
```

### Compile-time layout

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.50
Elapsed Cycles                cycle      6439198
Memory Throughput                 %        77.99
DRAM Throughput                   %         0.77
Duration                         ms         2.57
L1/TEX Cache Throughput           %        78.75
L2 Cache Throughput               %         11.70
SM Active Cycles              cycle   6365721.71
Compute (SM) Throughput           %        77.99
----------------------- ----------- ------------
```

### Loop fission

Move the break-loop with if-else into two separate loops in `vec_load_bbq`. Cycles didn't drop,
but register usage decreased from 57 to 40.

### Bug fix: loop boundary

Fix loop boundary `num_row_pairs * SKEWED_MAX_NUM_BI_SAMPLES` →
`num_row_pairs * MAX_NUM_BI_SAMPLES`.

```text
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         8.99
SM Frequency                    Ghz         2.33
Elapsed Cycles                cycle      6088656
Memory Throughput                 %        76.69
DRAM Throughput                   %         0.86
Duration                         ms         2.61
L1/TEX Cache Throughput           %        77.40
L2 Cache Throughput               %        23.73
SM Active Cycles              cycle   6015591.25
Compute (SM) Throughput           %        77.49
----------------------- ----------- ------------
```
