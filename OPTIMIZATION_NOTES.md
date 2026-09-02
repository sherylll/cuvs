# Symmetric BBQ local-join optimization notes

## Current status

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
L2 Cache Throughput               %        35.14
SM Active Cycles              cycle   1626822.01
Compute (SM) Throughput           %        46.06
----------------------- ----------- ------------
```

## Focused NCU report commands

Use section filters when importing an existing report; `--print-details header` avoids the
large raw metric dump.

```bash
report=my_tests/bbq/ncu-bbq-symmetric.ncu-rep

ncu --import "$report" --section SpeedOfLight --page details --print-details header
ncu --import "$report" --section MemoryWorkloadAnalysis --page details --print-details header
ncu --import "$report" --section Occupancy --page details --print-details header
ncu --import "$report" --section SchedulerStats --page details --print-details header
ncu --import "$report" --section WarpStateStats --page details --print-details header
```

For equivalent WMMA output, set `report=my_tests/bbq/ncu-wmma.ncu-rep`.

To profile one matching symmetric BBQ kernel while collecting only the most relevant sections:

```bash
ncu --section SpeedOfLight \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*local_join_kernel_bbq_symmetric.*' \
  --launch-count 1 \
  --target-processes all \
  -o my_tests/bbq/ncu-bbq-symmetric \
  ./my_tests/bbq/build/bbq_nn_descent_comparison \
    --data-file=/tmp/data/deep1m-256-euclidean/base.fbin \
    4 1
```

## Vectorized shared-memory loading

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

## `uint32_t` accumulation

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

## Reduce `BBQ_ROW_BYTES`: 128 → 64

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

## 2×1 microkernel

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

## Compile-time layout and tile size

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