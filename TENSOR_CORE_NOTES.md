# int4 tensor-core symmetric BBQ local join (`nnd-bbq-tc`)

`local_join_kernel_bbq_symmetric_int4` (`cpp/src/neighbors/detail/nn_descent.cuh`) replaces the
scalar `packed_nibble` (dp4a) symmetric local join with `u4xu4->s32` `nvcuda::wmma` MMA
(`m8n8k32`), modeled on `local_join_kernel_wmma`'s fragment/accumulator pattern. Dispatch
(`GNND::local_join`): `packed_nibble` -> int4 tensor-core path; `single_bit`/`dibit` -> kept on
the original scalar popc/dp4a path as reference points; `transpose_half_byte`/`seven_bit`/
`unsigned_byte` were dropped from this branch's symmetric dispatch (`RAFT_FAIL` on those).

int4 sub-byte MMA isn't supported on Blackwell (compute capability >= 10.0) -- the kernel's
`__CUDA_ARCH__ >= 750` compile-time guard alone would just silently compile to a no-op there, so
the `packed_nibble` dispatch case also has a host-side runtime check
(`raft::util::arch::SM_range(SM_75(), SM_generic<1000>())`, mirroring the existing WMMA `< 700`
check elsewhere in this file) that throws instead of silently launching nothing.

Test setup throughout: `my_tests/bbq/bbq_nn_descent_comparison`, Falcon dataset
(`base1_falcon_1024_1M.fbin`, 10k rows, dim=1024), L40S (sm_89).

```bash
ncu --kernel-name local_join_kernel_bbq_symmetric_int4 \
  --metrics launch__shared_mem_per_block_static,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,launch__registers_per_thread \
  --section Occupancy --section SpeedOfLight --launch-count 1 \
  ./my_tests/bbq/build/bbq_nn_descent_comparison --data-file=/tmp/data/base1_falcon_1024_1M.fbin 4 4 1
```

## Correctness

1. Isolated GPU probe confirmed `packed_nibble`'s existing byte layout (`byte = low | high<<4`)
   is directly MMA-compatible, no repacking, for same-buffer-as-both-operands self-joins.
2. First real run hit `cudaErrorMisalignedAddress`: sub-byte IMMA loads need 16-byte row
   alignment; the scalar kernel's 4-byte pad (132 B rows) didn't provide it. Fixed by dropping
   the pad (`BBQ_ROW_BYTES=128` is already a clean multiple of 16) and `__align__(16)`.
3. From then on, recall tracked the scalar baseline (0.929) within sampling noise (0.926-0.929)
   across every subsequent change below.
4. After restoring `single_bit`/`dibit` as reference points in the dispatch, ran all three side
   by side on the same dataset: `packed_nibble` (int4) 0.927, `single_bit` (scalar) 0.582,
   `dibit` (scalar) 0.785 -- all comfortably above their respective min-recall thresholds (0.15,
   0.50), dispatch confirmed working for all three together, not just packed_nibble in isolation.

## Results

| Stage | Cycles | Duration | Load bank conflicts | Notes |
|---|---:|---:|---:|---|
| Scalar baseline (`packed_nibble`, dp4a) | 4,516,246 | 1.90 ms | 157,722 | pre-existing |
| v1: straight MMA port | 3,741,893 | 1.49 ms | 286,888,873 | `a_frag`/`b_frag` reloaded redundantly in the 2x2 sub-tile loop |
| v1.1: hoist redundant fragment loads | 2,649,796 | 1.06 ms | 143,487,983 | halved `load_matrix_sync` calls, `mma_sync` count unchanged |
| v1.2: pad row stride (break 32-bank alignment) | 1,933,758 | 776 us | 109,198 (below baseline) | `MMA_PAD=16`; 128 B/row was exactly 32 banks |
| v1.3: custom store stride (tried, reverted) | 1,941,864 | 781 us | -- | store conflicts dropped 4.2x but cycles flat; not worth the added complexity, reverted to shared `SKEWED_MAX_NUM_BI_SAMPLES` |

**Net: ~2.3x fewer cycles, ~2.45x faster than the scalar baseline**, recall unchanged throughout.

Register pressure (64/thread, tied with SMEM at the 2-blocks/SM occupancy cap) was investigated
but not reduced: removing `#pragma unroll` from the `kk` loop had zero effect, and
`-Xptxas -v` confirmed 0 bytes spilled -- 64 is a clean allocation, not spill-driven, so the
register lever appears exhausted without forcing (untried: explicit `maxrregcount` to find the
real floor). Per the rule "only push SMEM reduction once registers actually drop," the
`s_ov`-aliasing SMEM idea (see below) was not attempted.

## Known follow-ups (not done)

- **`s_ov`-aliasing to remove the separate `s_distances_u32` buffer**, mirroring
  `local_join_kernel_wmma`'s `s_distances`/`s_ov` alias. Valid here for the same reason (register
  accumulator, one-time store) but gated on register pressure coming down first (see above) --
  SMEM and registers are both at the exact same occupancy cap right now, so shrinking only one
  won't change occupancy.
- Explicit `maxrregcount`/`__launch_bounds__` experiment to find the real register floor.
- Latency/ILP root cause: NCU flags "low compute and memory utilization on both axes, check
  Scheduler/Warp State Statistics" -- not yet pulled.
- Asymmetric BBQ local join has no int4 path; this branch only touches symmetric.
- Warp-tiling constants (`WARPS_PER_DIM`, `SUB_PER_DIM`, `WARP_TILE`) are explicit/derived in the
  kernel now, not hardcoded -- `SUB_PER_DIM=2` is a forced consequence of
  `(MAX_NUM_BI_SAMPLES/MMA_M)/WARPS_PER_DIM`, not an arbitrary choice.
- `BLOCK_SIZE=1024` (1:1 tile:warp mapping, no sub-tile loop) was considered and ruled out: L40S
  caps at 1536 resident threads/SM, so 1024 threads/block would cap occupancy at 1 block/SM,
  worse than today's 2.

See `git log` on this branch for the incremental progression; each stage above was independently
measured and recall-verified before moving to the next.
