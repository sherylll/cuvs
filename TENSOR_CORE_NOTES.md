# int4 tensor-core BBQ local join (`nnd-bbq-tc`)

`local_join_kernel_bbq_symmetric_int4` and `local_join_kernel_bbq_asymmetric_int4`
(`cpp/src/neighbors/detail/nn_descent.cuh`) replace the scalar `packed_nibble` (dp4a) local joins
with `u4xu4->s32` `nvcuda::wmma` MMA (`m8n8k32`), modeled on `local_join_kernel_wmma`'s
fragment/accumulator pattern. Dispatch (`GNND::local_join`): symmetric `packed_nibble` -> int4
tensor-core path (`single_bit`/`dibit` kept on the original scalar popc/dp4a path as reference
points; `transpose_half_byte`/`seven_bit`/`unsigned_byte` dropped from this branch's symmetric
dispatch, `RAFT_FAIL` on those); asymmetric `packed_nibble`(bits=2) x `packed_nibble`(bits=4) ->
int4 tensor-core path (see "Asymmetric int4" below; other asymmetric pairs -- `single_bit`+`dibit`,
`single_bit`+`transpose_half_byte`, `dibit`+`transpose_half_byte` -- stay on the scalar path).

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

## Asymmetric int4 (`local_join_kernel_bbq_asymmetric_int4`)

Extends the same u4xu4 MMA structure to the asymmetric case: 2-bit document x 4-bit query.

Changed `packed_nibble`'s on-disk encoding from halves-paired (`(v[b]<<4)|v[dim/2+b]`) to
contiguous (`(v[2k]<<4)|v[2k+1]`) so document and query bytes line up on the same dimensions.
First attempt reused `packed_nibble`'s nibble-width layout for the 2-bit document too (straight
copy, zero promotion) -- but that stores a 2-bit value in a 4-bit slot, doubling the document's
DRAM footprint and read bandwidth for no reason. Replaced with a new `bbq_code_layout::packed_dibit`
(added to `cpp/include/cuvs/preprocessing/quantize/bbq.hpp`): a genuinely dense 2-bit format, 4
values/byte, contiguous (`byte k = (v[4k]<<6)|(v[4k+1]<<4)|(v[4k+2]<<2)|v[4k+3]`), half the bytes
of `packed_nibble`. The document is promoted to nibble width during SMEM staging
(`promote_packed_dibit_word_to_nibble_pair`, branch-free SWAR: two lane-wise "spread nibble to
byte" ops + two `__byte_perm` calls to interleave, no per-byte loop) while the query still loads
via a plain word copy. Validated in isolation first (`my_tests/bbq/contiguous_int4_probe.cu`), the
SWAR promotion equivalence checked exhaustively against the straightforward per-byte version over
random 32-bit inputs, then end-to-end via `bbq_nn_descent_comparison`'s `2p` CLI option (now
`packed_dibit` instead of `packed_nibble` at bits=2).

| Kernel | Cycles | Duration | Recall | Static SMEM | Bank conflicts (ld / st) |
|---|---:|---:|---:|---:|---|
| Scalar `2+4t` (`dibit` doc x `transpose_half_byte` query) | 6,043,686 | 2.44 ms | 0.841 | 30.48 KiB | 0 / 0 |
| int4 `2p+4`, dense doc + promotion (`packed_dibit` 2-bit doc x `packed_nibble` 4-bit query) | 2,485,943 | 992.5 us | 0.835 | 36.37 KiB | 77,868 / 1,821,046 |

**~2.4x fewer cycles, ~2.5x faster than the scalar baseline**, recall unchanged (within sampling
noise, 128 sampled queries) -- consistent with the symmetric kernel's speedup. The promotion's
own cost is small: +2.5% cycles vs. the (rejected) straight-copy/wasteful-DRAM version measured
earlier (2,426,509 cycles), confirming the SWAR unpack is cheap relative to the MMA math and
memory traffic -- so there was no real tradeoff in fixing the DRAM footprint. Store bank conflicts
are non-trivial (same root cause as the symmetric kernel's v1.2 finding: `MMA_STORE_STRIDE =
SKEWED_MAX_NUM_BI_SAMPLES` isn't bank-friendly) but per that same finding fixing them didn't move
overall cycles for the symmetric kernel, so it wasn't chased here either.

## Known follow-ups (not done)

- **`s_ov`-aliasing to remove the separate `s_distances_u32` buffer**, mirroring
  `local_join_kernel_wmma`'s `s_distances`/`s_ov` alias. Valid here for the same reason (register
  accumulator, one-time store) but gated on register pressure coming down first (see above) --
  SMEM and registers are both at the exact same occupancy cap right now, so shrinking only one
  won't change occupancy.
- Explicit `maxrregcount`/`__launch_bounds__` experiment to find the real register floor.
- Latency/ILP root cause: NCU flags "low compute and memory utilization on both axes, check
  Scheduler/Warp State Statistics" -- not yet pulled.
- Warp-tiling constants (`WARPS_PER_DIM`, `SUB_PER_DIM`, `WARP_TILE`) are explicit/derived in the
  kernel now, not hardcoded -- `SUB_PER_DIM=2` is a forced consequence of
  `(MAX_NUM_BI_SAMPLES/MMA_M)/WARPS_PER_DIM`, not an arbitrary choice.
- `BLOCK_SIZE=1024` (1:1 tile:warp mapping, no sub-tile loop) was considered and ruled out: L40S
  caps at 1536 resident threads/SM, so 1024 threads/block would cap occupancy at 1 block/SM,
  worse than today's 2.

See `git log` on this branch for the incremental progression; each stage above was independently
measured and recall-verified before moving to the next.
