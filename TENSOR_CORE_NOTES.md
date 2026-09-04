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

### 1x4 (`single_bit` document)

Same kernel, templated on `DocumentLayout` (`bbq_layout::packed_dibit` or `bbq_layout::single_bit`)
so only the promotion function and its native:promoted expansion ratio differ (2x for
`packed_dibit`, 4x for `single_bit` -- 1 native byte of `single_bit` fully determines 1 promoted
output word on its own). First cut used a straightforward nested unrolled loop
(`promote_single_bit_word_to_nibble_quad`, correct but not vectorized); a branch-free SWAR version
(4 lane-wise 2-bit-field extractions + spreads, transposed into the 4 per-native-byte output words
via chained `__byte_perm` pairs, since one `__byte_perm` call only reaches 2 of the 4 field-words)
cut cycles by ~7.6%. Both promotion versions verified against a bit-by-bit reference exhaustively
(loop version: all 256 byte values; SWAR version: 200k random 32-bit words against the loop
version).

| Kernel | Cycles | Duration | Recall |
|---|---:|---:|---:|
| Scalar `1+4t` (`single_bit` doc x `transpose_half_byte` query) | 3,874,994 | 1.56 ms | 0.743-0.745 |
| int4 `1+4`, loop-based promotion | 2,791,889 | 1.11 ms | 0.745-0.746 |
| int4 `1+4`, SWAR/`__byte_perm` promotion | 2,578,307 | 1.03 ms | 0.745-0.746 |

**~1.5x fewer cycles, ~1.5x faster than the scalar baseline** with the SWAR promotion (up from
~1.4x with the loop version) -- a much smaller margin than `2x4`'s ~2.4x, and `1x4`'s absolute
cycle count is *higher* than `2x4`'s despite `single_bit`'s native document footprint being half
of `packed_dibit`'s. See "Why asymmetric is slower than symmetric" below for the full accounting;
short version: SMEM-write volume is identical between `1x4` and `2x4` (both promote to the same
512 B/row nibble-width buffer), so the remaining `1x4` vs `2x4` gap is promotion *compute*, not
bandwidth -- a 1-bit-to-nibble expansion needs a 4-way byte transpose, a 2-bit-to-nibble expansion
only a 2-way interleave.

### Comparison against the dense fp16 tensor-core baseline (`local_join_kernel_wmma`)

All three int4 kernels also beat the existing dense (unquantized, fp16 `wmma`) local-join kernel
used for the non-BBQ NN-Descent build, though by a much smaller margin than against the scalar
BBQ baselines -- expected, since that kernel is already tensor-core-based rather than scalar
dp4a/popc:

| Kernel | Cycles | Speedup vs `local_join_kernel_wmma` (2,826,551 cycles) |
|---|---:|---:|
| Symmetric `4x4` (`packed_nibble`) | 1,933,758 | ~1.46x |
| Asymmetric `2x4` (`packed_dibit` doc) | 2,485,943 | ~1.14x |
| Asymmetric `1x4` (`single_bit` doc, SWAR) | 2,578,307 | ~1.10x |

### Why asymmetric is slower than symmetric (not DRAM bandwidth -- SMEM write volume)

Initial explanation ("asymmetric issues more global-memory reads") turned out to be imprecise --
worth correcting. Per-row DRAM read volume is actually about *equal* between symmetric and `2x4`
(1536 B either way for dim=1024) and *lower* for `1x4` (1280 B) than for symmetric, since a
densely-packed document reads fewer native bytes than `packed_nibble`. The real asymmetry is in
**SMEM write volume**: the document is always promoted to the full 512 B/row nibble-width buffer
regardless of how compact its native storage is, and -- unlike the symmetric kernel's `new x new`
phase, which reuses a single 512 B SMEM buffer as both the `a_frag` and `b_frag` source for a
self-join -- the asymmetric kernel can never skip loading either side, since document and query
are always two distinct quantized representations of the same rows.

Per-row-slot byte accounting (dim=1024; `packed_nibble` row=512 B, `packed_dibit` row=256 B,
`single_bit` row=128 B, promoted/SMEM width is always 512 B):

| Kernel | Phase | Buffers (native/DRAM -> promoted/SMEM) | DRAM read | SMEM write |
|---|---|---|---:|---:|
| Symmetric `4x4` | `new x new` | `s_nv`: 512B -> 512B (reused as both A and B operand) | 512 B | 512 B |
| Symmetric `4x4` | `new x old` | `s_nv` (reload): 512B->512B; `s_ov`: 512B->512B | 1024 B | 1024 B |
| Symmetric `4x4` | **total** | | **1536 B** | **1536 B** |
| Asymmetric `2x4` | `new x new` | doc: 256B->512B (promoted); query: 512B->512B | 768 B | 1024 B |
| Asymmetric `2x4` | `new x old` | doc (reload): 256B->512B; query (old): 512B->512B | 768 B | 1024 B |
| Asymmetric `2x4` | **total** | | **1536 B** | **2048 B** |
| Asymmetric `1x4` | `new x new` | doc: 128B->512B (promoted); query: 512B->512B | 640 B | 1024 B |
| Asymmetric `1x4` | `new x old` | doc (reload): 128B->512B; query (old): 512B->512B | 640 B | 1024 B |
| Asymmetric `1x4` | **total** | | **1280 B** | **2048 B** |

SMEM-write ratio (asymmetric/symmetric) is 2048/1536 = **1.33x** for both `2x4` and `1x4` --
closely matching the measured cycle ratios (`2x4`/symmetric = 1.29x, `1x4`/symmetric = 1.33x
exactly). DRAM read volume predicts nothing here (it's equal or even lower for the asymmetric
kernels), which is why "extra bandwidth" needed this correction: it's SMEM traffic from the
mandatory full-width promotion plus the loss of phase-1 self-join reuse, not DRAM bandwidth.

This also explains the remaining `1x4` vs `2x4` gap on top of the (identical) 1.33x SMEM-write
ratio: `1x4`'s cycle ratio (1.33x) is fractionally *higher* than `2x4`'s (1.29x) despite `1x4`
reading less from DRAM, consistent with `1x4`'s promotion needing more compute (4-way transpose)
than `2x4`'s (2-way interleave) for the same SMEM-write volume.

## Original theoretical motivation: SOL analysis (pre-implementation)

Before any of the int4 kernels above existed, this instruction-count model motivated building
them: `popc = 15 cycles`, `dp4a = 2 cycles`, hypothetical `int8 MMA (m16n8k32) = 4 cycles`,
applied to the (then-only) scalar `code_inner_product_*_2x1` microkernels. Per-distance
MAC-instruction cost, `n` = row bytes, `bits` = code bit-width:

| Layout | bits | instr | cycles/distance |
|---|---:|---|---:|
| `single_bit` (symmetric) | 1 | popc, `n/4` | 480 |
| `dibit` (symmetric) | 2 | popc, `bits*(n/4)` | 960 |
| `transpose_half_byte` (symmetric) | 4 | popc, `bits*(n/4)` | 1920 |
| `packed_nibble` (symmetric) | 4 | dp4a, `n/8` | 128 |
| `unsigned_byte` (symmetric) | 8 | dp4a, `n/16` | 64 |
| 1+2 (asymmetric) | 1+2 | popc, `doc_bits*query_bits*(query_stride/4)` | 480 |
| 1+4t (asymmetric) | 1+4 | popc, same formula | 480 |
| 2+4t (asymmetric) | 2+4 | popc, same formula | 960 |
| int8 tensor core (hypothetical) | 8 | mma, `ceil(K/32)/(16*8)` | 0.125 |

Multi-plane symmetric layouts (`dibit`, `transpose_half_byte`) split the row into `bits` stripes
and cross every document stripe against every query stripe (`bits^2` sub-calls), so cost scales
as `bits*(n/4)` -- not free with bit-width. The asymmetric kernel has the same cross-plane
structure with `doc_bits * query_bits` sub-calls instead of `bits^2`.

**Takeaway at the time:** the encoding, not the bit-width, sets the cost. At equal byte count,
popc-based layouts cost `15/2 = 7.5x` more per instruction than dp4a-based ones -- 4-bit
`transpose_half_byte` (1920 cycles/distance) is **15x** more expensive than 4-bit `packed_nibble`
(128 cycles/distance) at the *same* bit width, purely from the popc-vs-dp4a choice. The
asymmetric path had no dp4a-based option at the time (no "packed nibble" analogue for either
operand), so it was stuck paying the popc tax -- exactly the gap `packed_dibit` and `single_bit`'s
int4 promotion in this branch were built to close. A naive int8/tensor-core promotion looked like
another large drop in MAC cost on top of that, but was never actually free: caching a full
int8-promoted row in static shared memory would have pushed several configurations over the
compile-time static-`__shared__` ceiling common to CUDA GPUs generally (not an L40-specific
number), forcing the same K-tiled streaming structure the real int4 kernels ended up using anyway
(`BBQ_ROW_BYTES = 128` in the actual implementation) rather than a single full-row MMA pass.

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
