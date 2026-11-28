# Inference performance

## Runtime kernels, 2026-11-28

The comparison starts at merged upstream `0538aa9`, which already includes the sparse eight-row kernel. Both images booted in MAME 0.289 with `apple2ee -aux rw3 -sl7 cffa202` and the same TNI6 model. Times are emulated seconds from submitting the question, including prompt processing and answer generation. The first-character measurement observes the 40-column text screen.

| Measurement | Merged main | Runtime kernels | Time saved |
| --- | ---: | ---: | ---: |
| France, first visible answer character | 1,068.5 s | 950.1 s | 11.1% |
| France, complete 31-character answer | 2,688.2 s | 2,409.2 s | 10.4% |
| Whale, complete 24-character answer | 1,737.5 s | 1,548.4 s | 10.9% |
| Cold boot to input prompt | 49.6 s | 56.8 s | 7.2 s longer |

The exact prompts were `WHAT IS THE CAPITAL OF FRANCE?` and `IS A WHALE A FISH`. Both builds answered `The capital of France is Paris.` and `No, a whale is a mammal.` respectively. Both used the same installed ROM set, for which MAME reports checksum differences on `341-0132-d.e12` and `sc01a.bin`. The physical Appletini comparison is pending.

Two intermediate builds isolate the rescaler and ReLU changes. All four stages produced the same France answer.

| France build | Emulated time | Incremental saving |
| --- | ---: | ---: |
| Merged main | 2,688.2 s | |
| Biased weights and cache bytes, shared Q/K/V preparation, zero-input quantizer shortcut and batched attention coefficients | 2,573.5 s | 114.7 s |
| Generated rescaler, ordinary F1 | 2,481.0 s | 92.5 s |
| Generated rescaler and ReLU-aware F1 | 2,409.2 s | 71.8 s |

- Biasing banked bytes during loading or cache storage lets the matrix and attention value loops use `LDY abs,X` directly. Embedding and position rows reverse the transform in their staging buffer. The disk model keeps its original bytes.
- K, V and Q share a prepared input until the first attention head replaces it. Cache-only prompt steps retain the last-layer early return. Zero q8 inputs bypass fractional division. Attention compares each numerator against the shared maximum shifted right eight bits before doing the full rounded division.
- After the model closes, the program claims `$0C00–$0FFF` in the ProDOS bitmap. A one-time emitter writes 797 bytes of straight-line code for 3×3-byte and 6×2-byte multiplication using the linked lookup tables. Normal exit releases those pages. F1 tracks the greatest negative magnitude across batches so that sparse rescaling preserves the original packing exponent.
- The exponential table stores its entries in three runs of 354 three-byte, 355 two-byte and 443 one-byte values. Startup expands the narrow runs into the original three-byte auxiliary layout using the idle inference scratch buffer. The SYS is 36,061 bytes, leaving 547 bytes under the ProDOS load limit.

### Per-vector F2 unrolling experiment

`python3 tests/bench-f2-unroll.py` replays all 218 F2 vectors from the France run in cc65's `sim65`. The harness checks all 20,928 output records against integer arithmetic. It gives the candidate 60 preinstalled instruction slots in the protected window, patches only three column operands per slot, and falls back to the regular kernel on partial batches and oversized vectors. 182 vectors fit. The flat simulation excludes accelerator video-write waits.

| F2 column traversal | Cycles for the replay |
| --- | ---: |
| Linked nonzero-column loop | 69,019,996 |
| Unrolled slots, including target-side preparation | 69,105,797 |
| Unrolled slots assuming free preparation | 68,745,206 |

Patching cost 360,591 cycles across the trace and recovered 274,790 in execution. The boot image uses the faster linked-column traversal.

### Reproduce the checks

Run from `v0.2-llvm-mos` with the Apple IIe ROMs and CFFA firmware configured in MAME. Use each revision's `tiny.po` with the runtime branch's timing script.

```sh
TINY_TIMING=1 mame apple2ee -aux rw3 -sl7 cffa202 -hard1 tiny.po \
  -natural -nothrottle -video none -sound none -skip_gameinfo \
  -autoboot_script tests/mame-profile.lua
```

`TINY_QUESTION` selects another prompt. Set `TINY_CHECK=1` to print logit hashes. These modes run without a symbol map. Compare hashes with `tiny-linux --check -m TINY.INT -q <question>`. All 122 native hash rows matched the upstream portable reference across the France and whale prompts and a 72-character input. In MAME, `QUIT` released the four read-ahead pages after inference initialization. The France stack watermark reached `$BE60` (160 bytes). The LLVM-MOS simulator also exercises zero/maximum counts, bank-tail batch sizes, signed products, F1 ties and the generated-code buffer guards via `make sim-test`.

The verified 800 KiB ProDOS image has SHA-256 `034366f3384111ccd0c0453b0c9ce4db40c69da99ab96cd141c403ccdc5f9cc0`. Its `TINY.SYSTEM` has SHA-256 `320bf951f42c474a8abdb5b0b1460637d7126aa26d269f5fd06eb1d6aa869193`.

## Stage 12E baseline and earlier optimization work

Measured 2026-11-27 in MAME (`apple2ee -aux rw3`), one question
(`WHAT IS THE CAPITAL OF FRANCE?`, 31-character answer, 62 inference steps).
The run is deterministic: two identical builds give the same time to 0.1 s.

Baseline: upstream `f6aaf15` plus "attention scores use the prepared kernel".
Inference takes 4,284.7 emulated seconds, about 69 s per token.

## Where the time goes

Call-stack samples, once per frame, about 257,000 samples.

| Share | What |
|---|---|
| 58.1% | `matvec`: inner kernel 41.6, per-row `tiny_rescale` 8.8, `pack` 2.9, `q8` 2.1, C row loop 1.5 |
| 26.7% | `attention`: only 3.8 is multiply work (dot 1.5, axpy 2.2); 64-bit helper arithmetic about 13; banked cache reads about 6 |
| 9.5% | `normalize`: `__mulsi3` for `v*v` 2.55, 64-bit multiplies 4.2, `pack` 0.9, `isqrt_round` 0.5 |
| 3.6% | `residual` |
| 1.5% | `row`, `cache_store` |

Other facts that drive the plan:

- 82.3% of the inputs to `f2` (256 columns) are exactly zero after ReLU and
  quantization. Every other matrix input is about 1% zero. Weights are about 1% zero.
- The prepared kernel costs 90.5 cycles per element (4 table loads cross a page
  about 40% of the time; the third accumulator byte carries about half the time).
- Attention costs about 39,000 cycles per (head, position) pair, of which the
  24-element dot is 2,200. Its share grows from 5% at the first token to 35-40%
  at position 60, and equals everything else at about position 93.
- The generic 64-bit multiply core costs about 90 cycles per byte product.
- `v*v` in `normalize` costs about 2,300 cycles per element.
- Memory: the linker region ends at `$AF00`; the baseline leaves 568 bytes.
  The soft stack used at most 162 bytes of its 4,096 during inference.
  Compiled C is bulky: ten lines in `attention` cost 542 bytes.

## Plan

Gains are relative to the baseline run. "Measured" means a prototype was timed.

| # | Change | Gain |
|---|---|---|
| 1 | Skip zero-activation columns in the prepared kernel | measured 8.3% |
| 2 | Attention reads cached keys in place; scales read once per position, not per head | measured 3.8% |
| 3 | Tighter inner loop in the 64-bit multiply core | measured 2.2% |
| 4 | Sum of squares in `normalize` without `__mulsi3` | estimated 2.7% |
| 5 | Several rows per pass in the matrix kernel (shares the per-column setup) | estimated 9-11% |
| 6 | Attention arithmetic in assembly (score, exp stage, axpy) | estimated 10-17%, more for long answers |
| 7 | `pack` and `residual` in assembly | estimated 5% |
| 8 | Unrolled `tiny_rescale` | estimated 2.4% beyond item 3 |

Memory has to be reclaimed along the way: remove code the native build no
longer uses, and move the linker limit into the idle stack reserve.

Every change must keep the `:check` hash trace identical to `tiny-linux --check`.

## Results

Same question, same MAME setup; every row passes the `:check` hash trace
against `tiny-linux --check` (31 positions).

| Build | Inference | vs. baseline | Per token |
|---|---|---|---|
| Baseline | 4,284.7 s | | 69.1 s |
| + zero-column skip, multiply loop, sum of squares (1, 3, 4) | 3,714.3 s | -13.3% | 59.9 s |
| + attention key reads in place, shared scales (2) | 3,551.6 s | -17.1% | 57.3 s |
| + eight rows per kernel pass (5) | 3,148.6 s | -26.5% | 50.8 s |
| + attention score, batch and value kernels in assembly (6) | 2,839.6 s | -33.7% | 45.8 s |
| + pack and residual in assembly (7) | 2,705.3 s | -36.9% | 43.6 s |
| + row sums in zero page | 2,688.2 s | -37.3% | 43.4 s |
| Same, built with `TINY_SINGLE_ROW` | 3,158.4 s | -26.3% | 50.9 s |

### Accelerators: do not write into the window in a hot loop

The kernels run with HIRES on so that $2000-$3FFF stays visible while
auxiliary memory is mapped. That range is then hi-res page 1, and on an
accelerator that synchronises video-memory writes with the display, every
write into it stalls. Measured on an accelerated Apple II (2026-11-27): with
row sums kept in instruction operands, 81% of the time was video wait
(21 MHz effective); with the sums in zero page, 0% (87 MHz effective).
MAME shows no difference, so check this on real hardware.

On that machine the benchmark question takes 12 s to the first character
and 31 s in total with eight rows per pass, and 32 s with `TINY_SINGLE_ROW`.
The single-row build runs its cycles faster (98.9 against 86.9 MHz
effective) but has 17% more of them; the cache-miss rate is the same (18%).

The multi-row kernel now keeps its sums in zero page (`rsum`, 24 bytes;
`-mreserve-zp=30`) and reaches the tables through zero-page pointers.
`TINY_SINGLE_ROW` uses the same code with one record per pass
(3,158.4 s in MAME against 2,688.2 s for eight rows per pass).
Still writing into the window: the attention value sums (three writes per
element, about 1.4% of the run), 18 address bytes per kernel call, and
`tiny_mac65_prepare` once per projection.

Item 8 (unrolled `tiny_rescale`) is not done. Rescaling is now 13.8% of the
run and unrolling would save roughly 600 of its 2,200 cycles per row, but an
unrolled 3x3 plus 6x2 byte multiply needs about 900 bytes and the load image
has about 200 to spare. A compact subroutine version is slower than the
existing loop (78 against 66 cycles per byte product).

What changed in the build:

- `tiny-link.ld` replaces the stock linker script. The stock one caps code
  plus zeroed data at $8f00; only the load image has that limit. The stack
  keeps 1 KB (measured use: 186 bytes including model load).
- The load image is the binding limit now (36,293 of 36,608 bytes). The first
  1,056 bytes of the exp table sit in `tiny-qsquare-mos.s` as the initial
  contents of the kernel workspace and of `qhb`, a biased table plane that
  `tiny_mac65_init` generates after the table has gone to auxiliary memory.
- The general dot and axpy and the single-row entry points are assembled only
  with `--defsym TINY_MAC_GENERAL=1` (the tests). `--defsym TINY_SINGLE_ROW=1`
  swaps the eight-row kernel for a one-row loop with the same interface, for
  comparing cache behaviour on accelerators (`make MACASM=...`).
- `attention` was split into smaller functions: llvm-mos's register allocator
  crashed on the larger one.

Tests: `make sim-test` runs every assembly kernel against C arithmetic in the
LLVM-MOS simulator (`tests/tiny-asm-test.c`). `tests/verify-wide.py` and the
`MACTEST` link were already failing on the upstream sources with this
toolchain and still are.

Where the time goes now (2,705 s run): multi-row kernel 39%, rescale 14%,
attention 25% (key scores 7), normalize 10% (its two multiplies per element
6), pack 6%, `q8` 5%, residual 3%.

## Reproducing

`tests/mame-profile.lua` boots the disk image, types a question, samples the
PC and the call stack once per frame, and writes `stacks.txt`, `pcs.txt` and
`series.txt` to the working directory. It needs `syms.lua` there, an array of
`{address, "name"}` pairs for the text symbols from `llvm-nm -n TINY.SYSTEM.elf`.
`tests/profile-report.py <dir>` prints inclusive and self time per function.

```sh
mame apple2ee -aux rw3 -sl7 cffa202 -hard1 tiny.po -nothrottle -video none \
     -sound none -seconds_to_run 30000 -autoboot_script tests/mame-profile.lua
```

Environment: `TINY_QUESTION` overrides the question; `TINY_CHECK=1` types
`:check` first and prints the hash rows instead of timing an answer.
