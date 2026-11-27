# Optimization notes (Stage 12E baseline)

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
