# TINY for the Apple //e

**A 367,000-parameter transformer running natively on a 65C02 under ProDOS.**

This is an Apple II port of [TINY 2.1](https://github.com/WollyMonjac/Tiny-2.1), a small character-level question-answering language model originally written for DOS. The port runs the original trained network on an enhanced Apple //e with RamWorks-compatible auxiliary memory, using entirely integer arithmetic and native 65C02 assembly for its most computationally intensive operations.

There is no Raspberry Pi, modern accelerator card doing the neural-network computation, remote inference server, or cloud API involved. The 65C02 does the mathematics itself. Training and model conversion take place on a modern computer; inference takes place on the Apple II.

This is very much an experiment in what old hardware can do, rather than an attempt to build a useful contemporary AI assistant. As with a dog that can sing, the remarkable thing is not how well it sings, but that it sings at all.

## It works!

On **25 November 2026**, the first successful end-to-end test in MAME's enhanced Apple //e emulation produced the following result:

```text
TINY 8B / Apple //e / RamWorks
Banks 1-12 reserved for TINY.
Loading model...
....

> WHAT IS THE CAPITAL OF FRANCE?
Thinking...
........................................
.
Paris
```

![TINY answering a question on an emulated enhanced Apple IIe](screenshot.png)

The answer took rather a long time to arrive, but Paris was still the capital of France when it did.

## v0.1 and v0.2

The two versions of the Apple II port are maintained in parallel in this repository. Both run the same trained model, with the same architecture, fixed-point representation and context length.

- **v0.1 — [`v0.1-cc65/`](v0.1-cc65/):** The original working Apple //e port, using the `cc65` compiler and its `ca65` assembler. It uses a small ProDOS system loader to start a binary linked at `$0803`. Profiling in MAME revealed significant overhead in the cc65 runtime.
- **v0.2 — [`v0.2-llvm-mos/`](v0.2-llvm-mos/):** The port to the more sophisticated [LLVM-MOS](https://llvm-mos.org/) compiler. Its initial generated code was considerably larger than cc65's, so the wide arithmetic library was rewritten in 65C02 assembly and the inference workspace reorganized to fit in main memory. It builds a standalone ProDOS SYS executable at `$2000`, with no separate loader. A preliminary side-by-side MAME comparison indicated roughly three times the inference throughput of v0.1.

Each directory has its own Makefile. **Build v0.1 first, even if you only intend to run v0.2:** the v0.1 Makefile produces the shared `TINY.INT` model file used by both versions. See [Building and running](#building-and-running) below.

## The original TINY 2.1 for MSDOS

TINY 2.1 was designed to run on 486DX- and Pentium-class DOS machines. It is a small autoregressive transformer trained to answer questions drawn from approximately 3,500 facts, presented in various forms. Unlike most modern language models, it reads and generates text one character at a time.

The Apple II port retains the original architecture, context window and trained 8-bit matrix weights. No retraining, pruning, or reduction in the number of layers was necessary.

| Model characteristic | Value |
| --- | ---: |
| Architecture | Decoder-only causal transformer |
| Trainable parameters | 366,528 |
| Transformer blocks | 4 |
| Embedding width | 96 |
| Attention heads per block | 4 |
| Dimensions per attention head | 24 |
| Feed-forward hidden width | 256 |
| Vocabulary | 97 tokens |
| Context window | 128 tokens |
| Matrix weights | Signed 8-bit integers |
| Position encoding | Learned positional embeddings |
| Normalization | RMS normalization |
| Feed-forward activation | ReLU |
| Output projection | Shared with token embedding matrix |

The 128-token context is shared between the question and its generated answer. The current frontend accepts up to 72 printable input characters. This is a deliberately tiny model with a limited body of knowledge, not a general-purpose conversational LLM. It can and does produce incorrect answers.

## What changed from TINY 2.1?

The original DOS implementation already stores its matrix weights and attention cache in quantized form, but carries out much of the inference arithmetic in floating point. Simply recompiling that implementation for a 65C02 would be impractical. The Apple II has no hardware multiplier, floating-point unit, or flat address space large enough for the model.

The port therefore reimplements the inference engine around the capabilities of an eight-bit processor.

| Component | Original TINY 2.1 | Apple II port |
| --- | --- | --- |
| Target | x86 / DOS | 65C02 / ProDOS |
| Development tools | Turbo C | GCC for reference builds; cc65/ca65 (v0.1) or LLVM-MOS (v0.2) |
| Matrix weights | INT8 with floating-point row scales | Original INT8 bytes with integer row scales |
| Activations and residuals | Floating point | 16-bit block fixed point |
| Matrix and attention MACs | Floating-point arithmetic | Signed byte arithmetic with 24-bit assembly accumulators |
| RMS normalization | Floating point | Integer square root and reciprocal scaling |
| Softmax | Floating-point exponential | Fixed-point exponential lookup table |
| Wide intermediate arithmetic | Native arithmetic | Multiword integer C (v0.1); native 65C02 assembly primitives (v0.2) |
| Model and KV cache storage | DOS conventional memory | Banked RamWorks auxiliary memory |
| Model file | `TINY.BIN` (`TNY2`) | `TINY.INT` (`TNI6`) |
| User interface | DOS full-screen display and PC speaker | Simple ProDOS text console |

The model converter retains the original weight bytes and converts the floating-point scale factors and normalization gains to integer representations. Activation quantization and fixed-point arithmetic introduce numerical approximations, so the results are not guaranteed to be identical to the original DOS implementation for every prompt.

### Integer arithmetic

Vectors are represented by signed 16-bit mantissas and a common binary exponent:

```text
x[i] = mantissa[i] * 2^exponent
```

This block fixed-point representation provides useful dynamic range without a floating-point operation for every individual activation. Matrix operations temporarily quantize activations to signed bytes, accumulate the products, then rescale the result to the block fixed-point representation. Residual connections align exponents before adding vectors.

Neither native runtime requires floating-point arithmetic or `libm`. Operations needing wider intermediate precision use explicit multiword integer arithmetic in v0.1 and hand-written 65C02 primitives in the default v0.2 build. The portable C implementation remains useful as a numerical reference. RMS normalization uses an integer square root; softmax uses an offline-generated lookup table covering the relevant negative-exponential range.

### 65C02 multiplication and accumulation

The 6502 family has no multiply instruction, so the native MAC kernels use the classic quarter-square identity:

```text
a * b = floor((a + b)^2 / 4) - floor((a - b)^2 / 4)
```

Both squared terms have the same parity, so truncating their fractional parts introduces no error. Two byte-plane lookup tables occupy just 1 KiB and implement exact signed 8-bit multiplication using table accesses and subtraction.

The assembly module implements complete dot-product and scaled-vector accumulation loops, avoiding the overhead of calling an assembly multiply routine for every pair of operands. Products accumulate into signed 24-bit sums, exposed to the C engine as sign-extended 32-bit results. Longer dot products are split where necessary to avoid overflow.

These kernels handle the projections, feed-forward layers, query-key attention and weighted value accumulation. Both versions use these kernels. In v0.1 the rest of the engine is largely integer C compiled by cc65; v0.2 uses LLVM-MOS for the C code and also implements the wide arithmetic primitives in assembly.

### RamWorks memory management

The Apple II's 16-bit address space is much smaller than the model. Rather than treating auxiliary memory as a flat address space, TINY stores data in fixed-size records with explicit bank and offset addressing. Records never straddle bank boundaries.

The v0.1 implementation uses banks 1 through 12, reserving bank 0 for the usual Apple II auxiliary-memory environment. Only addresses `$2000` through `$BFFF` are used in each expansion bank, providing a 40 KiB data window per bank. Its allocation is shown below.

| Auxiliary memory allocation | Bytes |
| --- | ---: |
| Model weights, scales and gains | 381,796 |
| Key cache (including scale metadata) | 51,200 |
| Value cache (including scale metadata) | 51,200 |
| Bank-boundary padding | 864 |
| **Total allocated** | **485,060** |

In v0.1 the inference engine, working vectors, staging buffers, lookup tables and C stack reside in main memory. Matrix rows and relevant attention-cache segments are copied to a main-memory buffer for computation.

The v0.1 transfers use the Apple //e firmware's `AUXMOVE` routine at `$C311`, with RamWorks bank selection at `$C073`. The v0.2 kernels access records directly in `$4000–$BFFF` in banks 1 through 15. They keep the matrix row sums in main zero page and restore the bank mapping before returning to C. The exponential table and its interpolation routine borrow space in auxiliary bank 0, backed up in bank 15 until exit.

### LLVM-MOS arithmetic and workspace (v0.2)

The wide arithmetic library implements 64-bit addition, subtraction, comparisons, shifts, multiplication and division directly in 65C02 assembly. It uses the LLVM-MOS calling convention, passing operand pointers in zero-page register pairs. Rounded division reuses the remainder produced by the division routine rather than reconstructing it with a multiplication. A specialized single-byte divisor path handles common scaling operations, including division by 127.

The v0.2 build shares vector buffers and overlays scratch storage whose lifetimes do not overlap. After the model file closes, the program claims the read-ahead pages and generates the two straight-line multiplication kernels used for matrix rescaling.

| Main-memory range | v0.2 use |
| --- | --- |
| `$0000–$001F` | LLVM-MOS imaginary registers |
| `$0060–$00FF` | Compiler-managed zero page and MAC pointers |
| `$0800–$0BFF` | ProDOS file buffer |
| `$0C00–$0FFF` | Model read-ahead buffer during loading, executable rescaler during inference |
| `$1000–$1FFF` | Shared inference scratch arena |
| `$2000–$BAFF` | Linked code and static data. The SYS load image ends by `$AEFF` |
| `$BB00–$BEFF` | Reserved software stack |

The LLVM-MOS version uses a small buffered read-only file interface to ProDOS's Machine Language Interface (MLI). The native assembly wrappers preserve the required compiler and firmware workspace when making system calls.

## Development history

The port was developed incrementally, using Linux as a reference platform before moving to the Apple II. The development process used deterministic generation, test questions, standalone assembly tests and, where appropriate, logit comparisons to validate successive changes.

| Stage | Work undertaken |
| --- | --- |
| 0 | Establish a floating-point Linux reference implementation. |
| 1 | Replace matrix-vector multiplication with integer MACs. |
| 2 | Convert query-key dot products and weighted value accumulation to integers. |
| 3 | Introduce block fixed-point activation and residual vectors. |
| 4 | Implement integer RMS normalization. |
| 5 | Implement lookup-table softmax and integer token sampling. |
| 6 | Remove all remaining floating point; introduce the offline `TINY.INT` converter. |
| 7A | Replace native 64-bit arithmetic with explicit multiword integer routines and audit 16-bit C portability. |
| 7B | Implement exact quarter-square multiplication. |
| 7C | Implement native 65C02 MAC kernels and test them with `sim65`. |
| 8A | Simulate RamWorks banked storage under Linux. |
| 8B | Implement native RamWorks transfers and the ProDOS console application; resolve cc65 linker and memory-layout issues. |
| 9A | Bring up the native cc65 executable using the ProDOS loader at `$0803` and the zero-page-compatible MAC kernel. |
| 9B | Exercise the native model loader, bank allocation and inference frontend in MAME. |
| 9C | Demonstrate end-to-end question answering in MAME (25 November 2026); consolidate the first working cc65 port as **v0.1**. |
| 10A | Collect a statistical PC profile in MAME and correlate samples with cc65 linker and debug symbols. Identify runtime arithmetic and software-stack overhead as the principal bottlenecks. |
| 10B | Port the C engine, MAC kernels and RamWorks interface to LLVM-MOS. Add native buffered ProDOS MLI file access and build a standalone SYS executable. Investigate code-size optimization and native C 64-bit arithmetic. |
| 10C | Replace wide arithmetic primitives with native 65C02 assembly, including optimized shifts and division. Pass the standalone WIDETEST in MAME and recover approximately 7.5 KB of linked memory. |
| 10D | Overlay vector and mathematical scratch buffers and use main memory below `$2000`. Fit the program within the standard LLVM-MOS memory layout and launch it successfully via Bitsy Bye. Consolidate the LLVM-MOS port as **v0.2**. |

The Stage 9 subdivisions above describe the final cc65 integration and bring-up retrospectively.

This approach made it possible to distinguish arithmetic errors from compiler, assembly and bank-switching errors. It also leaves a portable reference implementation for future optimization work.

## Building and running

### Requirements

- Enhanced Apple //e with a 65C02, or a compatible emulator such as MAME (`apple2ee`).
- RamWorks-compatible expansion providing 16 distinct 64 KiB auxiliary banks for v0.2 (bank 0 plus banks 1–15). A 1 MiB or larger card provides sufficient capacity.
- ProDOS 8 and a volume with enough free space for the chosen executable, its loader if required, and the approximately 373 KiB model file. ProDOS 2.4 or later with Bitsy Bye is recommended for launching v0.2.
- A modern build host with GNU Make, Python 3 and a consistent installation of the [cc65 toolchain](https://cc65.github.io/). These are required to build v0.1 and generate the shared model.
- The [LLVM-MOS SDK](https://github.com/llvm-mos/llvm-mos-sdk) with its Apple II target, for building v0.2.
- A means of transferring ProDOS files, such as [Cadius](https://github.com/mach-kernel/cadius).

**Warning:** TINY reserves and overwrites data in auxiliary banks 1–12 for v0.1 and 1–15 for v0.2. Do not run it while a RamWorks RAM disk or other application is using these banks. Save your work first. A startup probe checks for aliased or missing banks.

### 1. Build both versions

From the root of the repository:

```sh
cd v0.1-cc65
make
cd ../v0.2-llvm-mos
make
```

**The order is important.** The v0.1 build generates `TINY.INT` as well as the cc65 executable. The v0.2 build generates its own `TINY.SYSTEM` but uses the very same `TINY.INT` file from v0.1. Run the v0.1 build first even if you are not planning to use the cc65 executable.

The model is converted offline from the released `dos/TINY.BIN` in the [original TINY 2.1 repository](https://github.com/WollyMonjac/Tiny-2.1). If the original model is not already available to the v0.1 Makefile, obtain it from upstream. The converter transforms the `TNY2` model into the integer `TNI6` format:

```sh
# Optional manual conversion, from v0.1-cc65:
python3 tiny-convert.py TINY.BIN TINY.INT
```

Normally the Makefile takes care of conversion. Neither Python nor the original training software is required on the Apple II.

The v0.1 Makefile builds a raw `TINY` binary linked at `$0803`. Loading below the usual SYS address of `$2000` makes approximately 6 KiB more main memory available. A matching version of the cc65 ProDOS loader is required (see below). The successful development build used Ubuntu's packaged cc65 tools (`/usr/bin/cl65`), rather than a mismatched older installation in the shell's search path.

The v0.2 Makefile builds `TINY.SYSTEM`, a native ProDOS SYS executable linked at `$2000`. If the LLVM-MOS compiler is not on your PATH, specify its location when building v0.2:

```sh
make MOSCC=/opt/llvm-mos/bin/mos-apple2-clang
```

The default v0.2 build uses `-Oz`, link-time optimization, disabled function inlining and loop unrolling, the compact exponential lookup table and the assembly wide arithmetic backend. The memory budget is tight; changing compiler options can cause linker overflow.

### 2. Boot either the v0.1 or v0.2 disk image

**Launch v0.2 through Bitsy Bye, not BASIC.SYSTEM.** The executable is too large for BASIC.SYSTEM's usual loading and buffer allocation; `-TINY.SYSTEM` can fail with `NO BUFFERS AVAILABLE`. From BASIC, type:

```text
BYE
```
Navigate to the v0.2 installation directory and select `TINY.SYSTEM`. Alternatively, boot a ProDOS volume with the v0.2 `TINY.SYSTEM` as its first SYS program. In either case, ensure that the current ProDOS directory contains `TINY.INT`.

### 3. Ask a question

Both versions probe auxiliary memory and load `TINY.INT` on startup. At the `>` prompt, enter a question. A progress dot is printed after processing each input token, including the prompt terminator. Answer characters are displayed as they are generated.

```text
what is the capital of france
what is the capital of estonia
is a whale a fish
what is 7 times 8
spell computer
```

| Command | Function |
| --- | --- |
| `:c0` | Deterministic generation (default) |
| `:c1` … `:c9` | Select nonzero creativity for probabilistic sampling |
| `:check` | Toggle per-token logit checksum diagnostics |
| `:banks` | Display bank allocation and memory-transfer statistics |
| `quit` or `exit` | Return to ProDOS |

In MAME, enable a RamWorks-compatible expansion and attach your ProDOS boot and model volumes. Turning off throttling (normally F10 in MAME's UI keyboard mode) speeds up inference, but also makes keyboard repeat rather enthusiastic. Re-enable throttling while entering questions, and disable it again during inference. Short-lived console output may be almost invisible with throttling disabled.

## Testing on Linux

Both versions retain a portable C reference implementation with the RamWorks memory backend simulated and the portable C MAC kernels selected automatically. For example, from `v0.1-cc65`:

```sh
gcc -std=c99 -O2 -Wall -Wextra -DTINY_SMALL \
    tiny.c -o tiny-linux

./tiny-linux -q 'what is the capital of france?'
./tiny-linux --profile --banks -q 'what is the capital of france?'
```

From `v0.2-llvm-mos`, the consolidated Makefile also provides optional host and standalone test targets:

```sh
make host
make tests
make wide-test
make verify-wide
```

`make host` builds the Linux reference and host-side file reader/simulation tests. The native test executables are optional and are not part of `TINY.SYSTEM`. To run the Linux reference from the v0.2 directory, make the shared `TINY.INT` available there (for example, copy or symlink it from `../v0.1-cc65/`).

`--logits` emits a CSV of next-token scores for each input position; `--check` emits compact checksums useful for cross-platform regression testing.

```sh
./tiny-linux -q 'abc' --check
```

On the Apple II, enter `:check`, then `abc`, and compare the printed position, input token, shared binary exponent and FNV-1a hash with the Linux output. This checks the raw fixed-point logit mantissas without requiring a large diagnostic dump from the Apple II.

The original cc65 assembly test harness can also be run independently of the transformer, if the test sources are retained:

```sh
cl65 -t sim65c02 -O -o mac65-test tiny-mac65-test.c tiny-mac65-zp.s
sim65 mac65-test full
sim65 -c mac65-test bench
```

The original assembly passed the exhaustive signed-byte multiplication test and the dot-product and weighted-accumulation tests in `sim65`. The benchmark produced the expected checksum of `41417000` for 256,000 MACs.

The v0.2 `WIDETEST.SYSTEM` verifies the new assembly wide arithmetic primitives against precomputed expected results. It passed in MAME before the complete LLVM-MOS executable was brought up.

## Performance (or lack thereof)

A forward pass through the original model requires up to 353,376 dense matrix multiply-accumulates, plus attention work that grows with the number of tokens in context. At a full context of 128 tokens, the theoretical total reaches 451,680 MACs per token (before skipping zero-valued attention coefficients).

Since each output token represents one character, this is a considerable amount of arithmetic just to print one letter. The entire question must also be processed character by character before the answer begins. Memory transfers, normalization and multiword scaling add further overhead.

### Why port to LLVM-MOS?

A statistical profile of the v0.1 build in MAME revealed that most of the time was not being spent in the hand-written MAC kernels. After excluding the identified keyboard-polling samples, the approximate distribution was:

| Component | Share of active samples |
| --- | ---: |
| cc65 runtime library | 79.72% |
| 65C02 assembly MAC kernels | 9.82% |
| Compiled TINY C code | 7.12% |
| AUXMOVE firmware | 2.53% |
| RamWorks interface | 0.81% |

Approximately 35% of active samples were attributed to stack management and load/store helpers, 24% to 32-bit multiplication and 10% to 32-bit unsigned division. These are statistical estimates, not exact cycle counts or caller-attributed timings.

The initial LLVM-MOS build was too large to fit. Even with size-oriented optimization, the experimental build needed about 50 KB of linked main memory. Replacing the wide arithmetic primitives with roughly 1 KB of assembly reduced the linked footprint by about 7.5 KB. Workspace overlays and the use of low main memory then allowed the full program to fit, with the normal stack reservation intact.

A preliminary A/B comparison of v0.1 and v0.2 in two unthrottled MAME instances showed **roughly three times the throughput** for v0.2, judged by the rate of progress dots. This is an order-of-magnitude observation rather than a controlled emulated-cycle benchmark. The improvement reflects both the compiler change and the new assembly arithmetic; it cannot be attributed to LLVM-MOS alone.

Controlled v0.2 emulator timings, per-kernel experiments, and accelerator measurements for the preceding build are recorded in the [optimization notes](v0.2-llvm-mos/OPTIMIZATION.md).

Possible future work:

- Repeat the MAME profile for v0.2 and optimize the remaining hot paths, including the native MAC kernels and wide arithmetic.
- Reduce bank-transfer overhead by staging groups of matrix rows, or by developing bank-aware computation kernels.
- Optimize fixed-point scaling, division and normalization for the 65C02.
- Explore auxiliary bank 0 for additional code, workspace or lookup tables, and investigate native 65816 arithmetic and hardware acceleration (including the UltraWarp II and its interaction with expansion RAM).
- Explore smaller or ternary-weight models specifically trained for eight-bit inference.

## Status

**v0.1 (cc65):** The original trained model has been exercised through the intermediate Linux implementations, the arithmetic kernels have passed tests in `sim65`, and native ProDOS inference produced a correct answer in MAME on 25 November 2026. The screenshot above documents that milestone.

**v0.2 (LLVM-MOS):** The new assembly wide arithmetic test passed in MAME. The complete executable fits in the standard LLVM-MOS memory layout, launches through Bitsy Bye and processes input considerably faster than v0.1 in a preliminary side-by-side test. Both versions retain the same model format and architecture.

The versions are maintained in parallel. The runtime-kernel build has been checked against the portable reference in MAME. Its physical Appletini timing is pending.

## Acknowledgements and licensing

- **[WollyMonjac and TINY 2.1](https://github.com/WollyMonjac/Tiny-2.1)** — the original DOS project, model architecture, training pipeline, training data and pretrained model. This project would not exist without that work. The upstream README also credits Anthropic's Fable 5.2 for writing the original implementation and Opus 5.5 for training the model.
- **Bobbi (bobbimanners)** — initiated the Apple II port, directed its design, compiled and tested each stage, debugged the native build and brought up the first working ProDOS implementation in MAME.
- **ChatGPT (OpenAI)** — AI development collaborator on the Apple II port. ChatGPT helped design the staged conversion, wrote and revised the C and 65C02 assembly implementations, developed the model converter and regression test tools, and assisted with cc65 and memory-layout debugging, profiling, and the subsequent LLVM-MOS port and native wide arithmetic implementation. The original model was not trained by ChatGPT; its contribution was to the inference-engine port.
- The developers and maintainers of **cc65, LLVM-MOS, MAME, Cadius and ProDOS**, and the Apple II community, whose tools and accumulated knowledge make projects like this possible.

The original TINY 2.1 source code is MIT-licensed. Retain the upstream copyright and license notices when redistributing derived code. Consult the upstream `LICENSE` and `DATA_LICENSES.md` for the applicable terms, particularly for the model's training data and derived assets. This README does not establish a separate license for the Apple II port.

---

*The dog can sing. And it was even in tune.*
