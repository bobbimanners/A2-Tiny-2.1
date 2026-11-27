# Regression tests

These are optional diagnostics, not dependencies of `TINY.SYSTEM`. Run the
Makefile targets from the parent directory.

| Target executable | Purpose |
| --- | --- |
| `FILETEST.SYSTEM` | ProDOS sequential reader: byte count and FNV-1a hash |
| `RWTEST.SYSTEM` | RamWorks bank independence and page-crossing transfers |
| `MACTEST.SYSTEM` | Dot product and AXPY regression |
| `WIDETEST.SYSTEM` | Native wide arithmetic and LLVM-MOS calling convention |
| `MACFULL.SYSTEM` | Extended signed-byte multiplication test |
| `MACBENCH.SYSTEM` | MAC benchmark |

Build with `make tests` (or `make MOSCC=/opt/llvm-mos/bin/mos-apple2-clang tests`).
The last two targets are built individually. Import each SYS file with Cadius
using the `#FF2000` suffix and launch through ProDOS. They wait for a key
before returning. **RWTEST overwrites data in auxiliary banks 1–12**; do not
run it on a machine with those banks in use.

For FILETEST, place `TINY.INT` in the current ProDOS directory and compare its
output with `python3 tests/model-hash.py TINY.INT` on the host. WIDETEST should
report PASS and 1788 checks. MACBENCH's expected checksum is 41417000.

## Simulator and emulator checks

`make sim-test` builds `tiny-asm-test.c` for the LLVM-MOS simulator and
compares every assembly kernel with plain C arithmetic. It needs no Apple II
and runs in seconds; run it after any change to the `.s` files. Add
`MACASM=-Wa,--defsym,TINY_SINGLE_ROW=1` to test the single-row kernel variant.

`mame-profile.lua` boots a disk image in MAME, types a question, and either
times the answer while sampling the call stack or, with `TINY_CHECK=1`,
prints the `:check` hash rows for comparison with `tiny-linux --check`.
`profile-report.py` summarizes the samples. See `../OPTIMIZATION.md`.

## Host regression

```sh
make host
./tiny-linux --check -q abc > ref.csv
./tiny-reader-linux --check -q abc > reader.csv
diff -u ref.csv reader.csv
printf ':check\nabc\nquit\n' | ./tiny-apple-mock
make verify-wide
```

The host executables expect `TINY.INT` in the working directory, or use `-m`
with the first two. The host mock tests the frontend and buffer lifetimes using
C substitutes for hardware and assembly. It does not test actual bank
switching, ProDOS bitmap ownership, LLVM-MOS code generation or the native ABI.
`verify-wide.py` uses a small source-level instruction interpreter; it is not
an assembler or a full Apple II emulator. Compare native `:check` rows against
`tiny-linux --check` for the same trained model and prompt.
