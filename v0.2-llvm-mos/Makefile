MOSCC ?= /opt/llvm-mos/bin/mos-apple2-clang
HOSTCC ?= cc
MOSFLAGS ?= -mcpu=mos65c02 -Oz -flto -fno-inline-functions -fno-unroll-loops
# Keep these defaults for the memory-constrained native build.
EXTRA ?= -DTINY_COMPACT_LUT
MACFLAGS = -mreserve-zp=6
HOSTFLAGS ?= -std=c99 -O2 -Wall -Wextra
MAIN = tiny.c tiny-wide-mos.s tiny-rw65.c tiny-rw65-mos.s tiny-mac65-mos.s tiny-prodos.c tiny-prodos-asm.s
HEADERS = tiny-rw65.h tiny-prodos.h tiny-wide.h
TESTDIR = tests

.PHONY: all tests host wide-test verify-wide clean
all: tiny.po

tiny.po: TINY.SYSTEM ../v0.1-cc65/TINY.INT\#060000
	mv TINY.SYSTEM TINY.SYSTEM#FF2000
	cadius createvolume tiny.po TINY 800KB
	cadius addfile tiny.po /TINY ../prodos.242/PRODOS#FF0000
	cadius addfile tiny.po /TINY TINY.SYSTEM#FF2000
	cadius addfile tiny.po /TINY ../v0.1-cc65/TINY.INT#060000 

TINY.SYSTEM: $(MAIN) $(HEADERS) Makefile
	$(MOSCC) $(MOSFLAGS) $(MACFLAGS) $(EXTRA) -DTINY_MOS -DTINY_WIDE_ASM -Wl,-Map=tiny-mos.map $(MAIN) -o $@

# Optional standalone native diagnostics; not linked into TINY.SYSTEM.
tests: FILETEST.SYSTEM RWTEST.SYSTEM MACTEST.SYSTEM WIDETEST.SYSTEM
wide-test: WIDETEST.SYSTEM
verify-wide:
	python3 $(TESTDIR)/verify-wide.py

WIDETEST.SYSTEM: $(TESTDIR)/tiny-wide-test.c tiny-wide-mos.s tiny-wide.h
	$(MOSCC) $(MOSFLAGS) -I. -Wl,-Map=wide-test.map $(TESTDIR)/tiny-wide-test.c tiny-wide-mos.s -o $@
FILETEST.SYSTEM: $(TESTDIR)/tiny-file-test.c tiny-prodos.c tiny-prodos-asm.s tiny-prodos.h
	$(MOSCC) $(MOSFLAGS) -I. $(TESTDIR)/tiny-file-test.c tiny-prodos.c tiny-prodos-asm.s -o $@
RWTEST.SYSTEM: $(TESTDIR)/tiny-rw65-test.c tiny-rw65.c tiny-rw65-mos.s tiny-rw65.h
	$(MOSCC) $(MOSFLAGS) -I. $(TESTDIR)/tiny-rw65-test.c tiny-rw65.c tiny-rw65-mos.s -o $@
MACTEST.SYSTEM: $(TESTDIR)/tiny-mac65-test.c tiny-mac65-mos.s
	$(MOSCC) $(MOSFLAGS) $(MACFLAGS) -I. $^ -o $@
MACFULL.SYSTEM: $(TESTDIR)/tiny-mac65-test.c tiny-mac65-mos.s
	$(MOSCC) $(MOSFLAGS) $(MACFLAGS) -I. -DTINY_MAC_FULL $^ -o $@
MACBENCH.SYSTEM: $(TESTDIR)/tiny-mac65-test.c tiny-mac65-mos.s
	$(MOSCC) $(MOSFLAGS) $(MACFLAGS) -I. -DTINY_MAC_BENCH $^ -o $@

# Portable C references and host-only substitutes for hardware/MLI.
host: tiny-linux tiny-reader-linux tiny-apple-mock
tiny-linux: tiny.c Makefile
	$(HOSTCC) $(HOSTFLAGS) -DTINY_SMALL $(EXTRA) tiny.c -o $@
tiny-reader-linux: tiny.c tiny-prodos.c tiny-prodos.h Makefile
	$(HOSTCC) $(HOSTFLAGS) -DTINY_SMALL -DTINY_TEST_READER -DTINY_PRODOS_HOST_TEST $(EXTRA) tiny.c tiny-prodos.c -o $@
tiny-apple-mock: tiny.c tiny-prodos.c tiny-prodos.h tiny-rw65.h $(TESTDIR)/host-mock.c Makefile
	$(HOSTCC) $(HOSTFLAGS) -I. -DTINY_MOS -DTINY_PRODOS_HOST_TEST $(EXTRA) tiny.c tiny-prodos.c $(TESTDIR)/host-mock.c -o $@

clean:
	rm -f TINY.SYSTEM FILETEST.SYSTEM RWTEST.SYSTEM MACTEST.SYSTEM WIDETEST.SYSTEM MACFULL.SYSTEM MACBENCH.SYSTEM *.SYSTEM.elf tiny-mos.map wide-test.map tiny-linux tiny-reader-linux tiny-apple-mock

test:
	mame -w apple2ee -aux rw3 -sl7 cffa202 -hard1 tiny.po -nothrottle -natural -console -frameskip 0
