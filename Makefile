tiny.po: tiny TINY.INT
	mv tiny TINY#060803
	mv TINY.INT TINY.INT#060000
	cadius createvolume tiny.po TINY 800KB
	cadius addfile tiny.po /TINY TINY.SYSTEM#FF0000   # LOADER.SYSTEM from cc65
	cadius addfile tiny.po /TINY TINY#060803 
	cadius addfile tiny.po /TINY TINY.INT#060000 

tiny: tiny.c tiny-rw65.c tiny-rw65-asm.s tiny-mac65-zp.s
	/usr/bin/cl65 -t apple2enh -O --ld-args '-D__EXEHDR__=0' --ld-args '-D__HIMEM__=0xBF00' -m tiny.map -o tiny tiny.c tiny-rw65.c tiny-rw65-asm.s tiny-mac65-zp.s

TINY.INT: TINY.BIN
	python3 tiny-convert.py

clean:
	rm *.o TINY#060803 TINY.INT#060000
