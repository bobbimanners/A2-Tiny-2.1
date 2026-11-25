tiny.po: tiny
	mv tiny TINY#060803
	cadius createvolume tiny.po TINY 800KB
	cadius addfile tiny.po /TINY TINY.SYSTEM#FF0000 
	cadius addfile tiny.po /TINY TINY#060803 
	cadius addfile tiny.po /TINY TINY.INT#060000 

tiny: tiny-stage8b.c tiny-rw65.c tiny-rw65-asm.s tiny-mac65-zp.s
	/usr/bin/cl65 -t apple2enh -O --ld-args '-D__EXEHDR__=0' --ld-args '-D__HIMEM__=0xBF00' -m tiny.map -o tiny tiny-stage8b.c tiny-rw65.c tiny-rw65-asm.s tiny-mac65-zp.s
