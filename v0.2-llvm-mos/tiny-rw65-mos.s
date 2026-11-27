; Stage 11Q: native RamWorks copy, no ROM execution.
; 80STORE + HIRES + PAGE1 keeps main $2000-$3fff visible to both
; reads and writes even when RAMRD/RAMWRT select extended aux.
; The entire routine MUST link within that range (checked by probe).
; ALTZP off, main read/write at entry/exit, bank 0 outside calls.
; Caller-saved LLVM-MOS imaginary registers $02-$06 are scratch.
.global tiny_rw65_move, tiny_rw65_move_end

STOREOFF = $C000
STOREON  = $C001
RDMAIN   = $C002
RDAUX    = $C003
WRMAIN   = $C004
WRAUX    = $C005
RDSTORE  = $C018
RDPAGE2  = $C01C
RDHIRES  = $C01D
PAGE1    = $C054
PAGE2    = $C055
LORES    = $C056
HIRES    = $C057
RAMWORKS = $C073

.section .text,"ax",@progbits
tiny_rw65_move:
    php
    sei
    cld
    lda RDPAGE2
    pha
    lda RDHIRES
    pha
    lda RDSTORE
    pha
    sta PAGE1
    sta HIRES
    sta STOREON
    ; Fetch all main-memory globals before either mapping changes.
    lda tiny_rw_src
    sta $02
    lda tiny_rw_src+1
    sta $03
    lda tiny_rw_dst
    sta $04
    lda tiny_rw_dst+1
    sta $05
    lda tiny_rw_count
    sta $06
    ldx tiny_rw_count+1
    lda tiny_rw_bank
    sta RAMWORKS
    lda tiny_rw_direction
    beq .Lread
    sta WRAUX
    bra .Lcopy
.Lread:
    sta RDAUX
.Lcopy:
    ldy #0
    cpx #0
    beq .Ltail
.Lpage:
    lda ($02),y
    sta ($04),y
    iny
    bne .Lpage
    inc $03
    inc $05
    dex
    bne .Lpage
.Ltail:
    ldx $06
    beq .Ldone
.Lbyte:
    lda ($02),y
    sta ($04),y
    iny
    dex
    bne .Lbyte
.Ldone:
    sta RDMAIN
    sta WRMAIN
    stz RAMWORKS
    ; Restore display/addressing state before re-enabling interrupts.
    ; Disable 80STORE first when necessary: restoring PAGE2 while
    ; HIRES+80STORE are still set would switch out this very code.
    pla
    bmi .Lstoreok
    sta STOREOFF
.Lstoreok:
    pla
    bmi .Lhiresok
    sta LORES
.Lhiresok:
    pla
    bpl .Lpageok
    sta PAGE2
.Lpageok:
    plp
    rts
tiny_rw65_move_end:
