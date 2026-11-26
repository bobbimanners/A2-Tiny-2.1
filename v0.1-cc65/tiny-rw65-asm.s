; Apple //e enhanced, RamWorks-compatible bank select at $C073.
; Firmware AUXMOVE runs in ROM, so instruction fetch remains valid when
; RAMRD changes. All C code, globals, stack and staging buffers in main RAM.
.setcpu "65C02"
.export _tiny_rw65_move
.import _tiny_rw_bank, _tiny_rw_direction
.import _tiny_rw_src, _tiny_rw_end, _tiny_rw_dst

AUXMOVE   = $C311
STOREOFF  = $C000
STOREON   = $C001
RDMAIN    = $C002
WRMAIN    = $C004
CXEXTERNAL = $C006
CXINTERNAL = $C007
RDCX      = $C015
RDSTORE   = $C018
RAMWORKS  = $C073

.segment "CODE"
.proc _tiny_rw65_move
    php
    sei
    cld
    ; Save firmware workspace (including unused bytes between pointers).
    ldx #7
savezp:
    lda $3c,x
    pha
    dex
    bpl savezp
    lda RDSTORE
    pha
    lda RDCX
    pha
    sta STOREOFF
    sta CXINTERNAL
    ; Contract: main ZP/stack, main read/write at entry and exit.
    sta RDMAIN
    sta WRMAIN
    lda _tiny_rw_src
    sta $3c
    lda _tiny_rw_src+1
    sta $3d
    lda _tiny_rw_end
    sta $3e
    lda _tiny_rw_end+1
    sta $3f
    lda _tiny_rw_dst
    sta $42
    lda _tiny_rw_dst+1
    sta $43
    lda _tiny_rw_bank
    sta RAMWORKS
    lda _tiny_rw_direction
    lsr a                   ; C=1 main->aux, C=0 aux->main
    jsr AUXMOVE
    ; Firmware restores main read/write (as set above).
    sta RDMAIN
    sta WRMAIN
    stz RAMWORKS             ; bank zero before IRQs or console/ProDOS
    pla
    bmi :+
    sta CXEXTERNAL
:
    pla
    bpl :+
    sta STOREON
:
    ldx #0
restorezp:
    pla
    sta $3c,x
    inx
    cpx #8
    bne restorezp
    plp
    rts
.endproc
