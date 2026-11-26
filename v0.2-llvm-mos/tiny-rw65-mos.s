; Apple //e enhanced, RamWorks-compatible bank select at $C073.
; Firmware AUXMOVE runs in ROM, so instruction fetch remains valid when
; RAMRD changes. All C code, globals, stack and staging buffers in main RAM.
.global tiny_rw65_move

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

.section .text,"ax",@progbits
tiny_rw65_move:
    php
    sei
    cld
    ; Save firmware workspace (including unused bytes between pointers).
    ldx #7
.Lsavezp:
    lda $3c,x
    pha
    dex
    bpl .Lsavezp
    lda RDSTORE
    pha
    lda RDCX
    pha
    sta STOREOFF
    sta CXINTERNAL
    ; Contract: main ZP/stack, main read/write at entry and exit.
    sta RDMAIN
    sta WRMAIN
    lda tiny_rw_src
    sta $3c
    lda tiny_rw_src+1
    sta $3d
    lda tiny_rw_end
    sta $3e
    lda tiny_rw_end+1
    sta $3f
    lda tiny_rw_dst
    sta $42
    lda tiny_rw_dst+1
    sta $43
    lda tiny_rw_bank
    sta RAMWORKS
    lda tiny_rw_direction
    lsr a                   ; C=1 main->aux, C=0 aux->main
    jsr AUXMOVE
    ; Firmware restores main read/write (as set above).
    sta RDMAIN
    sta WRMAIN
    stz RAMWORKS             ; bank zero before IRQs or console/ProDOS
    pla
    bmi 9f
    sta CXEXTERNAL
9:
    pla
    bpl 9f
    sta STOREON
9:
    ldx #0
.Lrestorezp:
    pla
    sta $3c,x
    inx
    cpx #8
    bne .Lrestorezp
    plp
    rts

