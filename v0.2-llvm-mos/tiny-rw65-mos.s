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

; Stage 12E: protected gate into aux0 executable code. Argument is a
; pointer in $02/$03 to a uint32_t in main RAM, replaced by the result.
; C has checked d < 18*65536. Only caller-saved $04..$11 are used.
; As with tiny_rw65_move, main ZP/stack and main R/W at entry, bank 0.
.global tiny_aux_exp, tiny_aux_exp_gate_end
.global tiny_aux_exp_image, tiny_aux_exp_image_end

tiny_aux_exp:
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
    ldy #0
    lda ($02),y
    sta $04
    iny
    lda ($02),y
    sta $05
    iny
    lda ($02),y
    sta $06
    stz RAMWORKS
    sta RDAUX
    jsr $4d80
    sta RDMAIN
    ; Return the four result bytes to the main-memory argument.
    ldy #0
    lda $08
    sta ($02),y
    iny
    lda $09
    sta ($02),y
    iny
    lda $0a
    sta ($02),y
    iny
    lda $0b
    sta ($02),y
    pla
    bmi .Lexpstoreok
    sta STOREOFF
.Lexpstoreok:
    pla
    bmi .Lexphiresok
    sta LORES
.Lexphiresok:
    pla
    bpl .Lexppageok
    sta PAGE2
.Lexppageok:
    plp
    rts
tiny_aux_exp_gate_end:

; Position-independent image, copied to aux0 $4d80 at initialization.
; No absolute code references, no calls out, no self-modification.
; Input d=$04..$06; f=$04/$05, LUT pointer=$06/$07,
; a=$08..$0b, diff=$0c..$0e, fractional product=$0f..$11.
; All data accesses are aux0 LUT or main ZP. Main RAMWRT stays on.
tiny_aux_exp_image:
    lda $05
    sta $07
    lda $06
    lsr a
    ror $07
    lsr a
    ror $07
    tax
    lda $07
    sta $06
    stx $07
    lda $05
    and #3
    sta $05
    lda $06
    ora $07
    beq .Lexpfirst
    ; Offset = 3*(i-1), pointer = $4000+offset.
    lda $06
    bne .Lexpdec
    dec $07
.Lexpdec:
    dec $06
    lda $06
    asl a
    sta $0c
    lda $07
    rol a
    sta $0d
    clc
    lda $06
    adc $0c
    sta $06
    lda $07
    adc $0d
    adc #$40
    sta $07
    ldy #0
    lda ($06),y
    sta $08
    iny
    lda ($06),y
    sta $09
    iny
    lda ($06),y
    sta $0a
    iny
    stz $0b
    bra .Lexpdiff
.Lexpfirst:
    ; Table entry zero is implicit: 2^24, not a 24-bit table entry.
    stz $06
    lda #$40
    sta $07
    stz $08
    stz $09
    stz $0a
    lda #1
    sta $0b
    ldy #0
.Lexpdiff:
    sec
    lda $08
    sbc ($06),y
    sta $0c
    iny
    lda $09
    sbc ($06),y
    sta $0d
    iny
    lda $0a
    sbc ($06),y
    sta $0e
    ; diff fits 18 bits. Process f LSB first: (acc+bit*diff)/2.
    ; Ten iterations produce floor(diff*f/1024). Carry on the final
    ; shift gives the exact half-up rounding bit.
    stz $0f
    stz $10
    stz $11
    ldx #10
.Lexpbit:
    lsr $05
    ror $04
    bcc .Lexpshift
    clc
    lda $0f
    adc $0c
    sta $0f
    lda $10
    adc $0d
    sta $10
    lda $11
    adc $0e
    sta $11
.Lexpshift:
    lsr $11
    ror $10
    ror $0f
    dex
    bne .Lexpbit
    bcc .Lexpsub
    inc $0f
    bne .Lexpsub
    inc $10
    bne .Lexpsub
    inc $11
.Lexpsub:
    sec
    lda $08
    sbc $0f
    sta $08
    lda $09
    sbc $10
    sta $09
    lda $0a
    sbc $11
    sta $0a
    lda $0b
    sbc #0
    sta $0b
    rts
tiny_aux_exp_image_end:
