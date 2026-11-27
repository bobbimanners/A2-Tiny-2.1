; TINY stage 11O - prepared matrix MAC, plus general 11K kernels.
; Dot uses self-modifying absolute,Y loads and an immediate loop limit.
; Requires writable main-memory code; not reentrant. No new ZP reservation.
; Globals are provided by C. Non-reentrant, no bank switching.
; dot: 0..511 signed-byte pairs, signed 24-bit internal sum.
; axpy: accum[i] += coefficient * b[i], 32-bit little-endian slots;
;       low 24 bits accumulated, high byte sign-extended each time.
;       caller guarantees <= 128 terms per slot, n <= 128.
.global tiny_mac65_dot, tiny_mac65_axpy
.global tiny_mac65_prepare, tiny_mac65_prepared_dot
.global tiny_mac65_aux_dot, tiny_mac65_aux_end

; Private zero page, reserved by -mreserve-zp=6.
.section .zp.bss,"aw",@nobits
ap: .space 2
bp: .space 2
cp: .space 2

; Apple II LLVM-MOS maps RC2..RC19 to $02..$13; all are caller-saved.
; These leaf routines may use them without saving or reserving more ZP.
left = $08
lhs = $0a
rhs = $0b
pl = $0c
ph = $0d
acc = $0e

; Shared lookup planes are defined in tiny-qsquare-mos.s.
.section .text,"ax",@progbits
; Input lhs = signed a XOR $80; A = signed b (raw byte).
; Output pl:A = signed product, X and flags clobbered; Y unchanged.
; Q[u+v] - Q[u-v+256], u=a+128, v=b+128.
.macro PRODUCT
    eor #$80
    sta rhs
    clc
    adc lhs
    tax
    bcc 1f
    lda ql+256,x
    sta pl
    lda qh+256,x
    bra 2f
1:
    lda ql,x
    sta pl
    lda qh,x
2:
    sta ph
    lda lhs
    sec
    sbc rhs
    tax
    bcc 3f
    ; Carry already set by the index subtraction.
    lda pl
    sbc ql+256,x
    sta pl
    lda ph
    sbc qh+256,x
    bra 4f
3:
    sec
    lda pl
    sbc ql,x
    sta pl
    lda ph
    sbc qh,x
4:
.endm

.macro LOAD_B_N
    lda tiny_mac_b
    sta bp
    lda tiny_mac_b+1
    sta bp+1
    lda tiny_mac_n
    sta left
    lda tiny_mac_n+1
    sta left+1
.endm

; Decrement 16-bit count; Z set iff count becomes zero.
.macro DEC_COUNT
    lda left
    bne 9f
    dec left+1
9:
    dec left
    lda left
    ora left+1
.endm

tiny_mac65_dot:
    cld
    stz acc
    stz acc+1
    stz acc+2
    lda tiny_mac_n
    ora tiny_mac_n+1
    bne .Ldot_setup
    jmp .Ltiny_mac65_dot_done
.Ldot_setup:
    lda tiny_mac_a
    sta .Ldot_a+1
    lda tiny_mac_a+1
    sta .Ldot_a+2
    lda tiny_mac_b
    sta .Ldot_b+1
    lda tiny_mac_b+1
    sta .Ldot_b+2
    ; Process full 256-byte blocks first (limit=0), then the remainder.
    ; Input contract remains 0..511, with callers splitting longer dots.
    lda tiny_mac_n+1
    sta left+1
    beq .Ldot_partial
    lda #0
    bra .Ldot_set_limit
.Ldot_partial:
    lda tiny_mac_n
.Ldot_set_limit:
    sta .Ldot_limit+1
    ldy #0
.Ltiny_mac65_dot_loop:
.Ldot_a:
    lda $ffff,y
    eor #$80
    sta lhs
.Ldot_b:
    lda $ffff,y
    PRODUCT
    ; Bias each signed product by $8000, accumulate as unsigned.
    ; The sum wraps mod 2^24; remove n*$8000 once at the end.
    eor #$80
    tax
    clc
    lda pl
    adc acc
    sta acc
    txa
    adc acc+1
    sta acc+1
    bcc 9f
    inc acc+2
9:
    iny
.Ldot_limit:
    cpy #0
    bne .Ltiny_mac65_dot_loop
    lda left+1
    beq .Ltiny_mac65_dot_done
    dec left+1
    lda tiny_mac_n
    beq .Ltiny_mac65_dot_done
    inc .Ldot_a+2
    inc .Ldot_b+2
    sta .Ldot_limit+1
    jmp .Ltiny_mac65_dot_loop
.Ltiny_mac65_dot_done:
    ; n*$8000 mod 2^24 has bytes [0, (n&1)*128, n>>1].
    lda tiny_mac_n+1
    lsr a
    lda tiny_mac_n
    ror a
    sta ph
    lda #0
    ror a
    sta pl
    sec
    lda acc+1
    sbc pl
    sta acc+1
    lda acc+2
    sbc ph
    sta acc+2
    lda acc
    sta tiny_mac_result
    lda acc+1
    sta tiny_mac_result+1
    lda acc+2
    sta tiny_mac_result+2
    ldx #0
    ; sign extend 24-bit sum to the C int32_t result
    lda acc+2
    bpl 9f
    dex
9:
    stx tiny_mac_result+3
    rts


tiny_mac65_axpy:
    cld
    LOAD_B_N
    lda tiny_mac_acc
    sta cp
    lda tiny_mac_acc+1
    sta cp+1
    lda tiny_mac_coeff
    eor #$80
    sta lhs
    lda left
    ora left+1
    bne 9f
    jmp .Ltiny_mac65_axpy_done
9:
.Ltiny_mac65_axpy_loop:
    ldy #0
    lda (bp),y
    PRODUCT
    sta ph
    clc
    lda (cp),y
    adc pl
    sta (cp),y
    iny
    lda (cp),y
    adc ph
    sta (cp),y
    iny
    ldx #0
    lda ph
    bpl 9f
    dex
9:
    txa
    adc (cp),y
    sta (cp),y
    ; Extend the *new* accumulated result into its fourth byte.
    ldx #0
    lda (cp),y
    bpl 9f
    dex
9:
    iny
    txa
    sta (cp),y
    inc bp
    bne 9f
    inc bp+1
9:
    clc
    lda cp
    adc #4
    sta cp
    bcc 9f
    inc cp+1
9:
    DEC_COUNT
    ; Loop body is longer than a relative branch on some layouts.
    beq .Ltiny_mac65_axpy_done
    jmp .Ltiny_mac65_axpy_loop
 .Ltiny_mac65_axpy_done:
    rts


; Matrix-only fast path. Prepare once per projection, then call prepared_dot
; for each row. tiny_mac_a = fixed weight buffer, tiny_mac_b = writable
; 512-byte activation workspace, tiny_mac_n = 1..256. q8 guarantees signed
; activations -127..127 (weights may be -128). Prepare replaces the first
; plane with u=a+128 and writes 256-u to the second plane. Both fit in a
; byte. Q[u+v]-Q[256-u+v] == a*b for v=b+128, by Q symmetry.
; All prepared state survives unrelated routines in patched code and the
; workspace. Do not change tiny_mac_n or run prepare again until done.
tiny_mac65_prepare:
    cld
    lda tiny_mac_a
    sta .Lprepared_weight+1
    lda tiny_mac_a+1
    sta .Lprepared_weight+2
    lda tiny_mac_b
    sta bp
    sta cp
    sta .Lprepared_pos+1
    sta .Lprepared_neg+1
    lda tiny_mac_b+1
    sta bp+1
    sta .Lprepared_pos+2
    inc a
    sta cp+1
    sta .Lprepared_neg+2
    lda tiny_mac_n
    sta .Lprepared_limit+1
    ldy #0
.Lprepare_loop:
    lda (bp),y
    eor #$80
    sta (bp),y
    eor #$ff
    inc a
    sta (cp),y
    iny
    cpy tiny_mac_n
    bne .Lprepare_loop
    rts

tiny_mac65_prepared_dot:
    jsr .Lprepared_body
    jmp .Ltiny_mac65_dot_done
.Lprepared_body:
    cld
    stz $0e
    stz $0f
    stz $10
    ldy #0
.Lprepared_loop:
.Lprepared_pos:
    lda $ffff,y
    sta .Lprepared_pluslo+1
    sta .Lprepared_plushi+1
.Lprepared_neg:
    lda $ffff,y
    sta .Lprepared_minuslo+1
    sta .Lprepared_minushi+1
.Lprepared_weight:
    lda $ffff,y
    eor #$80
    tax
    sec
.Lprepared_pluslo:
    lda ql,x
.Lprepared_minuslo:
    sbc ql,x
    sta $0c
.Lprepared_plushi:
    lda qh,x
.Lprepared_minushi:
    sbc qh,x
    eor #$80
    tax
    clc
    lda $0c
    adc $0e
    sta $0e
    txa
    adc $0f
    sta $0f
    bcc 9f
    inc $10
9:
    iny
.Lprepared_limit:
    cpy #0
    bne .Lprepared_loop
    rts

; Direct matrix row: tiny_mac_a = aux record address (scale then weights),
; tiny_mac_b = main destination of 4-byte scale, tiny_mac_coeff = bank.
; Prepare the vector first in tiny_mac65_work. All code and the tables
; and prepared vector must reside in protected main HGR. Main ZP/stack,
; RAMRD/RAMWRT main and bank 0 at entry. Restore mapping before result
; correction accesses C globals. No C calls while aux reads enabled.
tiny_mac65_aux_dot:
    php
    sei
    cld
    lda $C01C
    pha
    lda $C01D
    pha
    lda $C018
    pha
    sta $C054
    sta $C057
    sta $C001
    lda tiny_mac_a
    sta ap
    clc
    adc #4
    sta .Lprepared_weight+1
    lda tiny_mac_a+1
    sta ap+1
    adc #0
    sta .Lprepared_weight+2
    lda tiny_mac_b
    sta bp
    lda tiny_mac_b+1
    sta bp+1
    lda tiny_mac_coeff
    sta $C073
    sta $C003
    ldy #3
.Laux_scale:
    lda (ap),y
    sta (bp),y
    dey
    bpl .Laux_scale
    jsr .Lprepared_body
    sta $C002
    stz $C073
    pla
    bmi .Laux_storeok
    sta $C000
.Laux_storeok:
    pla
    bmi .Laux_hiresok
    sta $C056
.Laux_hiresok:
    pla
    bpl .Laux_pageok
    sta $C055
.Laux_pageok:
    jsr .Ltiny_mac65_dot_done
    plp
    rts
tiny_mac65_aux_end:

