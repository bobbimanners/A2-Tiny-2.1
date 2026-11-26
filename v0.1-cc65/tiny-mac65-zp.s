; TINY stage 7C - ca65, 65C02.  No cc65 parameter stack calls.
; Globals are provided by C. Non-reentrant, no bank switching.
; dot: 0..511 signed-byte pairs, signed 24-bit internal sum.
; axpy: accum[i] += coefficient * b[i], 32-bit little-endian slots;
;       low 24 bits accumulated, high byte sign-extended each time.
;       caller guarantees <= 128 terms per slot, n <= 128.
.setcpu "65C02"
.export _tiny_mac65_dot, _tiny_mac65_axpy
.import _tiny_mac_a, _tiny_mac_b, _tiny_mac_n, _tiny_mac_result
.import _tiny_mac_coeff, _tiny_mac_acc

; Reuse cc65 caller-clobbered scratch pointers; do not allocate extra ZP.
; No C calls are made while these aliases are live.
.importzp ptr1, ptr2, ptr3
ap = ptr1
bp = ptr2
cp = ptr3

.segment "BSS"
left: .res 2
lhs: .res 1
rhs: .res 1
pl: .res 1
ph: .res 1
acc: .res 3

.segment "RODATA"
; Q[i] = floor((i-256)^2/4). Separate byte planes. No page alignment
; assumed: absolute indexed addressing handles page crossings.
ql:
.repeat 512, I
    .byte <(((I-256)*(I-256))/4)
.endrepeat
qh:
.repeat 512, I
    .byte >(((I-256)*(I-256))/4)
.endrepeat

.segment "CODE"
; Input lhs = signed a XOR $80; A = signed b (raw byte).
; Output pl:ph = signed product, X and flags clobbered; Y unchanged.
; Q[u+v] - Q[u-v+256], u=a+128, v=b+128.
.macro PRODUCT
    .local sum_low, sum_done, diff_low, diff_done
    eor #$80
    sta rhs
    clc
    adc lhs
    tax
    bcc sum_low
    lda ql+256,x
    sta pl
    lda qh+256,x
    bra sum_done
sum_low:
    lda ql,x
    sta pl
    lda qh,x
sum_done:
    sta ph
    lda lhs
    sec
    sbc rhs
    tax
    bcc diff_low
    sec
    lda pl
    sbc ql+256,x
    sta pl
    lda ph
    sbc qh+256,x
    bra diff_done
 diff_low:
    sec
    lda pl
    sbc ql,x
    sta pl
    lda ph
    sbc qh,x
 diff_done:
    sta ph
.endmacro

.macro LOAD_B_N
    lda _tiny_mac_b
    sta bp
    lda _tiny_mac_b+1
    sta bp+1
    lda _tiny_mac_n
    sta left
    lda _tiny_mac_n+1
    sta left+1
.endmacro

; Decrement 16-bit count; Z set iff count becomes zero.
.macro DEC_COUNT
    lda left
    bne :+
    dec left+1
:
    dec left
    lda left
    ora left+1
.endmacro

.proc _tiny_mac65_dot
    cld
    LOAD_B_N
    lda _tiny_mac_a
    sta ap
    lda _tiny_mac_a+1
    sta ap+1
    stz acc
    stz acc+1
    stz acc+2
    ldy #0
    lda left
    ora left+1
    bne :+
    jmp done
:
loop:
    lda (ap),y
    eor #$80
    sta lhs
    lda (bp),y
    PRODUCT
    clc
    lda acc
    adc pl
    sta acc
    lda acc+1
    adc ph
    sta acc+1
    ; Preserve carry from the high-byte add while extending the product.
    ldx #0
    lda ph
    bpl :+
    dex
:
    txa
    adc acc+2
    sta acc+2
    iny
    bne :+
    inc ap+1
    inc bp+1
:
    DEC_COUNT
    beq done
    jmp loop
 done:
    lda acc
    sta _tiny_mac_result
    lda acc+1
    sta _tiny_mac_result+1
    lda acc+2
    sta _tiny_mac_result+2
    ldx #0
    ; sign extend 24-bit sum to the C int32_t result
    lda acc+2
    bpl :+
    dex
:
    stx _tiny_mac_result+3
    rts
.endproc

.proc _tiny_mac65_axpy
    cld
    LOAD_B_N
    lda _tiny_mac_acc
    sta cp
    lda _tiny_mac_acc+1
    sta cp+1
    lda _tiny_mac_coeff
    eor #$80
    sta lhs
    lda left
    ora left+1
    bne :+
    jmp done
:
loop:
    ldy #0
    lda (bp),y
    PRODUCT
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
    bpl :+
    dex
:
    txa
    adc (cp),y
    sta (cp),y
    ; Extend the *new* accumulated result into its fourth byte.
    ldx #0
    lda (cp),y
    bpl :+
    dex
:
    iny
    txa
    sta (cp),y
    inc bp
    bne :+
    inc bp+1
:
    clc
    lda cp
    adc #4
    sta cp
    bcc :+
    inc cp+1
:
    DEC_COUNT
    ; Loop body is longer than a relative branch on some layouts.
    beq done
    jmp loop
 done:
    rts
.endproc
