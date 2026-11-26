; TINY stage 11A - faster LLVM-MOS 65C02 MAC kernels.
; Dot uses self-modifying absolute,Y loads and an immediate loop limit.
; Requires writable main-memory code; not reentrant. No new ZP reservation.
; Globals are provided by C. Non-reentrant, no bank switching.
; dot: 0..511 signed-byte pairs, signed 24-bit internal sum.
; axpy: accum[i] += coefficient * b[i], 32-bit little-endian slots;
;       low 24 bits accumulated, high byte sign-extended each time.
;       caller guarantees <= 128 terms per slot, n <= 128.
.global tiny_mac65_dot, tiny_mac65_axpy

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

.section .rodata,"a",@progbits
; Q[i] = floor((i-256)^2/4). Separate byte planes.
; Align both planes and their +256 halves for constant 4-cycle table loads.
.p2align 8
ql:
    .byte $00,$80,$01,$82,$04,$86,$09,$8c,$10,$94,$19,$9e,$24,$aa,$31,$b8
    .byte $40,$c8,$51,$da,$64,$ee,$79,$04,$90,$1c,$a9,$36,$c4,$52,$e1,$70
    .byte $00,$90,$21,$b2,$44,$d6,$69,$fc,$90,$24,$b9,$4e,$e4,$7a,$11,$a8
    .byte $40,$d8,$71,$0a,$a4,$3e,$d9,$74,$10,$ac,$49,$e6,$84,$22,$c1,$60
    .byte $00,$a0,$41,$e2,$84,$26,$c9,$6c,$10,$b4,$59,$fe,$a4,$4a,$f1,$98
    .byte $40,$e8,$91,$3a,$e4,$8e,$39,$e4,$90,$3c,$e9,$96,$44,$f2,$a1,$50
    .byte $00,$b0,$61,$12,$c4,$76,$29,$dc,$90,$44,$f9,$ae,$64,$1a,$d1,$88
    .byte $40,$f8,$b1,$6a,$24,$de,$99,$54,$10,$cc,$89,$46,$04,$c2,$81,$40
    .byte $00,$c0,$81,$42,$04,$c6,$89,$4c,$10,$d4,$99,$5e,$24,$ea,$b1,$78
    .byte $40,$08,$d1,$9a,$64,$2e,$f9,$c4,$90,$5c,$29,$f6,$c4,$92,$61,$30
    .byte $00,$d0,$a1,$72,$44,$16,$e9,$bc,$90,$64,$39,$0e,$e4,$ba,$91,$68
    .byte $40,$18,$f1,$ca,$a4,$7e,$59,$34,$10,$ec,$c9,$a6,$84,$62,$41,$20
    .byte $00,$e0,$c1,$a2,$84,$66,$49,$2c,$10,$f4,$d9,$be,$a4,$8a,$71,$58
    .byte $40,$28,$11,$fa,$e4,$ce,$b9,$a4,$90,$7c,$69,$56,$44,$32,$21,$10
    .byte $00,$f0,$e1,$d2,$c4,$b6,$a9,$9c,$90,$84,$79,$6e,$64,$5a,$51,$48
    .byte $40,$38,$31,$2a,$24,$1e,$19,$14,$10,$0c,$09,$06,$04,$02,$01,$00
    .byte $00,$00,$01,$02,$04,$06,$09,$0c,$10,$14,$19,$1e,$24,$2a,$31,$38
    .byte $40,$48,$51,$5a,$64,$6e,$79,$84,$90,$9c,$a9,$b6,$c4,$d2,$e1,$f0
    .byte $00,$10,$21,$32,$44,$56,$69,$7c,$90,$a4,$b9,$ce,$e4,$fa,$11,$28
    .byte $40,$58,$71,$8a,$a4,$be,$d9,$f4,$10,$2c,$49,$66,$84,$a2,$c1,$e0
    .byte $00,$20,$41,$62,$84,$a6,$c9,$ec,$10,$34,$59,$7e,$a4,$ca,$f1,$18
    .byte $40,$68,$91,$ba,$e4,$0e,$39,$64,$90,$bc,$e9,$16,$44,$72,$a1,$d0
    .byte $00,$30,$61,$92,$c4,$f6,$29,$5c,$90,$c4,$f9,$2e,$64,$9a,$d1,$08
    .byte $40,$78,$b1,$ea,$24,$5e,$99,$d4,$10,$4c,$89,$c6,$04,$42,$81,$c0
    .byte $00,$40,$81,$c2,$04,$46,$89,$cc,$10,$54,$99,$de,$24,$6a,$b1,$f8
    .byte $40,$88,$d1,$1a,$64,$ae,$f9,$44,$90,$dc,$29,$76,$c4,$12,$61,$b0
    .byte $00,$50,$a1,$f2,$44,$96,$e9,$3c,$90,$e4,$39,$8e,$e4,$3a,$91,$e8
    .byte $40,$98,$f1,$4a,$a4,$fe,$59,$b4,$10,$6c,$c9,$26,$84,$e2,$41,$a0
    .byte $00,$60,$c1,$22,$84,$e6,$49,$ac,$10,$74,$d9,$3e,$a4,$0a,$71,$d8
    .byte $40,$a8,$11,$7a,$e4,$4e,$b9,$24,$90,$fc,$69,$d6,$44,$b2,$21,$90
    .byte $00,$70,$e1,$52,$c4,$36,$a9,$1c,$90,$04,$79,$ee,$64,$da,$51,$c8
    .byte $40,$b8,$31,$aa,$24,$9e,$19,$94,$10,$8c,$09,$86,$04,$82,$01,$80
qh:
    .byte $40,$3f,$3f,$3e,$3e,$3d,$3d,$3c,$3c,$3b,$3b,$3a,$3a,$39,$39,$38
    .byte $38,$37,$37,$36,$36,$35,$35,$35,$34,$34,$33,$33,$32,$32,$31,$31
    .byte $31,$30,$30,$2f,$2f,$2e,$2e,$2d,$2d,$2d,$2c,$2c,$2b,$2b,$2b,$2a
    .byte $2a,$29,$29,$29,$28,$28,$27,$27,$27,$26,$26,$25,$25,$25,$24,$24
    .byte $24,$23,$23,$22,$22,$22,$21,$21,$21,$20,$20,$1f,$1f,$1f,$1e,$1e
    .byte $1e,$1d,$1d,$1d,$1c,$1c,$1c,$1b,$1b,$1b,$1a,$1a,$1a,$19,$19,$19
    .byte $19,$18,$18,$18,$17,$17,$17,$16,$16,$16,$15,$15,$15,$15,$14,$14
    .byte $14,$13,$13,$13,$13,$12,$12,$12,$12,$11,$11,$11,$11,$10,$10,$10
    .byte $10,$0f,$0f,$0f,$0f,$0e,$0e,$0e,$0e,$0d,$0d,$0d,$0d,$0c,$0c,$0c
    .byte $0c,$0c,$0b,$0b,$0b,$0b,$0a,$0a,$0a,$0a,$0a,$09,$09,$09,$09,$09
    .byte $09,$08,$08,$08,$08,$08,$07,$07,$07,$07,$07,$07,$06,$06,$06,$06
    .byte $06,$06,$05,$05,$05,$05,$05,$05,$05,$04,$04,$04,$04,$04,$04,$04
    .byte $04,$03,$03,$03,$03,$03,$03,$03,$03,$02,$02,$02,$02,$02,$02,$02
    .byte $02,$02,$02,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01
    .byte $01,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
    .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
    .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
    .byte $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
    .byte $01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$01,$02,$02
    .byte $02,$02,$02,$02,$02,$02,$02,$02,$03,$03,$03,$03,$03,$03,$03,$03
    .byte $04,$04,$04,$04,$04,$04,$04,$04,$05,$05,$05,$05,$05,$05,$05,$06
    .byte $06,$06,$06,$06,$06,$07,$07,$07,$07,$07,$07,$08,$08,$08,$08,$08
    .byte $09,$09,$09,$09,$09,$09,$0a,$0a,$0a,$0a,$0a,$0b,$0b,$0b,$0b,$0c
    .byte $0c,$0c,$0c,$0c,$0d,$0d,$0d,$0d,$0e,$0e,$0e,$0e,$0f,$0f,$0f,$0f
    .byte $10,$10,$10,$10,$11,$11,$11,$11,$12,$12,$12,$12,$13,$13,$13,$13
    .byte $14,$14,$14,$15,$15,$15,$15,$16,$16,$16,$17,$17,$17,$18,$18,$18
    .byte $19,$19,$19,$19,$1a,$1a,$1a,$1b,$1b,$1b,$1c,$1c,$1c,$1d,$1d,$1d
    .byte $1e,$1e,$1e,$1f,$1f,$1f,$20,$20,$21,$21,$21,$22,$22,$22,$23,$23
    .byte $24,$24,$24,$25,$25,$25,$26,$26,$27,$27,$27,$28,$28,$29,$29,$29
    .byte $2a,$2a,$2b,$2b,$2b,$2c,$2c,$2d,$2d,$2d,$2e,$2e,$2f,$2f,$30,$30
    .byte $31,$31,$31,$32,$32,$33,$33,$34,$34,$35,$35,$35,$36,$36,$37,$37
    .byte $38,$38,$39,$39,$3a,$3a,$3b,$3b,$3c,$3c,$3d,$3d,$3e,$3e,$3f,$3f

.section .text,"ax",@progbits
; Input lhs = signed a XOR $80; A = signed b (raw byte).
; Output pl:ph = signed product, X and flags clobbered; Y unchanged.
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
    sec
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
    sta ph
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
    clc
    lda acc
    adc pl
    sta acc
    lda acc+1
    adc ph
    sta acc+1
    ; Extend signed product into the 24-bit sum, preserving ADC carry.
    ldx #0
    lda ph
    bpl 9f
    dex
9:
    txa
    adc acc+2
    sta acc+2
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

