; Stage 10C: LLVM-MOS Apple II 65C02 wide arithmetic.
; ABI: pointer args in rc2/3, rc4/5, rc6/7; int args/returns A:X.
; Apple2 maps rc0..31 to $00..$1f. No callee-saved registers touched.
; D must be clear (normal C environment). No IRQ/reentrant use.
; Wide = 8 little-endian bytes. Exact aliases r==a or r==b supported.
R = $02
P = $04
B = $06
.section .bss,"aw",@nobits
.Lq: .space 8
.La: .space 8
.Lb: .space 8
.Ln: .space 1
.Lbits: .space 1
.Louter: .space 1
.Linner: .space 1
.Lprod: .space 1
.Lsign: .space 1
.Lround: .space 1
.section .text,"ax",@progbits
.global wa, ws, wneg, wmag, winc, wz, wn, wc, wcs, wl, wr, wbit, topbit, wm
.global tiny_wdiv, tiny_wrounddiv
; Binary operations. Carry preserved by INY/DEX/BNE.
wa:
    ldy #0
    ldx #8
    clc
.Ladd:
    lda (P),y
    adc (B),y
    sta (R),y
    iny
    dex
    bne .Ladd
    rts
ws:
    ldy #0
    ldx #8
    sec
.Lsub:
    lda (P),y
    sbc (B),y
    sta (R),y
    iny
    dex
    bne .Lsub
    rts
wmag:
    ldy #7
    lda (P),y
    bmi wneg
.Lcopy:
    ldy #7
.Lcopyloop:
    lda (P),y
    sta (R),y
    dey
    bpl .Lcopyloop
    rts
wneg:
    ldy #0
    ldx #8
    sec
.Lneg:
    lda #0
    sbc (P),y
    sta (R),y
    iny
    dex
    bne .Lneg
    rts
winc:
    ldy #0
.Linc:
    lda (R),y
    inc a
    sta (R),y
    bne .Lret
    iny
    cpy #8
    bne .Linc
.Lret:
    rts
wz:
    ldy #7
    lda #0
.Lzero:
    ora (R),y
    dey
    bpl .Lzero
    cmp #0
    bne .Lfalse
    bra .Ltrue
wn:
    ldy #7
    lda (R),y
    bmi .Ltrue
.Lfalse:
    lda #0
    tax
    rts
.Ltrue:
    lda #1
    ldx #0
    rts
wcs:
    ldy #7
    lda (R),y
    eor (P),y
    bpl wc
    lda (R),y
    bmi .Lless
    bra .Ltrue
wc:
    ldy #7
.Lcmp:
    lda (R),y
    cmp (P),y
    bcc .Lless
    bne .Ltrue
    dey
    bpl .Lcmp
    bra .Lfalse
.Lless:
    lda #$ff
    tax
    rts
; topbit(0) = 0, otherwise floor(log2(a)).
topbit:
    ldy #7
.Ltopscan:
    lda (R),y
    bne .Ltopfound
    dey
    bpl .Ltopscan
    bra .Lfalse
.Ltopfound:
    ldx #0
.Ltoploop:
    lsr a
    beq .Ltopdone
    inx
    bra .Ltoploop
.Ltopdone:
    stx .Ln
    tya
    asl a
    asl a
    asl a
    ora .Ln
    ldx #0
    rts
; wbit(a,n) : pointer R, signed int n in AX.
wbit:
    cpx #0
    bne .Lbitzero
    cmp #64
    bcs .Lbitzero
    pha
    lsr a
    lsr a
    lsr a
    tay
    pla
    and #7
    tax
    lda (R),y
    cpx #0
    beq .Lbitdone
.Lbitloop:
    lsr a
    dex
    bne .Lbitloop
.Lbitdone:
    and #1
    ldx #0
    rts
.Lbitzero:
    lda #0
    tax
    rts
; Shift by n <= 0 copies, n >= 64 clears. Move whole bytes first;
; at most seven single-bit passes. Intermediate scratch handles aliases.
wl:
    ldy #0
    bra .Lshift
wr:
    ldy #1
.Lshift:
    sty .Lsign
    cpx #$80
    bcc .Lshiftpositive
    jmp .Lcopy
.Lshiftpositive:
    cpx #0
    beq .Lshiftlow
    jmp .Lshiftzero
.Lshiftlow:
    cmp #0
    bne .Lshiftnonzero
    jmp .Lcopy
.Lshiftnonzero:
    cmp #64
    bcc .Lshiftvalid
    jmp .Lshiftzero
.Lshiftvalid:
    pha
    and #7
    sta .Lbits
    pla
    lsr a
    lsr a
    lsr a
    sta .Ln
    jsr .Lclearq
    lda .Lsign
    bne .Lrightbytes
    ldx .Ln
    ldy #0
.Lleftbytes:
    lda (P),y
    sta .Lq,x
    iny
    inx
    cpx #8
    bne .Lleftbytes
    ldx .Lbits
    beq .Lshiftout
.Lleftbits:
    asl .Lq
    rol .Lq+1
    rol .Lq+2
    rol .Lq+3
    rol .Lq+4
    rol .Lq+5
    rol .Lq+6
    rol .Lq+7
    dex
    bne .Lleftbits
    bra .Lshiftout
.Lrightbytes:
    ldy .Ln
    ldx #0
.Lrightcopy:
    lda (P),y
    sta .Lq,x
    iny
    inx
    cpy #8
    bne .Lrightcopy
    ldx .Lbits
    beq .Lshiftout
.Lrightbits:
    lsr .Lq+7
    ror .Lq+6
    ror .Lq+5
    ror .Lq+4
    ror .Lq+3
    ror .Lq+2
    ror .Lq+1
    ror .Lq
    dex
    bne .Lrightbits
    bra .Lshiftout
.Lshiftzero:
    jsr .Lclearq
.Lshiftout:
    jmp .Lstoreq
.Lcopyjump:
    jmp .Lcopy
; Common scratch helpers.
.Lclearq:
    ldx #7
.Lclearqloop:
    stz .Lq,x
    dex
    bpl .Lclearqloop
    rts
.Lstoreq:
    ldy #7
.Lstoreloop:
    lda .Lq,y
    sta (R),y
    dey
    bpl .Lstoreloop
    rts
.Lloadab:
    ldy #7
.Lloadloop:
    lda (P),y
    sta .La,y
    lda (B),y
    sta .Lb,y
    dey
    bpl .Lloadloop
    rts
; Modulo 2^64 schoolbook multiplication. Magnitudes avoid eight $ff
; bytes for small signed negative inputs. Zero bytes skipped; truncate
; products beyond byte 7. 8x8 uses compact shift/rotate multiplication.
wm:
    jsr .Lloadab
    jsr .Lclearq
    lda .La+7
    eor .Lb+7
    sta .Lsign
    lda .La+7
    bpl .Lapos
    sec
    lda #0
    sbc .La+0
    sta .La+0
    lda #0
    sbc .La+1
    sta .La+1
    lda #0
    sbc .La+2
    sta .La+2
    lda #0
    sbc .La+3
    sta .La+3
    lda #0
    sbc .La+4
    sta .La+4
    lda #0
    sbc .La+5
    sta .La+5
    lda #0
    sbc .La+6
    sta .La+6
    lda #0
    sbc .La+7
    sta .La+7
.Lapos:
    lda .Lb+7
    bpl .Lbpos
    sec
    lda #0
    sbc .Lb+0
    sta .Lb+0
    lda #0
    sbc .Lb+1
    sta .Lb+1
    lda #0
    sbc .Lb+2
    sta .Lb+2
    lda #0
    sbc .Lb+3
    sta .Lb+3
    lda #0
    sbc .Lb+4
    sta .Lb+4
    lda #0
    sbc .Lb+5
    sta .Lb+5
    lda #0
    sbc .Lb+6
    sta .Lb+6
    lda #0
    sbc .Lb+7
    sta .Lb+7
.Lbpos:
    stz .Louter
.Lmulouter:
    ldx .Louter
    lda .Lb,x
    beq .Lnextouter
    stz .Linner
.Lmulinner:
    ldx .Linner
    lda .La,x
    beq .Lnextinner
    ldx .Louter
    lda .Lb,x
    sta .Lprod
    ldx .Linner
    ldy #8
    lda #0
    lsr .Lprod
.Lmul8:
    bcc .Lmulnoadd
    clc
    adc .La,x
.Lmulnoadd:
    ror a
    ror .Lprod
    dey
    bne .Lmul8
    pha
    txa
    clc
    adc .Louter
    tax
    lda .Lprod
    clc
    adc .Lq,x
    sta .Lq,x
    inx
    pla
    ; Need retain carry across bounds check: use Y (remaining count).
    ; X==8 means high byte is outside result; compare via TXA would
    ; destroy A, CPX would destroy carry. Save carry with PHP/PLP.
    php
    cpx #8
    beq .Ldiscardhigh
    plp
    adc .Lq,x
    sta .Lq,x
    bcc .Lnextinner
.Lpropagate:
    inx
    cpx #8
    beq .Lnextinner
    inc .Lq,x
    beq .Lpropagate
    bra .Lnextinner
.Ldiscardhigh:
    plp
.Lnextinner:
    inc .Linner
    lda .Linner
    clc
    adc .Louter
    cmp #8
    bcc .Lmulinner
.Lnextouter:
    inc .Louter
    lda .Louter
    cmp #8
    bcs .Lmulend
    jmp .Lmulouter
.Lmulend:
    bit .Lsign
    bpl .Lmulout
    sec
    lda #0
    sbc .Lq+0
    sta .Lq+0
    lda #0
    sbc .Lq+1
    sta .Lq+1
    lda #0
    sbc .Lq+2
    sta .Lq+2
    lda #0
    sbc .Lq+3
    sta .Lq+3
    lda #0
    sbc .Lq+4
    sta .Lq+4
    lda #0
    sbc .Lq+5
    sta .Lq+5
    lda #0
    sbc .Lq+6
    sta .Lq+6
    lda #0
    sbc .Lq+7
    sta .Lq+7
.Lmulout:
    jmp .Lstoreq
; Unsigned divide. C wrapper checks divisor != 0. Inputs copied before
; writes; quotient in q, remainder in a. Rounded entry returns nearest,
; ties upward, directly using remainder (no second multiplication).
tiny_wdiv:
    stz .Lround
    bra .Ldivstart
tiny_wrounddiv:
    lda #1
    sta .Lround
.Ldivstart:
    ldy #7
.Ldivload:
    lda (P),y
    sta .Lq,y
    lda (B),y
    sta .Lb,y
    lda #0
    sta .La,y
    dey
    bpl .Ldivload
    lda .Lb+1
    ora .Lb+2
    ora .Lb+3
    ora .Lb+4
    ora .Lb+5
    ora .Lb+6
    ora .Lb+7
    bne .Ldivgeneral
    ; Fast one-byte divisor (e.g. 127): A holds running remainder.
    ldx #7
    lda #0
.Ldiv8byte:
    ldy #8
.Ldiv8bit:
    asl .Lq,x
    rol a
    bcs .Ldiv8sub
    cmp .Lb
    bcc .Ldiv8next
.Ldiv8sub:
    sec
    sbc .Lb
    inc .Lq,x
.Ldiv8next:
    dey
    bne .Ldiv8bit
    dex
    bpl .Ldiv8byte
    sta .La
    jmp .Ldivfinish
.Ldivgeneral:
    lda #64
    sta .Ln
.Ldivbit:
    asl .Lq
    rol .Lq+1
    rol .Lq+2
    rol .Lq+3
    rol .Lq+4
    rol .Lq+5
    rol .Lq+6
    rol .Lq+7
    rol .La
    rol .La+1
    rol .La+2
    rol .La+3
    rol .La+4
    rol .La+5
    rol .La+6
    rol .La+7
    bcs .Ldivsub
    ldx #7
.Ldivcmp:
    lda .La,x
    cmp .Lb,x
    bcc .Ldivnext
    bne .Ldivsub
    dex
    bpl .Ldivcmp
.Ldivsub:
    ldx #0
    ldy #8
    sec
.Ldivsubloop:
    lda .La,x
    sbc .Lb,x
    sta .La,x
    inx
    dey
    bne .Ldivsubloop
    inc .Lq
.Ldivnext:
    dec .Ln
    beq .Ldivfinish
    jmp .Ldivbit
.Ldivfinish:
    lda .Lround
    beq .Ldivout
    ; Round up iff rem >= divisor-rem, equivalent to 2*rem >= divisor
    ; without overflow. Compute divisor-rem in b, compare with a.
    ldx #0
    ldy #8
    sec
.Lroundsub:
    lda .Lb,x
    sbc .La,x
    sta .Lb,x
    inx
    dey
    bne .Lroundsub
    ldx #7
.Lroundcmp:
    lda .La,x
    cmp .Lb,x
    bcc .Ldivout
    bne .Lroundup
    dex
    bpl .Lroundcmp
.Lroundup:
    ldx #0
.Lroundinc:
    inc .Lq,x
    bne .Ldivout
    inx
    cpx #8
    bne .Lroundinc
.Ldivout:
    jmp .Lstoreq
