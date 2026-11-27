; Stage 11I: LLVM-MOS Apple II 65C02 wide arithmetic.
; ABI: pointer args in rc2/3, rc4/5, rc6/7; int args/returns A:X.
; Apple2 maps rc0..31 to $00..$1f. No callee-saved registers touched.
; D must be clear (normal C environment). No IRQ/reentrant use.
; Wide = 8 little-endian bytes. Exact aliases r==a or r==b supported.
R = $02
P = $04
B = $06
; Private wm temporaries, in caller-saved imaginary registers.
; R/P/B ($02..$07) are preserved until .Lstoreq.
MO = $08
MP = $09
MA = $0a
MB = $0b
MH = $0c
MT = $0d
MLA = $0e
MLB = $0f
MLIM = $10
.section .bss,"aw",@nobits
.Lq: .space 8
.La: .space 8
.Lb: .space 8
.Ln: .space 1
.Lbits: .space 1
.Lsign: .space 1
.Lround: .space 1
.section .text,"ax",@progbits
.global wa, ws, wneg, wmag, winc, wz, wn, wc, wcs, wl, wr, wbit, topbit, wm
.global tiny_wdiv, tiny_wrounddiv, tiny_rescale
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
; products beyond byte 7. Use the shared MAC quarter-square planes
; for exact unsigned byte products; skip leading zero bytes once.
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
    ldy #7
    ldx #7
    jsr .Lmul_unsigned
    jmp .Lsignedout
; Multiply magnitudes in La/Lb into cleared Lq.
; Y and X specify highest possible nonzero byte of A and B.
; R/P/B and Lsign are preserved. A zero operand leaves Lq zero.
.Lmul_unsigned:
.LscanA:
    lda .La,y
    bne .LfoundA
    dey
    bpl .LscanA
    rts
.LfoundA:
    iny
    sty MLA
    txa
    tay
.LscanB:
    lda .Lb,y
    bne .LfoundB
    dey
    bpl .LscanB
    rts
.LfoundB:
    iny
    sty MLB
    stz MO
.Lmulouter:
    ldx MO
    lda .Lb,x
    bne .Louter_nonzero
    jmp .Lnextouter
.Louter_nonzero:
    sta MB
    ; Don't visit terms whose output index is >= 8.
    lda #8
    sec
    sbc MO
    cmp MLA
    bcc .Lsetlimit
    lda MLA
.Lsetlimit:
    sta MLIM
    ldy #0
.Lmulinner:
    lda .La,y
    beq .Lnextinner
    sta MA
    ; Q[256+t] = floor(t*t/4), for 0 <= t <= 255.
    ; s = a+b, t=s mod 256. For s>=256 add 128*t+16384.
    clc
    adc MB
    tax
    bcs .Lsumcarry
    lda ql+256,x
    sta MP
    lda qh+256,x
    bra .Lsumdone
.Lsumcarry:
    txa
    lsr a
    sta MT
    lda #0
    ror a               ; low byte of 128*t; carry now clear
    adc ql+256,x
    sta MP
    lda qh+256,x
    adc MT              ; high byte, including low-byte carry
    adc #64             ; previous ADC cannot overflow (<=191)
.Lsumdone:
    sta MH
    ; Subtract floor(abs(a-b)^2/4). Both squares have same parity.
    lda MA
    sec
    sbc MB
    bcs .Ldiffpos
    eor #$ff
    inc a
.Ldiffpos:
    tax
    lda MP
    sec
    sbc ql+256,x
    sta MP
    lda MH
    sbc qh+256,x
    sta MH
    ; Add the unsigned 16-bit product at byte offset MO+Y.
    tya
    clc
    adc MO
    tax
    lda MP
    clc
    adc .Lq,x
    sta .Lq,x
    inx
    lda MH
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
    iny
    cpy MLIM
    bcc .Lmulinner
.Lnextouter:
    inc MO
    lda MO
    cmp MLB
    bcs .Lmulend
    jmp .Lmulouter
.Lmulend:
    rts
.Lsignedout:
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
; Matrix rescaling: signed 24-bit dot * unsigned 24-bit scale *
; unsigned 15-bit peak, yielding a signed 64-bit result.
; R = Wide output, P = little-endian row scale (4 bytes, top byte 0),
; B = TinyRescaleArgs { int32_t dot; uint16_t peak; }.
; Only the dot can be negative. Convert its magnitude once, then use
; the bounded unsigned multiply core twice, applying the sign at end.
tiny_rescale:
    ldy #2
.Lrsload:
    lda (B),y
    sta .La,y
    lda (P),y
    sta .Lb,y
    dey
    bpl .Lrsload
    lda .La+2
    sta .Lsign
    bpl .Lrspositive
    sec
    lda #0
    sbc .La
    sta .La
    lda #0
    sbc .La+1
    sta .La+1
    lda #0
    sbc .La+2
    sta .La+2
.Lrspositive:
    jsr .Lclearq
    ldy #2
    ldx #2
    jsr .Lmul_unsigned
    ldy #5
.Lrscopy:
    lda .Lq,y
    sta .La,y
    dey
    bpl .Lrscopy
    ldy #4
    lda (B),y
    sta .Lb
    iny
    lda (B),y
    sta .Lb+1
    jsr .Lclearq
    ldy #5
    ldx #1
    jsr .Lmul_unsigned
    jmp .Lsignedout
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

; Convert signed Wide to a packed mantissa. R = input, A:X = shift.
; Precondition: -14 <= shift <= 49, rounded result in [-32767,32767].
; Positive shift: nearest, ties away from zero; negative: exact left shift.
; Only the three magnitude bytes containing the rounding bit and result
; need shifting. Negative inputs use the existing La workspace.
; No source writes, no extra BSS or reserved ZP.
.global tiny_pack16
tiny_pack16:
    cpx #0
    beq .Lpknotleft
    jmp .Lpkleft
.Lpknotleft:
    cmp #0
    bne .Lpknonzero
    jmp .Lpkzero
.Lpknonzero:
    dec a
    pha
    and #7
    inc a
    tax                 ; 1..8 right shifts, final carry is rounding bit
    pla
    lsr a
    lsr a
    lsr a
    sta MT              ; first byte = (shift-1)/8
    ldy #7
    lda (R),y
    sta .Lsign
    bpl .Lpkpositive
    ; Negate source into La, carrying across the low discarded bytes.
    stx MB
    ldx #8
    ldy #0
    sec
.Lpkmag:
    lda #0
    sbc (R),y
    sta .La,y
    iny
    dex
    bne .Lpkmag
    ldx MB
    ldy MT
    lda .La,y
    sta MO
    iny
    lda .La,y
    sta MP
    iny
    cpy #8
    beq .Lpkhighzero
    lda .La,y
    bra .Lpkright
.Lpkpositive:
    ldy MT
    lda (R),y
    sta MO
    iny
    lda (R),y
    sta MP
    iny
    cpy #8
    beq .Lpkhighzero
    lda (R),y
    bra .Lpkright
.Lpkhighzero:
    lda #0
.Lpkright:
    lsr a
    ror MP
    ror MO
    dex
    bne .Lpkright
    bcc .Lpknoround
    inc MO
    bne .Lpknoround
    inc MP
.Lpknoround:
    bit .Lsign
    bpl .Lpkreturn
    sec
    lda #0
    sbc MO
    tay
    lda #0
    sbc MP
    tax
    tya
    rts
.Lpkreturn:
    lda MO
    ldx MP
    rts
.Lpkzero:
    ldy #1
    lda (R),y
    tax
    dey
    lda (R),y
    rts
.Lpkleft:
    eor #$ff
    inc a
    tax
    ldy #1
    lda (R),y
    sta MP
    dey
    lda (R),y
.Lpkleftloop:
    asl a
    rol MP
    dex
    bne .Lpkleftloop
    ldx MP
    rts

; Attention value coefficient: round(127*a/b), 0 <= a <= b <= 2^41.
; R = a, P = b. Return 0 if b=0 (as in the original C loop).
; Seven fractional division steps compute 128*a = q*b + rem.
; Subtract a from rem, borrowing one quotient unit if needed, then
; round by comparing 2*rem with b. For a=b the seven steps produce
; q=127, rem=b; the correction gives rem=0, so no special case.
; Six-byte remainder in caller-saved $08..$0d; denominator in Lb.
; No new BSS, no extra reserved zero page; inputs are untouched.
.global tiny_attention_coeff
tiny_attention_coeff:
    ldy #5
    ldx #5
    lda #0
    sta MLA
.Lacload:
    lda (R),y
    sta MO,x
    lda (P),y
    sta .Lb,y
    ora MLA
    sta MLA
    dex
    dey
    bpl .Lacload
    lda MLA
    beq .Laczero
    stz MLA
    lda #7
    sta MLB
.Lacloop:
    jsr .Lacdouble
    asl MLA
    jsr .Laccmp
    bcc .Lacnext
    ldx #0
    ldy #6
    sec
.Lacsub:
    lda MO,x
    sbc .Lb,x
    sta MO,x
    inx
    dey
    bne .Lacsub
    inc MLA
.Lacnext:
    dec MLB
    bne .Lacloop
    ; Subtract a to change 128*a to 127*a.
    lda #6
    sta MLB
    ldx #0
    ldy #0
    sec
.Lacadjust:
    lda MO,x
    sbc (R),y
    sta MO,x
    inx
    iny
    dec MLB
    bne .Lacadjust
    bcs .Lacround
    dec MLA
    ldx #0
    ldy #6
    clc
.Lacborrow:
    lda MO,x
    adc .Lb,x
    sta MO,x
    inx
    dey
    bne .Lacborrow
.Lacround:
    jsr .Lacdouble
    jsr .Laccmp
    lda MLA
    adc #0              ; carry from compare: round half up
    ldx #0
    rts
.Laczero:
    lda #0
    tax
    rts
.Lacdouble:
    asl MO
    rol MP
    rol MA
    rol MB
    rol MH
    rol MT
    rts
.Laccmp:
    ldx #5
.Laccmploop:
    lda MO,x
    cmp .Lb,x
    bcc .Lacret
    bne .Lacret
    dex
    bpl .Laccmploop
.Lacret:
    rts
