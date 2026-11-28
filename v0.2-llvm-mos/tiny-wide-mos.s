; Stage 12C: LLVM-MOS Apple II 65C02 wide arithmetic.
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
.Lq: .space 9
.La: .space 8
.Lb: .space 8
.Ln: .space 1
.Lbits: .space 1
.Lsign: .space 1
.Lround: .space 1
.Lpeak: .space 2
.Lpkargs: .space 6
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
    ; For unsigned bytes a,b, use Q[256+a+b]-Q[256+a-b].
    ; b is nonzero. Patch four low address bytes once per outer row.
    sta .Lmulsumlo+1
    sta .Lmulsumhi+1
    eor #$ff
    inc a
    sta .Lmuldifflo+1
    sta .Lmuldiffhi+1
    ; X walks the output index MO..limit-1, so the A byte is at .La+X-MO
    ; and the product lands at .Lq+X with no index arithmetic.
    sec
    lda #<.La
    sbc MO
    sta .Lmula+1
    lda #>.La
    sbc #0
    sta .Lmula+2
    ; limit = min(8, MO+MLA): do not visit output bytes >= 8.
    clc
    lda MO
    adc MLA
    cmp #8
    bcc .Lsetlimit
    lda #8
.Lsetlimit:
    sta MLIM
    ldx MO
.Lmulinner:
.Lmula:
    ldy $ffff,x
    beq .Lnextinner
.Lmulsumlo:
    lda ql+256,y
    sec
.Lmuldifflo:
    sbc ql,y
    sta MP
.Lmulsumhi:
    lda qh+256,y
.Lmuldiffhi:
    sbc qh,y
    tay
    clc
    lda MP
    adc .Lq,x
    sta .Lq,x
    tya
    adc .Lq+1,x             ; .Lq+8 is padding for the discarded byte
    sta .Lq+1,x
    bcc .Lnextinner
    phx
.Lpropagate:
    inx
    cpx #7
    bcs .Lpropdone
    inc .Lq+1,x
    beq .Lpropagate
.Lpropdone:
    plx
.Lnextinner:
    inx
    cpx MLIM
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
; Sum of squares of packed mantissas. R = Wide result, P = int16 values,
; A = count 1..256 (0 means 256). Each square is one 2x2-byte multiply
; accumulated straight into the result; 256 squares cannot overflow.
.global tiny_sumsq
tiny_sumsq:
    sta .Ln
    jsr .Lclearq
.Lsqloop:
    ldy #1
    lda (P),y
    bpl .Lsqpositive
    sec
    dey
    lda #0
    sbc (P),y
    sta .La
    sta .Lb
    iny
    lda #0
    sbc (P),y
    bra .Lsqhigh
.Lsqpositive:
    dey
    lda (P),y
    sta .La
    sta .Lb
    iny
    lda (P),y
.Lsqhigh:
    sta .La+1
    sta .Lb+1
    ldx #1
    jsr .Lmul_unsigned
    clc
    lda P
    adc #2
    sta P
    bcc .Lsqnext
    inc P+1
.Lsqnext:
    dec .Ln
    bne .Lsqloop
    jmp .Lstoreq
; Attention score for one cached position.
; R = Wide out, P = TinyScoreArgs {int32 dot; uint16 m, csm; uint32 factor; uint8 top}.
; raw = dot*m*csm; shift = max(0, topbit(|raw|)-29);
; out = round(raw / 2^shift) * factor, nearest with ties away from zero.
; Returns shift; args.top = topbit(|out|), or $ff when out is zero.
; Works on magnitudes with the shared multiply core and applies the sign last.
.global tiny_att_score
tiny_att_score:
    ldy #3
    lda (P),y
    sta .Lsign
    dey
.Lscload:
    lda (P),y
    sta .La,y
    dey
    bpl .Lscload
    bit .Lsign
    bpl .Lscpositive
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
.Lscpositive:
    ldy #4
    lda (P),y
    sta .Lb
    iny
    lda (P),y
    sta .Lb+1
    jsr .Lclearq
    ldy #2
    ldx #1
    jsr .Lmul_unsigned
    ldy #4
.Lsccopy:
    lda .Lq,y
    sta .La,y
    dey
    bpl .Lsccopy
    ldy #6
    lda (P),y
    sta .Lb
    iny
    lda (P),y
    sta .Lb+1
    jsr .Lclearq
    ldy #4
    ldx #1
    jsr .Lmul_unsigned
    stz .Ln
    jsr .Lsctop
    bmi .Lscnoshift         ; raw is zero
    sec
    sbc #29
    bcc .Lscnoshift
    beq .Lscnoshift
    sta .Ln
    ; Shift right by shift-1 (whole bytes, then bits), then once more so
    ; the carry is the rounding bit. Five bytes hold everything needed.
    dec a
    pha
    and #7
    sta .Lbits
    pla
    lsr a
    lsr a
    lsr a
    tax
    ldy #0
.Lscmove:
    lda .Lq,x
    sta .La,y
    inx
    iny
    cpy #5
    bne .Lscmove
    ldx .Lbits
    inx
.Lscbits:
    lsr .La+4
    ror .La+3
    ror .La+2
    ror .La+1
    ror .La
    dex
    bne .Lscbits
    bcc .Lscfactor
    inc .La
    bne .Lscfactor
    inc .La+1
    bne .Lscfactor
    inc .La+2
    bne .Lscfactor
    inc .La+3
    bra .Lscfactor
.Lscnoshift:
    ldy #3
.Lsckeep:
    lda .Lq,y
    sta .La,y
    dey
    bpl .Lsckeep
.Lscfactor:
    ldy #11
    ldx #3
.Lscfload:
    lda (P),y
    sta .Lb,x
    dey
    dex
    bpl .Lscfload
    jsr .Lclearq
    ldy #3
    ldx #3
    jsr .Lmul_unsigned
    jsr .Lsctop
    ldy #12
    sta (P),y
    jsr .Lsignedout
    lda .Ln
    ldx #0
    rts
; A = index of the highest set bit of .Lq, or $ff if it is zero.
.Lsctop:
    ldx #7
.Lsctopscan:
    lda .Lq,x
    bne .Lsctopfound
    dex
    bpl .Lsctopscan
    lda #$ff
    rts
.Lsctopfound:
    ldy #$ff
.Lsctopbit:
    iny
    lsr a
    bne .Lsctopbit
    sty MT
    txa
    asl a
    asl a
    asl a
    ora MT
    rts

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
    ldy #4
    lda (B),y
    sta .Lpeak
    iny
    lda (B),y
    sta .Lpeak+1
; .La = signed 24-bit dot, .Lb = 24-bit scale, .Lpeak = peak; result to R.
.Lrscore:
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
.Lnegative_rescale:
    jmp .Lrspositive        ; selected once per projection by tiny_f1_begin/finish
.Lrspositive:
    jsr .Lrsfirst
    jsr .Lrssecond
    jmp .Lsignedout

; Each JSR target is installed once after the model file closes.
; The six pack arguments are idle for the duration of matrix rescaling.
rs_product = .Lpkargs
.Lrsfirst:
    jmp $ffff               ; .La[3] * .Lb[3] -> rs_product[6]
.Lrssecond:
    jmp $ffff               ; rs_product[6] * .Lpeak[2] -> .Lq[8]

; Generate both fixed-size, straight-line schoolbook products. R points to
; writable main memory, with 1 KiB available. Constants and operand addresses
; are obtained from the final link. No code is changed in the inference loop.
; Both stages set their own table pointers so either can be called alone.
.global tiny_rescale_init
rs_left = $06
rs_right = $08
rs_out = $0a
rs_n = $0c
rs_m = $0d
rs_i = $0e
rs_j = $0f

.macro rs_copy first,last
    ldy #(\first-.Lrs_templates)
    ldx #(\last-\first)
    jsr .Lrs_stamp
.endm

tiny_rescale_init:
    lda $02
    sta .Lrsfirst+1
    lda $03
    sta .Lrsfirst+2
    ldx #0
    jsr .Lrs_generate
    lda $02
    sta .Lrssecond+1
    lda $03
    sta .Lrssecond+2
    ldx #8
.Lrs_generate:
    ldy #0
.Lrs_arguments:
    lda .Lrs_shapes,x
    sta rs_left,y
    inx
    iny
    cpy #8
    bne .Lrs_arguments
    rs_copy .Lrs_tables,.Lrs_rightpage
    stz rs_j
.Lrs_outer:
    rs_copy .Lrs_rightpage,.Lrs_rightaddr
    lda rs_j
    ldx #2
    jsr .Lrs_address
    rs_copy .Lrs_ptrs,.Lrs_input
    stz rs_i
.Lrs_inner:
    lda #$ac                ; LDY absolute
    jsr .Lrs_byte
    lda rs_i
    ldx #0
    jsr .Lrs_address
    rs_copy .Lrs_input,.Lrs_addcarry
    lda rs_i
    beq .Lrs_firstdigit
    rs_copy .Lrs_addcarry,.Lrs_adjust
    lda rs_j
    beq .Lrs_store
    rs_copy .Lrs_adjust,.Lrs_sum
    bra .Lrs_accumulate
.Lrs_firstdigit:
    lda rs_j
    beq .Lrs_store
    lda #$18                ; CLC before adding a previous partial sum
    jsr .Lrs_byte
.Lrs_accumulate:
    lda #$6d                ; ADC absolute
    jsr .Lrs_outoperand
.Lrs_store:
    lda #$8d                ; STA absolute
    jsr .Lrs_outoperand
    lda #$98                ; TYA (high product byte)
    jsr .Lrs_byte
    lda rs_i
    ora rs_j
    beq .Lrs_carry
    rs_copy .Lrs_sum,.Lrs_newcarry
.Lrs_carry:
    inc rs_i
    lda rs_i
    cmp rs_n
    beq .Lrs_lastdigit
    rs_copy .Lrs_newcarry,.Lrs_endtemplate
    bra .Lrs_inner
.Lrs_lastdigit:
    lda #$8d                ; final carry becomes the next result byte
    jsr .Lrs_outoperand
    inc rs_j
    lda rs_j
    cmp rs_m
    beq .Lrs_done
    jmp .Lrs_outer
.Lrs_done:
    lda #$60                ; RTS
    bra .Lrs_byte
.Lrs_outoperand:
    jsr .Lrs_byte
    lda rs_i
    clc
    adc rs_j
    ldx #4
.Lrs_address:
    clc
    adc rs_left,x
    jsr .Lrs_byte
    lda rs_left+1,x
    adc #0
.Lrs_byte:
    sta (R)
    inc $02
    bne .Lrs_return
    inc $03
.Lrs_return:
    rts
.Lrs_stamp:
    lda .Lrs_templates,y
    jsr .Lrs_byte
    iny
    dex
    bne .Lrs_stamp
    rts
.Lrs_shapes:
    .word .La,.Lb,rs_product
    .byte 3,3
    .word rs_product,.Lpeak,.Lq
    .byte 6,2
.Lrs_templates:
.Lrs_tables:
    lda #>(ql+256)
    sta $09
    lda #>(qh+256)
    sta $0d
.Lrs_rightpage:
    ldx #>ql
    ldy #>qh
    .byte $ad               ; LDA absolute, followed by right operand address
.Lrs_rightaddr:
.Lrs_ptrs:
    sta $08
    sta $0c
    eor #$ff
    inc a
    sta $0a
    sta $0e
    bne 1f
    inx                     ; when right=0 the negative offset is 256
    iny
1:  stx $0b
    sty $0f
.Lrs_input:
    sec
    lda ($08),y
    sbc ($0a),y
    sta $11
    lda ($0c),y
    sbc ($0e),y
    tay
    lda $11
.Lrs_addcarry:
    clc
    adc $10
.Lrs_adjust:
    bcc 2f
    iny
    clc
2:
.Lrs_sum:
    adc #0
.Lrs_newcarry:
    sta $10
.Lrs_endtemplate:
; Rescale a batch of kernel results. R = Wide outputs, P = eight-byte
; records (three scale bytes and a spare, then the dot as int32),
; B = uint16 args[2] {count 1..255, peak}.
.global tiny_rescale_rows
tiny_rescale_rows:
    ldy #0
    lda (B),y
    sta .Lround
    ldy #2
    lda (B),y
    sta .Lpeak
    iny
    lda (B),y
    sta .Lpeak+1
.Lrrloop:
    ldy #2
.Lrrscale:
    lda (P),y
    sta .Lb,y
    dey
    bpl .Lrrscale
    ldy #4
    lda (P),y
    sta .La
    iny
    lda (P),y
    sta .La+1
    iny
    lda (P),y
    sta .La+2
    jsr .Lrscore
    clc
    lda $02                 ; R
    adc #8
    sta $02
    bcc .Lrrout
    inc $03
.Lrrout:
    clc
    lda P
    adc #8
    sta P
    bcc .Lrrnext
    inc P+1
.Lrrnext:
    dec .Lround
    bne .Lrrloop
    rts
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
    lda .Lb+4
    ora .Lb+5
    ora .Lb+6
    ora .Lb+7
    beq .Ldivnarrow
    jmp .Ldivgeneral
.Ldivnarrow:
    lda .Lb+1
    ora .Lb+2
    ora .Lb+3
    beq .Ldiv8start
    jmp .Ldiv32
.Ldiv8start:
    ; Fast one-byte divisor (e.g. 127): A holds running remainder.
    ldx #7
.Ldiv8scan:
    lda .Lq,x
    bne .Ldiv8ready
    dex
    bne .Ldiv8scan
.Ldiv8ready:
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

; Largest magnitude of count signed Wide values. R = Wide out, P = values,
; A:X = count (at least 1). Magnitudes compare as unsigned 64-bit numbers.
.global tiny_wmaxmag
tiny_wmaxmag:
    sta .Ln
    stx .Lbits
    jsr .Lclearq
.Lmxloop:
    ldy #7
    lda (P),y
    bmi .Lmxneg
.Lmxcmp:
    lda (P),y
    cmp .Lq,y
    bcc .Lmxnext
    bne .Lmxtake
    dey
    bpl .Lmxcmp
    bra .Lmxnext
.Lmxtake:
    ldy #7
.Lmxcopy:
    lda (P),y
    sta .Lq,y
    dey
    bpl .Lmxcopy
    bra .Lmxnext
.Lmxneg:
    ldy #0
    ldx #8
    sec
.Lmxnegate:
    lda #0
    sbc (P),y
    sta .La,y
    iny
    dex
    bne .Lmxnegate
    ldy #7
.Lmxcmp2:
    lda .La,y
    cmp .Lq,y
    bcc .Lmxnext
    bne .Lmxtake2
    dey
    bpl .Lmxcmp2
    bra .Lmxnext
.Lmxtake2:
    ldy #7
.Lmxcopy2:
    lda .La,y
    sta .Lq,y
    dey
    bpl .Lmxcopy2
.Lmxnext:
    clc
    lda P
    adc #8
    sta P
    bcc .Lmxcount
    inc P+1
.Lmxcount:
    lda .Ln
    bne .Lmxdec
    dec .Lbits
.Lmxdec:
    dec .Ln
    lda .Ln
    ora .Lbits
    bne .Lmxloop
    jmp .Lstoreq

; Pack count Wide values with one shift. R = int16 out, P = Wide values,
; B = int16 args[2] {count >= 1, shift}; tiny_pack16's preconditions apply.
.global tiny_pack_rows
tiny_pack_rows:
    lda $02                 ; tiny_pack16 reads its input through R
    ldx $04
    stx $02
    sta $04
    lda $03
    ldx $05
    stx $03
    sta $05
    ldy #3
.Lprargs:
    lda (B),y
    sta .Lpkargs,y
    dey
    bpl .Lprargs
.Lprloop:
    lda .Lpkargs+2
    ldx .Lpkargs+3
    jsr tiny_pack16
    ldy #0
    sta (P),y
    iny
    txa
    sta (P),y
    clc
    lda $02
    adc #8
    sta $02
    bcc .Lprout
    inc $03
.Lprout:
    clc
    lda P
    adc #2
    sta P
    bcc .Lprcount
    inc P+1
.Lprcount:
    lda .Lpkargs
    bne .Lprdec
    dec .Lpkargs+1
.Lprdec:
    dec .Lpkargs
    lda .Lpkargs
    ora .Lpkargs+1
    bne .Lprloop
    rts

; Residual sum of two packed vectors at a common exponent.
; R = Wide out, P = int16 a, B = TinyResidualArgs {const int16_t *b;
; uint16 count; int8 sa, sb}. out[i] = shift(a[i],sa) + shift(b[i],sb),
; where a shift of 0..16 is an exact left shift and -1..-16 a right shift
; rounded to nearest, ties away from zero.
.global tiny_residual_rows
rb = $11
tiny_residual_rows:
    ldy #5
.Lrdargs:
    lda (B),y
    sta .Lpkargs,y
    dey
    bpl .Lrdargs
    lda .Lpkargs
    sta rb
    lda .Lpkargs+1
    sta rb+1
.Lrdloop:
    ldy #0
    lda (P),y
    sta MO
    iny
    lda (P),y
    sta MP
    lda .Lpkargs+4
    jsr .Lrdalign
    ldx #3
.Lrdsave:
    lda MO,x
    sta .La,x
    dex
    bpl .Lrdsave
    ldy #0
    lda (rb),y
    sta MO
    iny
    lda (rb),y
    sta MP
    lda .Lpkargs+5
    jsr .Lrdalign
    ldy #0
    clc
    lda .La
    adc MO
    sta (R),y
    iny
    lda .La+1
    adc MP
    sta (R),y
    iny
    lda .La+2
    adc MA
    sta (R),y
    iny
    lda .La+3
    adc MB
    sta (R),y
    iny
    ; Fifth byte: the two sign extensions plus the carry; then its sign.
    php
    ldx #0
    lda .La+3
    bpl .Lrdexta
    dex
.Lrdexta:
    stx MT
    ldx #0
    lda MB
    bpl .Lrdextb
    dex
.Lrdextb:
    txa
    plp
    adc MT
    sta (R),y
    asl a
    lda #0
    sbc #0
    eor #$ff
    iny
    sta (R),y
    iny
    sta (R),y
    iny
    sta (R),y
    clc
    lda $02
    adc #8
    sta $02
    bcc .Lrdn1
    inc $03
.Lrdn1:
    clc
    lda P
    adc #2
    sta P
    bcc .Lrdn2
    inc P+1
.Lrdn2:
    clc
    lda rb
    adc #2
    sta rb
    bcc .Lrdn3
    inc rb+1
.Lrdn3:
    lda .Lpkargs+2
    bne .Lrddec
    dec .Lpkargs+3
.Lrddec:
    dec .Lpkargs+2
    lda .Lpkargs+2
    ora .Lpkargs+3
    beq .Lrddone
    jmp .Lrdloop
.Lrddone:
    rts
; MO,MP = signed 16-bit value, A = shift -16..16. Result: MO..MB, signed 32-bit.
.Lrdalign:
    tax
    bmi .Lrdright
    ldy #0
    lda MP
    bpl .Lrdlext
    dey
.Lrdlext:
    sty MA
    sty MB
.Lrdlbytes:
    cpx #8
    bcc .Lrdlbits
    lda MA
    sta MB
    lda MP
    sta MA
    lda MO
    sta MP
    stz MO
    txa
    sbc #8                  ; carry is set
    tax
    bra .Lrdlbytes
.Lrdlbits:
    cpx #0
    beq .Lrdret
.Lrdlbit:
    asl MO
    rol MP
    rol MA
    rol MB
    dex
    bne .Lrdlbit
.Lrdret:
    rts
.Lrdright:
    eor #$ff
    inc a
    tax                     ; 1..16 right shifts of the magnitude
    lda MP
    sta MT                  ; sign
    bpl .Lrdrmag
    sec
    lda #0
    sbc MO
    sta MO
    lda #0
    sbc MP
    sta MP
.Lrdrmag:
    lsr MP
    ror MO
    dex
    bne .Lrdrmag
    bcc .Lrdrsign
    inc MO
    bne .Lrdrsign
    inc MP
.Lrdrsign:
    stz MA
    stz MB
    bit MT
    bpl .Lrdret
    sec
    lda #0
    sbc MO
    sta MO
    lda #0
    sbc MP
    sta MP
    lda #0
    sbc #0
    sta MA
    sta MB
    rts

; Attention value coefficient: round(127*a/b), 0 <= a <= b <= 2^41.
; R = a, P = b. Return 0 if b=0 (as in the original C loop).
; Seven fractional division steps compute 128*a = q*b + rem.
; Subtract a from rem, borrowing one quotient unit if needed, then
; round by comparing 2*rem with b. For a=b the seven steps produce
; q=127, rem=b; the correction gives rem=0, so no special case.
; Six-byte remainder in caller-saved $08..$0d; denominator in Lb.
; No new BSS, no extra reserved zero page; inputs are untouched.
.global tiny_attention_coeff, tiny_attention_coeffs
tiny_attention_coeff:
    jsr .Lacdenom
    beq .Lacscalarzero
    jsr .Lacone
.Lacscalarzero:
    ldx #0
    rts

; R = input scores, P = shared maximum, B = signed byte output, A = 1..128.
tiny_attention_coeffs:
    sta .Ln
    jsr .Lacdenom
.Lacrow:
    jsr .Lacone
    sta (B)
    clc
    lda $02                 ; R
    adc #8
    sta $02
    bcc .Lacnextout
    inc $03
.Lacnextout:
    inc B
    bne .Lacnextrow
    inc B+1
.Lacnextrow:
    dec .Ln
    bne .Lacrow
    rts
.Lacdenom:
    stz .Lb+6
    lda #0
    ldy #5
    tax
.Laccopyden:
    lda (P),y
    sta .Lb,y
    ; Return Z when b is zero for the single-coefficient entry point.
    beq .Lacdennext
    tax
.Lacdennext:
    dey
    bpl .Laccopyden
    txa
    rts
.Lacone:
    ; .Lb+1..+6 is floor(b/256). At or below that cutoff 127*a/b
    ; is strictly below a half, including a zero score. If b=0 all
    ; scores in a valid batch are zero. Equality can return zero.
    ldy #5
.Laccutoff:
    lda (R),y
    cmp .Lb+1,y
    bcc .Laczero
    bne .Lacdivide
    dey
    bpl .Laccutoff
.Laczero:
    lda #0
    rts
.Lacdivide:
    ldx #5
    ldy #5
.Lacload:
    lda (R),y
    sta MO,x
    dex
    dey
    bpl .Lacload
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

; Stage 11N: quantize a packed vector to signed bytes, exactly as q8.
; R = int16 input, P = int8 output, B = uint16 args[2] {n,peak}.
; 1 <= n <= 256, 1 <= peak <= 32767, abs(input[i]) <= peak.
; Zero encoded count means 256. Source and destination do not overlap.
; Compute seven fractional quotient bits, then subtract the original
; magnitude to turn 128*a into 127*a. Round ties away from zero.
; Two-byte remainder suffices (2*peak <= 65534). No extra ZP or BSS.
.global tiny_q8_vector
tiny_q8_vector:
    ldy #0
    lda (B),y
    sta MT              ; remaining count, modulo 256
    ldy #2
    lda (B),y
    sta MLB             ; denominator low
    iny
    lda (B),y
    sta MLIM            ; denominator high
.Lq8loop:
    ldy #0
    lda (R),y
    sta MO
    iny
    lda (R),y
    sta MH              ; sign
    sta MP
    bpl .Lq8mag
    sec
    lda #0
    sbc MO
    sta MO
    lda #0
    sbc MP
    sta MP
.Lq8mag:
    lda MO
    sta MA
    lda MP
    sta MB
    ora MO              ; a literal zero (common after ReLU) quantizes to zero
    beq .Lq8store
    lda #2              ; sentinel: shifts out after seven quotient bits
    sta MLA
.Lq8div:
    asl MA
    rol MB
    sec
    lda MA
    sbc MLB
    tay
    lda MB
    sbc MLIM
    bcc .Lq8bit
    sty MA
    sta MB
.Lq8bit:
    rol MLA             ; quotient bit from subtract; carry = sentinel
    bcc .Lq8div
    sec
    lda MA
    sbc MO
    sta MA
    lda MB
    sbc MP
    sta MB
    bcs .Lq8round
    dec MLA
    clc
    lda MA
    adc MLB
    sta MA
    lda MB
    adc MLIM
    sta MB
.Lq8round:
    asl MA
    rol MB
    sec
    lda MA
    sbc MLB
    lda MB
    sbc MLIM
    lda MLA
    adc #0
    bit MH
    bpl .Lq8store
    eor #$ff
    inc a
.Lq8store:
    ldy #0
    sta (P),y
    clc
    lda $02
    adc #2
    sta $02
    bcc .Lq8srcok
    inc $03
.Lq8srcok:
    inc P
    bne .Lq8dstok
    inc P+1
.Lq8dstok:
    dec MT
    beq .Lq8done
    jmp .Lq8loop
.Lq8done:
    rts

; Unsigned 64/32 division, 2..4 significant divisor bytes. Process the
; dividend one byte at a time, overwriting that byte with quotient bits.
; Remainder in caller-saved ZP MO..MB, divisor in MH..MLB. The carry out
; of the 32-bit remainder shift represents its implicit 33rd bit.
; All input bytes have already been copied, and La was zeroed by caller.
.Ldiv32:
    lda .Lb
    sta MH
    lda .Lb+1
    sta MT
    lda .Lb+2
    sta MLA
    lda .Lb+3
    sta MLB
    stz MO
    stz MP
    stz MA
    stz MB
    ldx #7
.Ld32scan:
    lda .Lq,x
    bne .Ld32byte
    dex
    bne .Ld32scan
.Ld32byte:
    ldy #8
.Ld32bit:
    asl .Lq,x
    rol MO
    rol MP
    rol MA
    rol MB
    bcs .Ld32sub
    lda MB
    cmp MLB
    bcc .Ld32next
    bne .Ld32sub
    lda MA
    cmp MLA
    bcc .Ld32next
    bne .Ld32sub
    lda MP
    cmp MT
    bcc .Ld32next
    bne .Ld32sub
    lda MO
    cmp MH
    bcc .Ld32next
.Ld32sub:
    sec
    lda MO
    sbc MH
    sta MO
    lda MP
    sbc MT
    sta MP
    lda MA
    sbc MLA
    sta MA
    lda MB
    sbc MLB
    sta MB
    inc .Lq,x
.Ld32next:
    dey
    bne .Ld32bit
    dex
    bpl .Ld32byte
    lda MO
    sta .La
    lda MP
    sta .La+1
    lda MA
    sta .La+2
    lda MB
    sta .La+3
    jmp .Ldivfinish

; F1 is always followed by ReLU. The pack exponent depends on the largest
; magnitude of either sign. Retain all positive outputs and the single
; largest negative output across batches; every other negative becomes zero.
; The input peak is shared over the projection, so compare |dot|*row_scale
; before the peak multiply. With row_scale <= 2^23, a dot no larger than
; (largest_negative_product >> 23) cannot displace the current winner.
.section .bss,"aw",@nobits
.Lf1max:    .space 6
.Lf1bound:  .space 3
.Lf1winner: .space 2
.section .text,"ax",@progbits
.global tiny_f1_begin, tiny_f1_finish
tiny_f1_begin:
    lda #0
    ldx #10
.Lf1reset:
    sta .Lf1max,x
    dex
    bpl .Lf1reset
    lda #<.Lf1negative
    sta .Lnegative_rescale+1
    lda #>.Lf1negative
    sta .Lnegative_rescale+2
    rts

.Lf1negative:
    ldx #2
.Lf1checkbound:
    lda .La,x
    cmp .Lf1bound,x
    bcc .Lf1zero
    bne .Lf1candidate
    dex
    bpl .Lf1checkbound
    bra .Lf1zero
.Lf1candidate:
    jsr .Lrsfirst
    ldx #5
.Lf1compare:
    lda rs_product,x
    cmp .Lf1max,x
    bcc .Lf1zero
    bne .Lf1larger
    dex
    bpl .Lf1compare
    bra .Lf1zero
.Lf1larger:
    ; Higher bytes already compare equal. Copy the differing byte and below.
    sta .Lf1max,x
    dex
    bmi .Lf1threshold
    lda rs_product,x
    bra .Lf1larger
.Lf1threshold:
    lda .Lf1max+2
    asl a
    ldx #0
    ldy #3
.Lf1boundbyte:
    lda .Lf1max+3,x
    rol a
    sta .Lf1bound,x
    inx
    dey                     ; preserve carry for the next ROL
    bne .Lf1boundbyte
    lda $02                 ; R
    sta .Lf1winner
    lda $03
    sta .Lf1winner+1
.Lf1zero:
    lda #0
    ldy #7
.Lf1clear:
    sta (R),y
    dey
    bpl .Lf1clear
    rts

tiny_f1_finish:
    lda #<.Lrspositive
    sta .Lnegative_rescale+1
    lda #>.Lrspositive
    sta .Lnegative_rescale+2
    lda .Lf1winner
    sta $02                 ; R
    ora .Lf1winner+1
    beq .Lf1done
    lda .Lf1winner+1
    sta $03
    ldx #5
.Lf1restore:
    lda .Lf1max,x
    sta rs_product,x
    dex
    bpl .Lf1restore
    jsr .Lrssecond
    lda #$80
    sta .Lsign
    jmp .Lsignedout
.Lf1done:
    rts
