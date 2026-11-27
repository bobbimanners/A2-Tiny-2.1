; TINY - prepared sparse matrix MAC, plus general 11K kernels.
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

; Assembly options (--defsym NAME=1):
;   TINY_MAC_GENERAL  also assemble the general dot and axpy and the
;                     single-row prepared entry points, for standalone tests.
;   TINY_SINGLE_ROW   tiny_mac65_aux_rows makes one pass over the columns
;                     per record instead of sharing a pass between eight.
;                     Same code path, interface and results; it reads one
;                     weight stream at a time, for comparison on accelerators.
.ifdef TINY_MAC_GENERAL
TINY_SINGLE_BODY = 1
.endif

; Private zero page, reserved by -mreserve-zp=30.
.section .zp.bss,"aw",@nobits
ap: .space 2
bp: .space 2
cp: .space 2
; Running sums of the multi-row kernel: low, middle and high byte per row.
; They are in zero page, not in the code, because the code sits in hi-res
; page 1 with HIRES switched on: a write there is a write to video memory.
rsum: .space 24

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

; The general dot and axpy and the single-row prepared entry points are only
; used by the standalone tests; TINY itself uses the multi-row and in-place
; kernels. Assemble with --defsym TINY_MAC_GENERAL=1 to include them.
.ifdef TINY_MAC_GENERAL
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
.Ldot_fix:
    ; A = count>>1, C = count&1
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
.endif


.ifdef TINY_MAC_GENERAL
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
.endif


; Matrix-only fast path. Prepare once per projection, then call prepared_dot
; for each row. tiny_mac_a = fixed weight buffer, tiny_mac_b = writable
; 512-byte activation workspace, tiny_mac_n = 1..256. q8 guarantees signed
; activations -127..127 (weights may be -128). Prepare replaces each nonzero
; activation in the first plane with u=a+128 and links the nonzero columns
; through the second plane: next[i] is the following nonzero column, 0 at the
; end (column 0 can only be the first). Zero activations contribute nothing
; and are never visited; after ReLU most of a hidden vector is zero.
; Q[u+v]-Q[256-u+v] == a*b for v=b+128, by Q symmetry.
; All prepared state survives unrelated routines in patched code and the
; workspace. Do not run prepare again until done.
tiny_mac65_prepare:
    cld
.ifdef TINY_SINGLE_BODY
    lda tiny_mac_a
    sta .Lprepared_weight+1
    lda tiny_mac_a+1
    sta .Lprepared_weight+2
.endif
    lda tiny_mac_b
    sta bp
    sta cp
.ifdef TINY_SINGLE_BODY
    sta .Lprepared_pos+1
    sta .Lprepared_next+1
.endif
    lda tiny_mac_b+1
    sta bp+1
.ifdef TINY_SINGLE_BODY
    sta .Lprepared_pos+2
.endif
    inc a
    sta cp+1
.ifdef TINY_SINGLE_BODY
    sta .Lprepared_next+2
.endif
    stz .Lprepared_count
    stz .Lprepared_count+1
    stz lhs                 ; next nonzero column so far (0 = none)
    ldy tiny_mac_n          ; low byte 0 means 256
.Lprepare_loop:
    dey
    lda (bp),y
    beq .Lprepare_skip
    eor #$80
    sta (bp),y
    lda lhs
    sta (cp),y
    sty lhs
    inc .Lprepared_count
    bne .Lprepare_skip
    inc .Lprepared_count+1
.Lprepare_skip:
    cpy #0
    bne .Lprepare_loop
    lda lhs
    sta .Lprepared_first+1
    rts
; Number of nonzero columns, and the first of them. They live with the code
; so that they stay readable while auxiliary memory is switched in.
.Lprepared_count: .byte 0,0
.Lprepared_first: .byte 0,0

.ifdef TINY_MAC_GENERAL
tiny_mac65_prepared_dot:
    jsr .Lprepared_body
.Lprepared_done:
    ; Remove count*$8000, as dot does for n.
    lda .Lprepared_count+1
    lsr a
    lda .Lprepared_count
    ror a
    jmp .Ldot_fix
.endif
.ifdef TINY_SINGLE_BODY
.Lprepared_body:
    cld
    stz $0e
    stz $0f
    stz $10
    lda .Lprepared_count
    ora .Lprepared_count+1
    beq .Lprepared_empty
.Lprepared_first_single:
    ldy .Lprepared_first+1
.Lprepared_loop:
.Lprepared_pos:
    lda $ffff,y
    sta .Lprepared_pluslo+1
    sta .Lprepared_plushi+1
    eor #$ff                ; 256-u
    inc a
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
.Lprepared_next:
    lda $ffff,y             ; there is no LDY abs,Y
    tay
    bne .Lprepared_loop
.Lprepared_empty:
    rts
.endif

.ifdef TINY_MAC_GENERAL
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
    jsr .Lprepared_done
    plp
    rts
.endif

; Several rows per pass. The per-column table setup is the expensive part of
; the single-row loop, and it depends only on the activation, so up to eight
; consecutive records share it. Table planes are reached through zero-page
; pointers and each row keeps its running sum in zero page, so the loop
; writes nothing into the window.
;
; tiny_mac_a = first record in auxiliary memory (four scale bytes, weights),
; tiny_mac_coeff = bank, tiny_mac_rows = 1..8, tiny_mac_stride = bytes from
; one record to the next, tiny_mac_b = main-memory output, eight bytes per
; row: the four bytes ahead of the weights, then the dot as int32.
; Prepare first, in tiny_mac65_work. Same placement and mapping rules as
; tiny_mac65_aux_dot.
.global tiny_mac65_aux_rows
tp1 = $02
tm1 = $04
tp2 = $06
tm2 = $08
rstride = $0e
rrows = $10

.macro MACROW r
.Lrow\r:
.Lroww\r:
    lda $ffff,x
    eor #$80
    tay
    sec
    lda (tp1),y
    sbc (tm1),y
    sta pl
    lda (tp2),y
    sbc (tm2),y
    tay
    clc
    lda pl
    adc rsum+\r
    sta rsum+\r
    tya
    adc rsum+8+\r
    sta rsum+8+\r
    bcc 1f
    inc rsum+16+\r
1:
.endm

tiny_mac65_aux_rows:
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
    ; Everything read from here on is zero page or inside the window.
    lda tiny_mac_stride
    sta rstride
    lda tiny_mac_stride+1
    sta rstride+1
    lda tiny_mac_rows
    sta rrows
    lda tiny_mac_b
    sta bp
    lda tiny_mac_b+1
    sta bp+1
    lda tiny_mac_a
    sta ap
    clc
    adc #4
    sta cp
    lda tiny_mac_a+1
    sta ap+1
    adc #0
    sta cp+1
    ; Point each row at its weights and clear its sum. The operand
    ; addresses come from a table; cp steps from record to record.
    ldx #14
.Lrows_setup:
    lda .Lrows_wtab,x
    sta tp1
    lda .Lrows_wtab+1,x
    sta tp1+1
    ldy #1
    lda cp+1
    sta (tp1),y
    lda cp
    sta (tp1)
    clc
    lda cp
    adc rstride
    sta cp
    lda cp+1
    adc rstride+1
    sta cp+1
    dex
    dex
    bpl .Lrows_setup
    ldx #23
.Lrows_clear:
    stz rsum,x
    dex
    bpl .Lrows_clear
    lda #>ql
    sta tp1+1
    sta tm1+1
    lda #>qhb
    sta tp2+1
    lda #>qh
    sta tm2+1
    ; With k rows, enter at row k-1 and fall through to row 0.
.ifdef TINY_SINGLE_ROW
    lda #2                  ; always enter at row 0
.else
    lda rrows
    asl a
.endif
    tax
    lda .Lrows_entry-2,x
    sta .Lrows_jmp+1
    lda .Lrows_entry-1,x
    sta .Lrows_jmp+2
    lda tiny_mac_coeff
    sta $C073
    sta $C003
.ifdef TINY_SINGLE_ROW
.Lrows_again:
    ; Row 0 takes the next record: weights at ap+4, sum cleared.
    clc
    lda ap
    adc #4
    sta .Lroww0+1
    lda ap+1
    adc #0
    sta .Lroww0+2
    stz rsum
    stz rsum+8
    stz rsum+16
.endif
    lda .Lprepared_count
    ora .Lprepared_count+1
    bne .Lrows_some
    jmp .Lrows_collect
.Lrows_some:
    ldx .Lprepared_first+1
.Lrows_column:
    lda tiny_mac65_work,x
    sta tp1
    sta tp2
    eor #$ff                ; 256-u
    inc a
    sta tm1
    sta tm2
.Lrows_jmp:
    jmp $ffff
    MACROW 7
    MACROW 6
    MACROW 5
    MACROW 4
    MACROW 3
    MACROW 2
    MACROW 1
    MACROW 0
    lda tiny_mac65_work+256,x
    tax
    beq .Lrows_collect
    jmp .Lrows_column
.Lrows_collect:
    ; The plus term came from qhb, so each product carried +$4000.
    ; count*$4000 mod 2^24 has bytes [0, (count&3)<<6, count>>2].
    lda .Lprepared_count+1
    lsr a
    lda .Lprepared_count
    ror a
    sta ph
    lda #0
    ror a
    sta pl
    lsr ph
    ror pl
    ldx #0
.Lrows_out:
    ldy #3
.Lrows_scale:
    lda (ap),y
    sta (bp),y
    dey
    bpl .Lrows_scale
    ldy #4
    lda rsum,x
    sta (bp),y
    iny
    sec
    lda rsum+8,x
    sbc pl
    sta (bp),y
    iny
    lda rsum+16,x
    sbc ph
    sta (bp),y
    iny
    asl a                   ; sign of the 24-bit sum
    lda #0
    sbc #0
    eor #$ff
    sta (bp),y
    clc
    lda ap
    adc rstride
    sta ap
    lda ap+1
    adc rstride+1
    sta ap+1
    clc
    lda bp
    adc #8
    sta bp
    bcc .Lrows_next
    inc bp+1
.Lrows_next:
.ifdef TINY_SINGLE_ROW
    dec rrows
    beq .Lrows_done
    jmp .Lrows_again
.Lrows_done:
.else
    inx
    cpx rrows
    bne .Lrows_out
.endif
    sta $C002
    stz $C073
    pla
    bmi .Lrows_storeok
    sta $C000
.Lrows_storeok:
    pla
    bmi .Lrows_hiresok
    sta $C056
.Lrows_hiresok:
    pla
    bpl .Lrows_pageok
    sta $C055
.Lrows_pageok:
    plp
    rts
.Lrows_entry:
    .word .Lrow0,.Lrow1,.Lrow2,.Lrow3,.Lrow4,.Lrow5,.Lrow6,.Lrow7
; Operand addresses, in the order the records are laid out. The table is
; walked backwards, so it lists the last row first: the setup loop steps cp
; forwards while X counts down.
.Lrows_wtab:
    .word .Lroww7+1,.Lroww6+1,.Lroww5+1,.Lroww4+1,.Lroww3+1,.Lroww2+1,.Lroww1+1,.Lroww0+1

; Attention value accumulation, reading cached values in place.
; One coefficient multiplies a whole 24-byte head, so the table pointers are
; set once per call and the loop is a lookup and a 24-bit add per element.
;   tiny_mac65_axpy_begin  clears the sums
;   tiny_mac65_aux_axpy    sums[i] += tiny_mac_coeff * value[i]; tiny_mac_a =
;                          auxiliary address of the 24 values, tiny_mac_rows =
;                          bank, tiny_mac_coeff = 1..127
;   tiny_mac65_axpy_end    writes the sums to tiny_mac_acc as int32
; At most 128 calls between begin and end. The sums live in the activation
; planes, which are idle once a head has been scored.
.global tiny_mac65_axpy_begin, tiny_mac65_aux_axpy, tiny_mac65_axpy_end
AXN = 24
.Lax_lo = tiny_mac65_work
.Lax_mid = tiny_mac65_work+32
.Lax_hi = tiny_mac65_work+64

tiny_mac65_axpy_begin:
    ldx #AXN-1
.Lax_clear:
    stz .Lax_lo,x
    stz .Lax_mid,x
    stz .Lax_hi,x
    dex
    bpl .Lax_clear
    stz .Lax_terms
    rts

tiny_mac65_aux_axpy:
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
    sta .Lax_value+1
    lda tiny_mac_a+1
    sta .Lax_value+2
    lda tiny_mac_coeff
    eor #$80                ; u = coefficient + 128
    sta tp1
    sta tp2
    eor #$ff
    inc a
    sta tm1
    sta tm2
    lda #>ql
    sta tp1+1
    sta tm1+1
    lda #>qhb               ; each product arrives as product + $4000
    sta tp2+1
    lda #>qh
    sta tm2+1
    inc .Lax_terms
    lda tiny_mac_rows
    sta $C073
    sta $C003
    ldx #AXN-1
.Lax_loop:
.Lax_value:
    lda $ffff,x
    eor #$80
    tay
    sec
    lda (tp1),y
    sbc (tm1),y
    sta pl
    lda (tp2),y
    sbc (tm2),y
    tay
    clc
    lda pl
    adc .Lax_lo,x
    sta .Lax_lo,x
    tya
    adc .Lax_mid,x
    sta .Lax_mid,x
    bcc .Lax_next
    inc .Lax_hi,x
.Lax_next:
    dex
    bpl .Lax_loop
    sta $C002
    stz $C073
    pla
    bmi .Lax_storeok
    sta $C000
.Lax_storeok:
    pla
    bmi .Lax_hiresok
    sta $C056
.Lax_hiresok:
    pla
    bpl .Lax_pageok
    sta $C055
.Lax_pageok:
    plp
    rts
.Lax_terms: .byte 0

tiny_mac65_axpy_end:
    lda tiny_mac_acc
    sta cp
    lda tiny_mac_acc+1
    sta cp+1
    ; terms*$4000 mod 2^24 has bytes [0, (terms&3)<<6, terms>>2].
    lda .Lax_terms
    lsr a
    sta ph
    lda #0
    ror a
    sta pl
    lsr ph
    ror pl
    ldx #0
    ldy #0
.Lax_out:
    lda .Lax_lo,x
    sta (cp),y
    iny
    sec
    lda .Lax_mid,x
    sbc pl
    sta (cp),y
    iny
    lda .Lax_hi,x
    sbc ph
    sta (cp),y
    iny
    asl a                   ; sign of the 24-bit sum
    lda #0
    sbc #0
    eor #$ff
    sta (cp),y
    iny
    inx
    cpx #AXN
    bne .Lax_out
    rts
; Turn a batch of key results into attention scores.
; tiny_mac_b = eight-byte records from tiny_mac65_aux_rows, tiny_mac_rows =
; count. For each record: when tiny_att_first is set, its first four bytes
; are the cached position's scale {m, exp} and are saved at tiny_att_ks;
; the score goes to tiny_att_out, its exponent base+exp+shift to
; tiny_att_exps, and tiny_att_hi tracks the highest magnitude bit
; (-32768 while every score is zero). The three pointers advance.
; tiny_att_args supplies the query peak and the attention factor.
.global tiny_att_batch
abn = $11
abhi = $12
ablo = $13
tiny_att_batch:
    lda tiny_mac_b
    sta cp
    lda tiny_mac_b+1
    sta cp+1
    lda tiny_att_ks
    sta ap
    lda tiny_att_ks+1
    sta ap+1
    lda tiny_att_exps
    sta bp
    lda tiny_att_exps+1
    sta bp+1
    lda tiny_att_out
    sta $02
    lda tiny_att_out+1
    sta $03
    lda #<tiny_att_args
    sta $04
    lda #>tiny_att_args
    sta $05
    lda tiny_mac_rows
    sta abn
.Lab_loop:
    lda tiny_att_first
    beq .Lab_have
    ldy #3
.Lab_scale:
    lda (cp),y
    sta (ap),y
    dey
    bpl .Lab_scale
.Lab_have:
    ldy #4
    lda (cp),y
    sta tiny_att_args
    iny
    lda (cp),y
    sta tiny_att_args+1
    iny
    lda (cp),y
    sta tiny_att_args+2
    iny
    lda (cp),y
    sta tiny_att_args+3
    ldy #0
    lda (ap),y
    sta tiny_att_args+6
    iny
    lda (ap),y
    sta tiny_att_args+7
    jsr tiny_att_score      ; A = shift; $02-$05 are preserved
    clc
    adc tiny_att_base
    sta ablo
    lda tiny_att_base+1
    adc #0
    sta abhi
    ldy #2
    clc
    lda ablo
    adc (ap),y
    sta ablo
    iny
    lda abhi
    adc (ap),y
    sta abhi
    ldy #0
    lda ablo
    sta (bp),y
    iny
    lda abhi
    sta (bp),y
    lda tiny_att_args+12
    cmp #$ff
    beq .Lab_next           ; zero score
    clc
    adc ablo
    sta ablo
    bcc .Lab_cmp
    inc abhi
.Lab_cmp:
    lda tiny_att_hi         ; signed hi < value?
    cmp ablo
    lda tiny_att_hi+1
    sbc abhi
    bvc .Lab_sign
    eor #$80
.Lab_sign:
    bpl .Lab_next
    lda ablo
    sta tiny_att_hi
    lda abhi
    sta tiny_att_hi+1
.Lab_next:
    clc
    lda $02
    adc #8
    sta $02
    bcc .Lab_n1
    inc $03
.Lab_n1:
    clc
    lda cp
    adc #8
    sta cp
    bcc .Lab_n2
    inc cp+1
.Lab_n2:
    clc
    lda ap
    adc #4
    sta ap
    bcc .Lab_n3
    inc ap+1
.Lab_n3:
    clc
    lda bp
    adc #2
    sta bp
    bcc .Lab_n4
    inc bp+1
.Lab_n4:
    dec abn
    beq .Lab_done
    jmp .Lab_loop
.Lab_done:
    lda $02
    sta tiny_att_out
    lda $03
    sta tiny_att_out+1
    lda ap
    sta tiny_att_ks
    lda ap+1
    sta tiny_att_ks+1
    lda bp
    sta tiny_att_exps
    lda bp+1
    sta tiny_att_exps+1
    rts
tiny_mac65_aux_end:
