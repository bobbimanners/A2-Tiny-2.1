; Isolated sim65 benchmark of per-vector F2 column unrolling. Appended to a
; temporary copy of the production MAC file to reuse its table, row bodies,
; sparse preparation, result correction, bank setup and tail batches.
; Input data, bank splits and expected dots come from bench-f2-unroll.py.
.section .text,"ax",@progbits
.global bench_start, bench_rows, bench_slots, bench_mode, bench_active, bench_expected
.global tiny_mac_a,tiny_mac_b,tiny_mac_n,tiny_mac_result,tiny_mac_coeff
.global tiny_mac_stride,tiny_mac_rows,tiny_mac_acc,tiny_att_score
.global tiny_att_args,tiny_att_out,tiny_att_exps,tiny_att_ks,tiny_att_base
.global tiny_att_first,tiny_att_hi
tiny_mac_a: .space 2
tiny_mac_b: .space 2
tiny_mac_n: .space 2
tiny_mac_result: .space 4
tiny_mac_coeff: .space 1
tiny_mac_stride: .space 2
tiny_mac_rows: .space 1
tiny_mac_acc: .space 2
tiny_att_score: rts
tiny_att_args: .space 13
tiny_att_out: .space 2
tiny_att_exps: .space 2
tiny_att_ks: .space 2
tiny_att_base: .space 2
tiny_att_first: .space 1
tiny_att_hi: .space 2
bench_mode: .byte 0
bench_active: .byte 0
bench_index: .byte 0
bench_rows: .space 16*3           ; address and row count for each batch; zero terminates
bench_expected = $0800           ; raw scales and sign-extended expected sums
bench_out = $0c00
bench_start:
    cld
    jsr tiny_mac65_init
    lda #<tiny_mac65_work
    sta tiny_mac_b
    lda #>tiny_mac65_work
    sta tiny_mac_b+1
    stz tiny_mac_n
    lda #1
    sta tiny_mac_n+1             ; 256 activation bytes
    jsr tiny_mac65_prepare
    lda bench_mode
    beq .Lbench_run
    cmp #2
    beq .Lbench_run              ; mode 2 uses host-prepatched candidate for a lower bound
    jsr .Lbench_compile
.Lbench_run:
    lda #4
    sta tiny_mac_stride
    lda #1
    sta tiny_mac_stride+1        ; four scale bytes plus 256 weights per row
    sta tiny_mac_coeff
    lda #<bench_out
    sta tiny_mac_b
    lda #>bench_out
    sta tiny_mac_b+1
    stz bench_index
.Lbench_batch:
    ldx bench_index
    lda bench_rows+2,x
    beq .Lbench_verify
    sta tiny_mac_rows
    lda bench_rows,x
    sta tiny_mac_a
    lda bench_rows+1,x
    sta tiny_mac_a+1
    jsr tiny_mac65_aux_rows
    lda tiny_mac_rows
    asl a
    asl a
    asl a
    clc
    adc tiny_mac_b
    sta tiny_mac_b
    bcc .Lbench_advance
    inc tiny_mac_b+1
.Lbench_advance:
    lda bench_index
    clc
    adc #3
    sta bench_index
    bra .Lbench_batch
.Lbench_verify:
    ldx #0
.Lbench_check:
    lda bench_out,x
    cmp bench_expected,x
    bne .Lbench_fail
    lda bench_out+256,x
    cmp bench_expected+256,x
    bne .Lbench_fail
    lda bench_out+512,x
    cmp bench_expected+512,x
    bne .Lbench_fail
    inx
    bne .Lbench_check
    lda #0
    jmp $fff9                   ; sim65 exit
.Lbench_fail:
    lda #1
    jmp $fff9

; All modes include the same dispatcher. Only full eight-row batches can call
; the straight-line body. Restore the original LDA absolute,X chain load
; after each candidate call, so partial batches still use the ordinary loop.
.Lbench_dispatch:
    lda bench_active
    beq .Lbench_fallback
    lda rrows
    cmp #8
    bne .Lbench_fallback
    lda #$60
    sta .Lbench_tail
    jsr bench_slots
    lda #$bd
    sta .Lbench_tail
    jmp .Lrows_collect
.Lbench_fallback:
    ldx .Lprepared_first+1
    jmp .Lrows_column

; The 60 full call slots are preinstalled. Patch only X, u and 256-u in
; each 17-byte slot, then overwrite the first unused opcode with RTS.
; Restore the previous terminating slot on the next vector.
.Lbench_compile:
    stz bench_active
    lda .Lprepared_count+1
    bne .Lbench_done
    lda .Lprepared_count
    beq .Lbench_done
    cmp #61
    bcs .Lbench_done
    inc bench_active
    lda #$a2
.Lbench_restore:
    sta .Lbench_restore_dummy
    lda #>(bench_slots+1)
    sta .Lbench_col+2
    lda #>(bench_slots+3)
    sta .Lbench_u+2
    lda #>(bench_slots+9)
    sta .Lbench_mu+2
    lda #>bench_slots
    sta .Lbench_end+2
    ldx .Lprepared_first+1
    ldy #0
.Lbench_emit:
    txa
.Lbench_col:
    sta bench_slots+1,y
    lda tiny_mac65_work,x
.Lbench_u:
    sta bench_slots+3,y
    eor #$ff
    inc a
.Lbench_mu:
    sta bench_slots+9,y
    tya
    clc
    adc #17
    tay
    bcc .Lbench_next
    inc .Lbench_col+2
    inc .Lbench_u+2
    inc .Lbench_mu+2
    inc .Lbench_end+2
.Lbench_next:
    lda tiny_mac65_work+256,x
    tax
    bne .Lbench_emit
    lda #$60
.Lbench_end:
    sta bench_slots,y
    tya
    clc
    adc #<bench_slots
    sta .Lbench_restore+1
    lda .Lbench_end+2
    adc #0
    sta .Lbench_restore+2
.Lbench_done:
    rts
.Lbench_restore_dummy: .byte 0
.p2align 8
bench_slots:
.rept 60
    ldx #0
    lda #0
    sta tp1
    sta tp2
    lda #0
    sta tm1
    sta tm2
    jsr .Lrow7
.endr
    rts
