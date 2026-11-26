; LLVM-MOS: three ProDOS MLI calls with inline parameter pointers.
; A returns the 8-bit error code (zero on success). Preserve the entire
; imaginary register file across firmware calls, including the soft SP.
; The Apple II SDK maps __rc0..__rc31 contiguously to $00..$1F.
.section .bss,"aw",@nobits
; Keep MLI parameter lists and pathname in ordinary main RAM, not
; compiler-allocated zero page (LTO cannot move these assembly objects).
.global tiny_pd_open_pb, tiny_pd_read_pb, tiny_pd_close_pb, tiny_pd_pathname
tiny_pd_open_pb: .space 6
tiny_pd_read_pb: .space 8
tiny_pd_close_pb: .space 2
tiny_pd_pathname: .space 65
.Lpd_zp: .space 32
.Lpd_status: .space 1
.section .text,"ax",@progbits
.global tiny_mli_open, tiny_mli_read, tiny_mli_close
.Lpd_save:
    ldx #31
1:
    lda $00,x
    sta .Lpd_zp,x
    dex
    bpl 1b
    cld
    rts
.Lpd_restore:
    bcs 1f
    lda #0
1:
    sta .Lpd_status
    ldx #31
2:
    lda .Lpd_zp,x
    sta $00,x
    dex
    bpl 2b
    lda .Lpd_status
    cld
    rts
tiny_mli_open:
    jsr .Lpd_save
    jsr $bf00
    .byte $c8
    .short tiny_pd_open_pb
    jmp .Lpd_restore
tiny_mli_read:
    jsr .Lpd_save
    jsr $bf00
    .byte $ca
    .short tiny_pd_read_pb
    jmp .Lpd_restore
tiny_mli_close:
    jsr .Lpd_save
    jsr $bf00
    .byte $cc
    .short tiny_pd_close_pb
    jmp .Lpd_restore
