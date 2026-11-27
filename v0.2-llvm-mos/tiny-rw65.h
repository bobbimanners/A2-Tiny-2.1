#ifndef TINY_RW65_H
#define TINY_RW65_H
#include <stdint.h>
/* Non-reentrant. Aux range $4000..$BFFF; main buffer in $0000..$BEFF
 * (must not overlap registers, stack or the executing routine).
 * Assumes main ZP/stack and main read/write on entry, bank 0 outside calls.
 * The main HGR page must be visible at entry (not all of 80STORE,
 * HIRES and PAGE2 set). No transfer may cross $C000. IRQs masked. */
extern uint8_t tiny_rw_bank, tiny_rw_direction;
extern uint16_t tiny_rw_src, tiny_rw_count, tiny_rw_dst;
void tiny_rw65_move(void);
extern const uint8_t tiny_rw65_move_end[];
#define TINY_RW_LAST_BANK 15u
#define TINY_RW_LO 0x4000u
void tiny_rw_read(unsigned bank, unsigned off, void *dst, unsigned n);
void tiny_rw_write(unsigned bank, unsigned off, const void *src, unsigned n);
/* Returns zero on success; otherwise a suspect bank (1..15), or 255 if code is misplaced.
 * Temporarily writes one byte per bank, then restores original bytes. */
unsigned tiny_rw_probe(void);
#endif
