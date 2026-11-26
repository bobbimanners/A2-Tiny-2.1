#ifndef TINY_RW65_H
#define TINY_RW65_H
#include <stdint.h>
/* Non-reentrant. Main-memory buffer must be in $0800..$BEFF.
 * Assumes main ZP/stack and main read/write on entry, bank 0 outside calls.
 * No transfer may cross $C000. Interrupts masked during transfer. */
extern uint8_t tiny_rw_bank, tiny_rw_direction;
extern uint16_t tiny_rw_src, tiny_rw_end, tiny_rw_dst;
void tiny_rw65_move(void);
void tiny_rw_read(unsigned bank, unsigned off, void *dst, unsigned n);
void tiny_rw_write(unsigned bank, unsigned off, const void *src, unsigned n);
/* Returns zero on success; otherwise a suspect bank (1..12).
 * Temporarily writes one byte per bank, then restores original bytes. */
unsigned tiny_rw_probe(void);
#endif
