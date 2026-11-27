#include "tiny-rw65.h"
#include <stddef.h>
uint8_t tiny_rw_bank, tiny_rw_direction;
uint16_t tiny_rw_src, tiny_rw_count, tiny_rw_dst;
static void move(unsigned bank,unsigned src,unsigned dst,unsigned n,unsigned direction)
{
    if(!n)return;
    tiny_rw_bank=(uint8_t)bank;
    tiny_rw_src=(uint16_t)src;
    tiny_rw_count=(uint16_t)n;
    tiny_rw_dst=(uint16_t)dst;
    tiny_rw_direction=(uint8_t)direction;
    tiny_rw65_move();
}
void tiny_rw_read(unsigned bank,unsigned off,void *dst,unsigned n)
{
    move(bank,off,(unsigned)(uintptr_t)dst,n,0);
}
void tiny_rw_write(unsigned bank,unsigned off,const void *src,unsigned n)
{
    move(bank,(unsigned)(uintptr_t)src,off,n,1);
}
/* Restore even on failure. Detect mirrored banks including bank zero.
 * This is not a complete RAM test; it is a preflight for the 15 used banks. */
unsigned tiny_rw_probe(void)
{
    static uint8_t old[TINY_RW_LAST_BANK+1], value;
    unsigned b,bad=0;
    /* Fail before switching banks if the linker moved the routine. */
    if((uintptr_t)tiny_rw65_move<0x2000u ||
       (uintptr_t)tiny_rw65_move_end>0x4000u)return 255u;
    for(b=0;b<=TINY_RW_LAST_BANK;b++)tiny_rw_read(b,TINY_RW_LO,&old[b],1);
    for(b=1;b<=TINY_RW_LAST_BANK;b++){
        value=(uint8_t)(old[0]+b);
        tiny_rw_write(b,TINY_RW_LO,&value,1);
    }
    for(b=0;b<=TINY_RW_LAST_BANK;b++){
        tiny_rw_read(b,TINY_RW_LO,&value,1);
        if(value!=(uint8_t)(old[0]+b)&&!bad)bad=b?b:1;
    }
    for(b=TINY_RW_LAST_BANK;b>0;b--)tiny_rw_write(b,TINY_RW_LO,&old[b],1);
    return bad;
}
