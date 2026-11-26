/* Destructive test of banks 1..12 only; do not run with a RamWorks RAM disk. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "tiny-rw65.h"
static uint8_t out[520], in[520];
static uint8_t pattern(unsigned b,unsigned i){return (uint8_t)(b*19u+i*13u+(i>>8));}
static const unsigned offs[]={0x2000u,0x23fdu,0xbdfcu};
static const unsigned lens[]={1,516,516};
static int run(void)
{
    unsigned b,k,i,bad=tiny_rw_probe();
    if(bad){printf("Bank probe FAIL: %u\n",bad);return 1;}
    puts("Bank alias probe PASS");
    for(b=1;b<=12;b++)for(k=0;k<3;k++){
        for(i=0;i<lens[k];i++)out[i]=pattern(b,i);
        tiny_rw_write(b,offs[k],out,lens[k]);
    }
    for(b=1;b<=12;b++)for(k=0;k<3;k++){
        memset(in,0xa5,sizeof(in));
        tiny_rw_read(b,offs[k],in+1,lens[k]);
        if(in[0]!=0xa5||in[lens[k]+1]!=0xa5){puts("Buffer guard FAIL");return 1;}
        for(i=0;i<lens[k];i++)if(in[i+1]!=pattern(b,i)){
            printf("FAIL bank %u addr %04X offset %u\n",b,offs[k],i);return 1;
        }
    }
    puts("PASS: 12 banks, independent data, page crossings, $BFFF end, guards");
    return 0;
}

int main(void){int r=run();puts("Press a key.");(void)getchar();return r;}
