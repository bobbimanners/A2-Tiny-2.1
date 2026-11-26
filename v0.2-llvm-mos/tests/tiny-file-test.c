/* Verify ProDOS OPEN/READ/CLOSE separately from RamWorks and inference. */
#include <stdio.h>
#include <stdint.h>
#include "tiny-prodos.h"
int main(void) {
    TinyModel *f=tiny_model_open("TINY.INT");
    static uint8_t b[257];size_t n,i;uint32_t size=0,hash=2166136261UL;
    if(!f){printf("OPEN error $%02X\n",(unsigned)tiny_model_last_error);goto fail;}
    while((n=tiny_model_read(f,b,sizeof(b)))!=0){
        size+=(uint32_t)n;
        for(i=0;i<n;i++)hash=(hash^b[i])*16777619UL;
    }
    if(tiny_model_error(f)){printf("READ error $%02X\n",(unsigned)tiny_model_last_error);tiny_model_abort();goto fail;}
    if(tiny_model_close(f)){printf("CLOSE error $%02X\n",(unsigned)tiny_model_last_error);goto fail;}
    printf("Bytes: %lu\nFNV-1a: %08lX\n",(unsigned long)size,(unsigned long)hash);
    puts("Press a key.");(void)getchar();return 0;
fail:
    puts("Press a key.");(void)getchar();return 1;
}
