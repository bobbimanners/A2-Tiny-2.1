/* Host-only hardware substitutes to test the Apple frontend and model reader.
 * These are NOT tests of the real assembly, compiler ABI or Apple hardware. */
#include <stdint.h>
#include <string.h>
#include "tiny-rw65.h"
static uint8_t ram[TINY_RW_LAST_BANK+1][65536];
void tiny_rw_read(unsigned b,unsigned a,void *p,unsigned n){memcpy(p,ram[b]+a,n);}
void tiny_rw_write(unsigned b,unsigned a,const void *p,unsigned n){memcpy(ram[b]+a,p,n);}
unsigned tiny_rw_probe(void){return 0;}
extern const int8_t *tiny_mac_a,*tiny_mac_b;
extern uint16_t tiny_mac_n;
extern int32_t tiny_mac_result,*tiny_mac_acc;
extern int8_t tiny_mac_coeff;
void tiny_mac65_dot(void){unsigned i;tiny_mac_result=0;for(i=0;i<tiny_mac_n;i++)tiny_mac_result+=(int32_t)tiny_mac_a[i]*tiny_mac_b[i];}
void tiny_mac65_axpy(void){unsigned i;for(i=0;i<tiny_mac_n;i++)tiny_mac_acc[i]+=(int32_t)tiny_mac_coeff*tiny_mac_b[i];}
/* Model just the prepared kernel contract, not the MMU or the native ABI. */
#include <assert.h>
static const uint8_t *pos;
static const int8_t *weights;
int8_t tiny_mac65_work[512];
void tiny_mac65_prepare(void){
    unsigned i;uint8_t *p=(uint8_t *)(uintptr_t)tiny_mac_b;
    weights=tiny_mac_a;pos=p;
    for(i=0;i<tiny_mac_n;i++){p[i]^=128;p[256+i]=(uint8_t)(256-p[i]);}
}
void tiny_mac65_prepared_dot(void){
    unsigned i;tiny_mac_result=0;
    for(i=0;i<tiny_mac_n;i++)tiny_mac_result+=((int32_t)pos[i]-128)*weights[i];
}
void tiny_mac65_aux_dot(void){
    unsigned i,b=(uint8_t)tiny_mac_coeff,a=(unsigned)(uintptr_t)tiny_mac_a;
    assert(b>0&&b<=TINY_RW_LAST_BANK&&a>=TINY_RW_LO&&a+4+tiny_mac_n<=0xc000);
    assert(pos==(const uint8_t *)tiny_mac65_work);
    memcpy((void *)(uintptr_t)tiny_mac_b,ram[b]+a,4);
    weights=(const int8_t *)(ram[b]+a+4);tiny_mac65_prepared_dot();
}
