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
extern uint16_t tiny_mac_stride;
extern uint8_t tiny_mac_rows;
void tiny_mac65_dot(void){unsigned i;tiny_mac_result=0;for(i=0;i<tiny_mac_n;i++)tiny_mac_result+=(int32_t)tiny_mac_a[i]*tiny_mac_b[i];}
void tiny_mac65_axpy(void){unsigned i;for(i=0;i<tiny_mac_n;i++)tiny_mac_acc[i]+=(int32_t)tiny_mac_coeff*tiny_mac_b[i];}
/* Model just the prepared kernel contract, not the MMU or the native ABI. */
#include <assert.h>
static const uint8_t *pos;
static const int8_t *weights;
int8_t tiny_mac65_work[512];
void tiny_mac65_init(void){}
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
void tiny_mac65_aux_rows(void){
    unsigned k,i,j,b=(uint8_t)tiny_mac_coeff,a=(unsigned)(uintptr_t)tiny_mac_a;
    uint8_t *out=(uint8_t *)(uintptr_t)tiny_mac_b;
    assert(tiny_mac_rows>=1&&tiny_mac_rows<=8&&b>0&&b<=TINY_RW_LAST_BANK&&a>=TINY_RW_LO);
    assert(a+(tiny_mac_rows-1u)*tiny_mac_stride+4u+tiny_mac_n<=0xc000);
    assert(pos==(const uint8_t *)tiny_mac65_work);
    for(k=0;k<tiny_mac_rows;k++){
        const uint8_t *rec=ram[b]+a+k*tiny_mac_stride;uint32_t u;int32_t sum=0;
        for(i=0;i<tiny_mac_n;i++)sum+=((int32_t)pos[i]-128)*(int8_t)rec[4+i];
        memcpy(out+8*k,rec,4);u=(uint32_t)sum;
        for(j=0;j<4;j++)out[8*k+4+j]=(uint8_t)(u>>(8*j));
    }
}
static int32_t axpy_sum[24];
void tiny_mac65_axpy_begin(void){memset(axpy_sum,0,sizeof axpy_sum);}
void tiny_mac65_aux_axpy(void){
    unsigned i,b=tiny_mac_rows,a=(unsigned)(uintptr_t)tiny_mac_a;
    assert(b>0&&b<=TINY_RW_LAST_BANK&&a>=TINY_RW_LO&&a+24u<=0xc000&&tiny_mac_coeff>0);
    for(i=0;i<24;i++)axpy_sum[i]+=(int32_t)tiny_mac_coeff*(int8_t)ram[b][a+i];
}
void tiny_mac65_axpy_end(void){memcpy(tiny_mac_acc,axpy_sum,sizeof axpy_sum);}
