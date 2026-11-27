/* Optional native ABI regression. Link with tiny-wide-mos.s and
 * tiny-qsquare-mos.s using the same LLVM-MOS Apple II target as TINY.
 * Example (from v0.2 directory):
 * mos-apple2-clang -mcpu=mos65c02 -Oz -I. tests/tiny-q8-test.c \
 *   tiny-wide-mos.s tiny-qsquare-mos.s -o Q8TEST.SYSTEM
 * Import the SYS file into ProDOS exactly as for WIDETEST.
 */
#include <stdint.h>
#include <stdio.h>
#include "tiny-wide.h"
static int16_t input[256];
static int8_t output[258];
static uint16_t state=12345;
static uint16_t next(void) {
    state=(uint16_t)(state*25173u+13849u);return state;
}
int main(void) {
    static const uint16_t peaks[]={1,2,3,127,128,129,255,256,257,16384,32766,32767};
    static const uint16_t sizes[]={1,2,24,96,128,255,256};
    unsigned i,j,k,checks=0;uint16_t args[2];
    for(i=0;i<sizeof(peaks)/sizeof(peaks[0]);i++){
        uint16_t m=peaks[i];
        for(j=0;j<sizeof(sizes)/sizeof(sizes[0]);j++){
            uint16_t n=sizes[j];args[0]=n;args[1]=m;
            for(k=0;k<n;k++){
                int32_t a=(int32_t)(next()%((uint32_t)m+1u));
                input[k]=(int16_t)((next()&0x100u)?-a:a);
            }
            input[0]=(int16_t)m;
            if(n>1)input[1]=-(int16_t)m;
            if(n>2)input[2]=0;
            if(n>3)input[3]=(int16_t)(m/2);
            output[0]=55;output[n+1]=-55;
            tiny_q8_vector(input,output+1,args);
            if(output[0]!=55||output[n+1]!=-55){puts("Q8 GUARD FAIL");return 1;}
            for(k=0;k<n;k++){
                int32_t a=input[k],mag=a<0?-a:a;
                int16_t q=(int16_t)((mag*127L+m/2u)/m);
                if(a<0)q=-q;
                if(output[k+1]!=q){puts("Q8 VALUE FAIL");return 1;}
                checks++;
            }
        }
    }
    printf("Q8 PASS: %u values\n",checks);return 0;
}
