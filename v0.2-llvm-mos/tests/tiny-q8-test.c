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
/* Check the shifted cutoff, the actual zero-to-one rounding transition,
 * both ends of each interval, and the full context batch length. */
static int test_coefficients(void) {
    static const uint64_t maxima[] = {0, 1, 127, 253, 254, 255, 256, 257,
        65535, 65536, 0xffffffffULL, 0x100000000ULL, 1ULL << 41};
    static Wide scores[128];
    unsigned i, j, checks = 0;
    for (i = 0; i < sizeof maxima / sizeof maxima[0]; i++) {
        uint64_t b = maxima[i], cutoff = b >> 8;
        Wide maximum = {(uint32_t)b, (uint32_t)(b >> 32)};
        uint8_t want[128];
        for (j = 0; j < 128; j++) {
            uint64_t a;
            switch (j) {
                case 0: a = 0; break;
                case 1: a = b; break;
                case 2: a = cutoff; break;
                case 3: a = cutoff < b ? cutoff + 1 : b; break;
                case 4: a = b / 254; break;
                case 5: a = (b + 253) / 254; break;
                default: a = (b * next()) >> 16; break;
            }
            scores[j].lo = (uint32_t)a;
            scores[j].hi = (uint32_t)(a >> 32);
            want[j] = b ? (uint8_t)((127 * a + b / 2) / b) : 0;
        }
        output[0] = output[129] = -99;
        tiny_attention_coeffs(scores, &maximum, output + 1, 128);
        for (j = 0; j < 128; j++) {
            if (output[j + 1] != want[j] ||
                tiny_attention_coeff(&scores[j], &maximum) != want[j]) {
                printf("COEFFICIENT FAIL: %u, %u\n", i, j);return 1;
            }
            checks += 2;
        }
        if (output[0] != -99 || output[129] != -99) {puts("COEFFICIENT GUARD FAIL");return 1;}
    }
    printf("COEFFICIENT PASS: %u values\n",checks);
    return 0;
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
    printf("Q8 PASS: %u values\n",checks);return test_coefficients();
}
