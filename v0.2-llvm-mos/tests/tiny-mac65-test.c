#include <stdint.h>
#include <stdio.h>
#include <string.h>
const int8_t *tiny_mac_a, *tiny_mac_b;
uint16_t tiny_mac_n;
int32_t tiny_mac_result;
int8_t tiny_mac_coeff;
int32_t *tiny_mac_acc;
uint16_t tiny_mac_stride;
uint8_t tiny_mac_rows;
extern void tiny_mac65_dot(void);
extern void tiny_mac65_axpy(void);
static int8_t a[516], b[516];
static int32_t acc[130], ref[130];
static uint32_t rng=123456789UL;
static int8_t sample(void) {
    rng=rng*1664525UL+1013904223UL;
    return (int8_t)((int)(rng>>24)-128);
}
static int32_t dot(const int8_t *p,const int8_t *q,int n) {
    int32_t sum=0;
    while(n>511){tiny_mac_a=p;tiny_mac_b=q;tiny_mac_n=256;
        tiny_mac65_dot();sum+=tiny_mac_result;p+=256;q+=256;n-=256;}
    tiny_mac_a=p;tiny_mac_b=q;tiny_mac_n=(uint16_t)n;tiny_mac65_dot();
    return sum+tiny_mac_result;
}
static int check(const int8_t *p,const int8_t *q,int n) {
    int i;int32_t expected=0,actual;
    for(i=0;i<n;i++)expected+=(int32_t)((int16_t)p[i]*(int16_t)q[i]);
    actual=dot(p,q,n);
    if(actual!=expected){printf("dot n=%d got %ld expected %ld\n",n,(long)actual,(long)expected);return 0;}
    return 1;
}
static int run(void) {
    static const int lengths[]={0,1,2,24,95,96,127,128,255,256,257,511,512};
    static const int edge[]={-128,-127,-64,-1,0,1,63,127};
    int i,j,k,n,c,full=0;

#ifdef TINY_MAC_FULL
    full=1;
#endif
#ifdef TINY_MAC_BENCH
    {
        int32_t checksum=0;
        for(i=0;i<256;i++){a[i]=sample();b[i]=sample();}
        tiny_mac_a=a;tiny_mac_b=b;tiny_mac_n=256;
        for(k=0;k<1000;k++){tiny_mac65_dot();checksum+=tiny_mac_result;}
        printf("256000 MACs, checksum %ld\n",(long)checksum);return checksum!=41417000L;
    }
#endif
    for(i=0;i<(full?256:8);i++)for(j=0;j<(full?256:8);j++){
        a[0]=(int8_t)(full?i-128:edge[i]);b[0]=(int8_t)(full?j-128:edge[j]);
        if(!check(a,b,1))return 1;
    }
    for(i=0;i<516;i++)a[i]=b[i]=-128;
    if(!check(a,b,512)||!check(a,b,511))return 2;
    for(i=0;i<516;i++)b[i]=127;
    if(!check(a,b,512))return 3;
    for(k=0;k<40;k++){
        for(i=0;i<516;i++){a[i]=sample();b[i]=sample();}
        for(j=0;j<(int)(sizeof(lengths)/sizeof(lengths[0]));j++)
            if(!check(a+(k&3),b+((k+1)&3),lengths[j]))return 4;
    }
    for(n=0;n<=128;n+=24){
        memset(acc,0,sizeof(acc));memset(ref,0,sizeof(ref));
        acc[0]=ref[0]=123456L;acc[n+1]=ref[n+1]=-123456L;
        for(k=0;k<128;k++){
            c=sample();for(i=0;i<n;i++)b[i]=sample();
            tiny_mac_coeff=(int8_t)c;tiny_mac_b=b;tiny_mac_acc=acc+1;
            tiny_mac_n=(uint16_t)n;tiny_mac65_axpy();
            for(i=0;i<n;i++)ref[i+1]+=(int32_t)((int16_t)c*(int16_t)b[i]);
            for(i=0;i<n+2;i++)if(acc[i]!=ref[i]){printf("axpy n=%d step=%d slot=%d\n",n,k,i);return 5;}
        }
    }
    /* Explicit 128-wide extreme test, including sign extension. */
    memset(acc,0,sizeof(acc));memset(b,0x80,128);
    tiny_mac_coeff=-128;tiny_mac_b=b;tiny_mac_acc=acc;tiny_mac_n=128;
    for(k=0;k<128;k++)tiny_mac65_axpy();
    for(i=0;i<128;i++)if(acc[i]!=2097152L)return 6;
    tiny_mac_coeff=127;
    for(k=0;k<128;k++)tiny_mac65_axpy();
    for(i=0;i<128;i++)if(acc[i]!=16384L)return 7;
    puts("PASS: signed products, dot lengths/page crossings, weighted value accumulation");
    return 0;
}

int main(void){int r=run();printf("Status: %d\nPress a key.\n",r);(void)getchar();return r;}
