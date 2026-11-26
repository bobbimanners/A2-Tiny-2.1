/* TINY stage 8B: integer-only transformer inference and diagnostics.
 * Model architecture from WollyMonjac's Tiny 2.1.
 * https://github.com/WollyMonjac/Tiny-2.1 (MIT)
 * Convert original TNY2 file offline using tiny-convert.py.
 * Build: cc -std=c99 -O2 tiny-stage8a.c -o tiny-stage8a
 * Linux reference or ca65 65C02 MAC backend (TINY_ASM_MAC).
 * Define TINY_NATIVE_MUL to retain the Stage 7A multiply kernel.
 * Linux simulation of a RamWorks data window; no flat model pointers.
 * TINY_APPLE2 selects native RamWorks transfers and a compact ProDOS UI.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#define U32(x) ((uint32_t)(x##UL))
#define VOCAB 97
#define CTX 128
#ifdef __APPLE2ENH__
#ifndef TINY_APPLE2
#define TINY_APPLE2
#endif
#endif
#ifdef TINY_APPLE2
#include "tiny-rw65.h"
#ifndef TINY_ASM_MAC
#define TINY_ASM_MAC
#endif
#endif
#if defined(TINY_APPLE2) || defined(TINY_SMALL)
#define MAXD 96
#define MAXHID 256
#define MAXL 4
#else
#define MAXD 128
#define MAXHID 512
#define MAXL 8
#endif
#define MAXQ 72
/* A vector represents v[i] * 2^exp. Zero vectors have exp = 0.
 * Each table has one binary exponent shared by its 24-bit row scales. */
typedef struct { int16_t v[MAXHID]; int exp; } Vec;
/* A region contains fixed-size records, none crossing a bank boundary. */
typedef struct { uint8_t bank; uint16_t off, first, per, stride, rows; } Region;
typedef struct { int rows, cols, exp; Region data; } Table;
typedef struct { int exp; Region data; } BankGain;
typedef struct { int32_t v[MAXD]; int exp; } Gain;
/* cache scale = m * 2^exp / 127 (m is the peak before quantization). */
typedef struct { uint16_t m; int exp; } CacheScale;
typedef struct { Table q,k,v,o,f1,f2; BankGain g1,g2;
    Region key,value; } Layer;
static Table emb,pos;
static Layer layers[MAXL];
static BankGain gf;
static Gain gain_work;
static int D,L,H,HD,HID, creativity=0, profiling=0;
static uint32_t att_factor; /* 2^44 / (127^2 * sqrt(HD)) */
static Vec score;
static Vec x,normed,tmp,query,att,hidden;
/* Wide scratch is declared after the arithmetic helpers. */
static uint32_t packs, zeros, rounded_zero, saturation;
static int min_exp=INT_MAX,max_exp=INT_MIN;
typedef struct { uint16_t m; int exp; } Peak;
static Peak peak_residual,peak_norm,peak_proj,peak_hidden,peak_att,peak_logit;

static void die(const char *s) { fprintf(stderr,"TINY: %s\n",s); exit(1); }
/* Reserve bank 0 for the normal auxiliary display/system environment.
 * Only $2000..$BFFF of banks 1..127 is available to the allocator.
 * This is a conservative proposed layout, not an Apple II MMU emulator.
 * No inference code may dereference simulated auxiliary memory directly. */
#define RW_BANKS 128u
#define RW_LO 0x2000u
#define RW_HI 0xc000u
#define RW_SIZE (RW_HI-RW_LO)
#ifndef TINY_APPLE2
static uint8_t rw_mem[RW_BANKS][65536UL];
#endif
static unsigned rw_next_bank=1, rw_next_off=RW_LO;
static unsigned rw_selected=0, rw_last_bank;
static uint32_t rw_payload, rw_padding;
static uint32_t rw_reads,rw_writes,rw_read_bytes,rw_write_bytes,rw_switches;
static int rw_report;
static void rw_reset_stats(void) {
    rw_reads=rw_writes=rw_read_bytes=rw_write_bytes=rw_switches=0;
    rw_selected=0; /* start each measured phase with bank 0 selected */
}
static void rw_next(void) {
    rw_padding+=(uint32_t)(RW_HI-rw_next_off);
    ++rw_next_bank;rw_next_off=RW_LO;
    if(rw_next_bank>=RW_BANKS)die("RamWorks capacity exceeded");
#ifdef TINY_APPLE2
    if(rw_next_bank>12u)die("model exceeds 12 verified banks");
#endif
}
static void rw_alloc(Region *r,unsigned rows,unsigned stride) {
    unsigned left=rows,avail,take;
    if(!rows||!stride||stride>RW_SIZE)die("invalid banked region");
    if((RW_HI-rw_next_off)/stride==0)rw_next();
    r->bank=(uint8_t)rw_next_bank;r->off=(uint16_t)rw_next_off;
    r->rows=(uint16_t)rows;r->stride=(uint16_t)stride;
    r->per=(uint16_t)(RW_SIZE/stride);
    avail=(RW_HI-rw_next_off)/stride;
    r->first=(uint16_t)(rows<avail?rows:avail);
    while(left){
        avail=(RW_HI-rw_next_off)/stride;
        if(!avail){rw_next();continue;}
        take=left<avail?left:avail;
        rw_next_off+=take*stride;left-=take;
        rw_last_bank=rw_next_bank;
    }
    rw_payload+=(uint32_t)rows*stride;
}
static void rw_addr(const Region *r,unsigned row,unsigned start,unsigned n,
                    unsigned *bank,unsigned *off) {
    if(row>=r->rows||start>r->stride||n>r->stride-start)die("banked record bounds");
    if(row<r->first){*bank=r->bank;*off=r->off+row*r->stride;}
    else{row-=r->first;*bank=(unsigned)r->bank+1u+row/r->per;*off=RW_LO+(row%r->per)*r->stride;}
    *off+=start;
    if(*bank==0||*bank>=RW_BANKS||*off<RW_LO||*off>RW_HI||n>RW_HI-*off)die("RamWorks window bounds");
}
static void rw_select(unsigned bank) {
    if(bank!=rw_selected){rw_selected=bank;++rw_switches;}
}
static void rw_read(const Region *r,unsigned row,unsigned start,void *dst,unsigned n) {
    unsigned bank,off;rw_addr(r,row,start,n,&bank,&off);rw_select(bank);
    #ifdef TINY_APPLE2
    tiny_rw_read(bank,off,dst,n);
#else
    memcpy(dst,&rw_mem[bank][off],n);
#endif
    ++rw_reads;rw_read_bytes+=n;
}
static void rw_write(const Region *r,unsigned row,unsigned start,const void *src,unsigned n) {
    unsigned bank,off;rw_addr(r,row,start,n,&bank,&off);rw_select(bank);
    #ifdef TINY_APPLE2
    tiny_rw_write(bank,off,src,n);
#else
    memcpy(&rw_mem[bank][off],src,n);
#endif
    ++rw_writes;rw_write_bytes+=n;
}
static void rw_stats(const char *phase) {
    if(!rw_report)return;
    fprintf(stderr,"RamWorks %s: banks=%u/%u payload=%lu padding=%lu window=%u bytes/bank\n",phase,rw_last_bank,RW_BANKS-1,(unsigned long)rw_payload,(unsigned long)rw_padding,RW_SIZE);
    fprintf(stderr,"  reads=%lu (%lu bytes), writes=%lu (%lu bytes), bank changes=%lu\n",(unsigned long)rw_reads,(unsigned long)rw_read_bytes,(unsigned long)rw_writes,(unsigned long)rw_write_bytes,(unsigned long)rw_switches);
}
/* Explicit little-endian storage: identical on Linux and 65C02. */
static uint32_t get32(const uint8_t *b) {
    return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
}
static int32_t signed32(uint32_t u) {return u<=INT32_MAX?(int32_t)u:-1-(int32_t)(~u);}
static void put32(uint8_t *b,uint32_t v) {
    unsigned i;for(i=0;i<4;i++){b[i]=(uint8_t)v;v>>=8;}
}
static unsigned get16(const uint8_t *b){return b[0]+256u*b[1];}
static int signed16(unsigned u){return u<32768u?(int)u:(int)((int32_t)u-65536L);}
static void put16(uint8_t *b,unsigned v){b[0]=(uint8_t)v;b[1]=(uint8_t)(v>>8);}
/* Older cc65 does not fold a conditional expression in an array bound. */
#if (MAXHID + 4) > (4 * MAXD)
#define ROWBUF_SIZE (MAXHID + 4)
#else
#define ROWBUF_SIZE (4 * MAXD)
#endif
static uint8_t rowbuf[ROWBUF_SIZE];
static void read_exact(FILE *f,void *p,size_t n) { if(fread(p,1,n,f)!=n) die("truncated model"); }
static unsigned read16(FILE *f) { unsigned char b[2]; read_exact(f,b,2); return b[0]+256u*b[1]; }
static uint32_t read32(FILE *f) {
    unsigned char b[4]; read_exact(f,b,4);
    return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
}
static int read_exp(FILE *f) {
    unsigned v=read16(f); int e=v<32768u?(int)v:(int)((int32_t)v-65536);
    if(e < -171 || e > 105)die("invalid model exponent");
    return e;
}
static int32_t read_coeff(FILE *f) {
    uint32_t u=read32(f);
    int32_t v=u<=INT32_MAX?(int32_t)u:-1-(int32_t)(~u);
    if(v < -8388608 || v > 8388608)die("invalid model coefficient");
    return (int32_t)v;
}
static void load_table(FILE *f,Table *t,int rows,int cols) {
    int r;int32_t c;t->rows=rows;t->cols=cols;t->exp=read_exp(f);
    rw_alloc(&t->data,(unsigned)rows,(unsigned)cols+4u);
    for(r=0;r<rows;r++){
        c=read_coeff(f);if(c<0)die("negative row scale");
        put32(rowbuf,(uint32_t)c);read_exact(f,rowbuf+4,cols);
        rw_write(&t->data,r,0,rowbuf,(unsigned)cols+4u);
    }
}
static void gains(FILE *f,BankGain *g) {
    int i;g->exp=read_exp(f);rw_alloc(&g->data,1,4u*(unsigned)D);
    for(i=0;i<D;i++)put32(rowbuf+4*i,(uint32_t)read_coeff(f));
    rw_write(&g->data,0,0,rowbuf,4u*(unsigned)D);
}
static void fetch_gain(const BankGain *g) {
    int i;gain_work.exp=g->exp;rw_read(&g->data,0,0,rowbuf,4u*(unsigned)D);
    for(i=0;i<D;i++)gain_work.v[i]=signed32(get32(rowbuf+4*i));
}
static void load_model(const char *path) {
    FILE *f=fopen(path,"rb");char magic[4];int l;
    if(!f)die("cannot open integer model (run tiny-convert.py first)");
    read_exact(f,magic,4);if(memcmp(magic,"TNI6",4))die("not a TNI6 integer model (run tiny-convert.py first)");
    { unsigned d=read16(f),lcount=read16(f),h=read16(f),hid=read16(f);
      if(d>MAXD||lcount>MAXL||h>MAXD||hid>MAXHID)die("unsupported model dimensions");
      D=(int)d;L=(int)lcount;H=(int)h;HID=(int)hid; }
    if(D<8||D>MAXD||L<1||L>MAXL||H<1||H>D||D%H||HID<8||HID>MAXHID)die("unsupported model dimensions");
    HD=D/H;att_factor=read32(f);
    if(!att_factor||att_factor>U32(1090717716))die("invalid attention factor");
    #ifdef TINY_APPLE2
    if(D!=96||L!=4||H!=4||HID!=256)die("Apple build requires standard 96/4/4/256 model");
    puts("Loading model...");
#endif
    load_table(f,&emb,VOCAB,D);load_table(f,&pos,CTX,D);
    for(l=0;l<L;l++) {Layer *a=&layers[l];gains(f,&a->g1);
        load_table(f,&a->q,D,D);load_table(f,&a->k,D,D);load_table(f,&a->v,D,D);load_table(f,&a->o,D,D);
        gains(f,&a->g2);load_table(f,&a->f1,HID,D);load_table(f,&a->f2,D,HID);
#ifdef TINY_APPLE2
        putchar('.');fflush(stdout);
#endif
    }
#ifdef TINY_APPLE2
    putchar('\n');
#endif
    gains(f,&gf);if(fgetc(f)!=EOF||ferror(f))die("unexpected model length");fclose(f);
    for(l=0;l<L;l++){
        rw_alloc(&layers[l].key,CTX,(unsigned)D+4u);
        rw_alloc(&layers[l].value,CTX,(unsigned)D+4u);
    }
}
/* Two-word, unsigned storage for signed two's-complement or unsigned
 * wide intermediates. All operations use at most 32-bit native types.
 * Pointer API: cc65 cannot pass or return 8-byte structs by value.
 * Arithmetic wraps modulo 2^64; callers retain Stage 6's range bounds. */
typedef struct { uint32_t lo, hi; } Wide;
static void wu(Wide *r,uint32_t a) {r->lo=a;r->hi=0;}
static void wi(Wide *r,int32_t a) {r->lo=(uint32_t)a;r->hi=a<0?UINT32_MAX:0;}
static int wz(const Wide *a) {return !(a->lo|a->hi);}
static int wn(const Wide *a) {return (a->hi>>31)!=0;}
static int wc(const Wide *a,const Wide *b) {
    if(a->hi!=b->hi)return a->hi>b->hi?1:-1;
    return a->lo==b->lo?0:a->lo>b->lo?1:-1;
}
static int wcs(const Wide *a,const Wide *b) {
    if(wn(a)!=wn(b))return wn(a)?-1:1;
    return wc(a,b);
}
static void wa(Wide *r,const Wide *a,const Wide *b) {
    uint32_t lo=a->lo+b->lo,hi=a->hi+b->hi+(lo<a->lo);
    r->lo=lo;r->hi=hi;
}
static void ws(Wide *r,const Wide *a,const Wide *b) {
    uint32_t lo=a->lo-b->lo,hi=a->hi-b->hi-(a->lo<b->lo);
    r->lo=lo;r->hi=hi;
}
static void wneg(Wide *r,const Wide *a) {
    uint32_t lo=0UL-a->lo,hi=~a->hi+(lo==0);
    r->lo=lo;r->hi=hi;
}
static void wmag(Wide *r,const Wide *a) {if(wn(a))wneg(r,a);else *r=*a;}
static void wl(Wide *r,const Wide *a,int n) {
    uint32_t lo,hi;
    if(n<=0){*r=*a;return;}
    if(n>=64){wu(r,0);return;}
    if(n>=32){hi=a->lo<<(n-32);lo=0;}
    else{hi=(a->hi<<n)|(a->lo>>(32-n));lo=a->lo<<n;}
    r->lo=lo;r->hi=hi;
}
static void wr(Wide *r,const Wide *a,int n) {
    uint32_t lo,hi;
    if(n<=0){*r=*a;return;}
    if(n>=64){wu(r,0);return;}
    if(n>=32){lo=a->hi>>(n-32);hi=0;}
    else{lo=(a->lo>>n)|(a->hi<<(32-n));hi=a->hi>>n;}
    r->lo=lo;r->hi=hi;
}
static void winc(Wide *a) {if(++a->lo==0)++a->hi;}
static unsigned wbit(const Wide *a,int n) {
    if(n<0||n>=64)return 0;
    return (unsigned)((n<32?a->lo>>n:a->hi>>(n-32))&1UL);
}
static int topbit(const Wide *a) {
    uint32_t v=a->hi?a->hi:a->lo;int n=a->hi?32:0;
    while(v>>=1)n++;
    return n;
}
/* Full 32x32 product via four 16x16 products, then cross terms. */
static void wm(Wide *r,const Wide *a,const Wide *b) {
    uint32_t al=a->lo&65535UL,ah=a->lo>>16,bl=b->lo&65535UL,bh=b->lo>>16;
    uint32_t p=al*bl,t=ah*bl+(p>>16),u=al*bh+(t&65535UL);
    uint32_t lo=(p&65535UL)|(u<<16);
    uint32_t hi=ah*bh+(t>>16)+(u>>16)+a->hi*b->lo+a->lo*b->hi;
    r->lo=lo;r->hi=hi;
}
static void wmu(Wide *r,const Wide *a,uint32_t b) {Wide v;wu(&v,b);wm(r,a,&v);}
static void wmi(Wide *r,const Wide *a,int32_t b) {Wide v;wi(&v,b);wm(r,a,&v);}
/* Unsigned division. Small divisors use four base-65536 digits; general
 * division uses shift/subtract. Output may alias either input. */
static void wd(Wide *r,const Wide *a,const Wide *b) {
    Wide q,rem,d;int n;
    uint32_t x,y,z,k;
    if(wz(b))die("wide divide by zero");
    if(!b->hi && b->lo<=65535UL){
        k=b->lo;x=a->hi>>16;y=x/k;z=x%k;
        x=(z<<16)|(a->hi&65535UL);q.hi=(y<<16)|(x/k);z=x%k;
        x=(z<<16)|(a->lo>>16);y=x/k;z=x%k;
        x=(z<<16)|(a->lo&65535UL);q.lo=(y<<16)|(x/k);*r=q;return;
    }
    if(wc(a,b)<0){wu(r,0);return;}
    rem=*a;wu(&q,0);n=topbit(a)-topbit(b);wl(&d,b,n);
    for(;n>=0;n--){
        if(wc(&rem,&d)>=0){ws(&rem,&rem,&d);if(n<32)q.lo|=U32(1)<<n;else q.hi|=U32(1)<<(n-32);}
        wr(&d,&d,1);
    }
    *r=q;
}
static void wdu(Wide *r,const Wide *a,uint32_t b) {Wide v;wu(&v,b);wd(r,a,&v);}
/* Unsigned round-to-nearest without overflow in numerator+denominator/2. */
static void wrounddiv(Wide *r,const Wide *a,const Wide *b) {
    Wide q,rem,t,half;wd(&q,a,b);wm(&t,&q,b);ws(&rem,a,&t);
    wr(&half,b,1);if(b->lo&1UL)winc(&half);
    if(wc(&rem,&half)>=0)winc(&q);
    *r=q;
}
static void rshift(Wide *r,const Wide *a,int n) {
    Wide m;int neg=wn(a);
    if(n<=0){*r=*a;return;}
    wmag(&m,a);wr(r,&m,n);if(wbit(&m,n-1))winc(r);
    if(neg)wneg(r,r);
}
static void rdiv(Wide *r,const Wide *a,uint32_t d) {
    Wide m,b;int neg=wn(a);wmag(&m,a);wu(&b,d);wrounddiv(r,&m,&b);if(neg)wneg(r,r);
}
static int32_t wtoi(const Wide *a) {
    if(a->hi==0 && a->lo<=INT32_MAX)return (int32_t)a->lo;
    if(a->hi==UINT32_MAX && a->lo>=U32(2147483648))return -1-(int32_t)(~a->lo);
    die("wide result does not fit signed 32 bits");return 0;
}
static Wide wide[MAXHID];
static void note(Vec *dst,int n,Peak *peak) {
    int i;int32_t m=0;
    if(!profiling)return;
    for(i=0;i<n;i++){int32_t a=dst->v[i];if(a<0)a=-a;if(a>m)m=a;}
    /* All nonzero packed vectors have peaks in [16384,32767]. */
    if(m){if(!peak->m||dst->exp>peak->exp||(dst->exp==peak->exp&&m>peak->m)){
            peak->m=(uint16_t)m;peak->exp=dst->exp;}
        if(dst->exp<min_exp)min_exp=dst->exp;
        if(dst->exp>max_exp)max_exp=dst->exp;}
}
/* Pack wide integer values into a 16-bit vector; choose the finest common
 * binary exponent for which rounding does not exceed +/-32767. */
static void pack(Vec *dst,const Wide *src,int n,int exp) {
    Wide m,a,v;int i,shift=0;int32_t val;
    wu(&m,0);
    for(i=0;i<n;i++){wmag(&a,&src[i]);if(wc(&a,&m)>0)m=a;}
    if(profiling)packs++;
    if(wz(&m)){memset(dst->v,0,(size_t)n*sizeof(dst->v[0]));dst->exp=0;if(profiling)zeros++;return;}
    while(m.hi||m.lo>32767UL){unsigned odd=(unsigned)(m.lo&1UL);wr(&m,&m,1);if(odd)winc(&m);shift++;}
    while(m.lo<=16383UL){m.lo*=2;shift--;}
    dst->exp=exp+shift;
    for(i=0;i<n;i++){
        if(shift>=0)rshift(&v,&src[i],shift);else wl(&v,&src[i],-shift);
        val=wtoi(&v);
        if(val>32767L||val< -32767L){if(profiling)saturation++;die("fixed-point packing overflow");}
        if(profiling&&!wz(&src[i])&&!val)rounded_zero++;
        dst->v[i]=(int16_t)val;
    }
}
static int peak16(const int16_t *v,int n) {int i;int32_t m=0,a;for(i=0;i<n;i++){a=v[i];if(a<0)a=-a;if(a>m)m=a;}if(m>32767L)die("invalid packed vector");return (int)m;}
static int q8(const int16_t *v,int8_t *out,int n) {
    int i,m=peak16(v,n);int32_t a;
    if(!m){memset(out,0,(size_t)n);return 0;}
    for(i=0;i<n;i++){a=(int32_t)v[i]*127L;out[i]=(int8_t)(a<0?-((-a+m/2)/m):(a+m/2)/m);}
    return m;
}
/* Assembly interface uses globals to avoid depending on cc65 fastcall ABI.
 * TINY_ASM_MAC is for a little-endian 65C02 target only. The portable
 * C fallback remains the Stage 8A arithmetic and is used by default. */
#ifdef TINY_ASM_MAC
const int8_t *tiny_mac_a, *tiny_mac_b;
uint16_t tiny_mac_n;
int32_t tiny_mac_result;
int8_t tiny_mac_coeff;
int32_t *tiny_mac_acc;
extern void tiny_mac65_dot(void);
extern void tiny_mac65_axpy(void);
static int32_t dot8_chunk(const int8_t *a,const int8_t *b,int n) {
    tiny_mac_a=a; tiny_mac_b=b; tiny_mac_n=(uint16_t)n;
    tiny_mac65_dot(); return tiny_mac_result;
}
static int32_t dot8(const int8_t *a,const int8_t *b,int n) {
    int32_t sum=0;
    /* 512 arbitrary signed byte pairs can exceed signed 24-bit by one.
     * Chunking keeps the assembly primitive exact even for that case. */
    while(n>511) {sum+=dot8_chunk(a,b,256);a+=256;b+=256;n-=256;}
    return sum+dot8_chunk(a,b,n);
}
static void axpy8(int32_t *accum,int8_t c,const int8_t *v,int n) {
    tiny_mac_acc=accum;tiny_mac_coeff=c;tiny_mac_b=v;
    tiny_mac_n=(uint16_t)n;tiny_mac65_axpy();
}
#else
/* Exact signed byte multiplication using quarter squares.
 * Q[t+256] = floor(t*t/4), -256 <= t <= 255.
 * The sum and difference of two integers have equal parity, so the
 * discarded fractional quarters cancel: a*b = Q[a+b+256]-Q[a-b+256].
 * Separate low/high byte planes for the future 65C02 implementation.
 * No table initialization, bank switches or multiply in this kernel.
 */
#ifndef TINY_NATIVE_MUL
static const uint8_t qs_lo[512] = {
    0x00, 0x80, 0x01, 0x82, 0x04, 0x86, 0x09, 0x8c, 0x10, 0x94, 0x19, 0x9e, 0x24, 0xaa, 0x31, 0xb8,
    0x40, 0xc8, 0x51, 0xda, 0x64, 0xee, 0x79, 0x04, 0x90, 0x1c, 0xa9, 0x36, 0xc4, 0x52, 0xe1, 0x70,
    0x00, 0x90, 0x21, 0xb2, 0x44, 0xd6, 0x69, 0xfc, 0x90, 0x24, 0xb9, 0x4e, 0xe4, 0x7a, 0x11, 0xa8,
    0x40, 0xd8, 0x71, 0x0a, 0xa4, 0x3e, 0xd9, 0x74, 0x10, 0xac, 0x49, 0xe6, 0x84, 0x22, 0xc1, 0x60,
    0x00, 0xa0, 0x41, 0xe2, 0x84, 0x26, 0xc9, 0x6c, 0x10, 0xb4, 0x59, 0xfe, 0xa4, 0x4a, 0xf1, 0x98,
    0x40, 0xe8, 0x91, 0x3a, 0xe4, 0x8e, 0x39, 0xe4, 0x90, 0x3c, 0xe9, 0x96, 0x44, 0xf2, 0xa1, 0x50,
    0x00, 0xb0, 0x61, 0x12, 0xc4, 0x76, 0x29, 0xdc, 0x90, 0x44, 0xf9, 0xae, 0x64, 0x1a, 0xd1, 0x88,
    0x40, 0xf8, 0xb1, 0x6a, 0x24, 0xde, 0x99, 0x54, 0x10, 0xcc, 0x89, 0x46, 0x04, 0xc2, 0x81, 0x40,
    0x00, 0xc0, 0x81, 0x42, 0x04, 0xc6, 0x89, 0x4c, 0x10, 0xd4, 0x99, 0x5e, 0x24, 0xea, 0xb1, 0x78,
    0x40, 0x08, 0xd1, 0x9a, 0x64, 0x2e, 0xf9, 0xc4, 0x90, 0x5c, 0x29, 0xf6, 0xc4, 0x92, 0x61, 0x30,
    0x00, 0xd0, 0xa1, 0x72, 0x44, 0x16, 0xe9, 0xbc, 0x90, 0x64, 0x39, 0x0e, 0xe4, 0xba, 0x91, 0x68,
    0x40, 0x18, 0xf1, 0xca, 0xa4, 0x7e, 0x59, 0x34, 0x10, 0xec, 0xc9, 0xa6, 0x84, 0x62, 0x41, 0x20,
    0x00, 0xe0, 0xc1, 0xa2, 0x84, 0x66, 0x49, 0x2c, 0x10, 0xf4, 0xd9, 0xbe, 0xa4, 0x8a, 0x71, 0x58,
    0x40, 0x28, 0x11, 0xfa, 0xe4, 0xce, 0xb9, 0xa4, 0x90, 0x7c, 0x69, 0x56, 0x44, 0x32, 0x21, 0x10,
    0x00, 0xf0, 0xe1, 0xd2, 0xc4, 0xb6, 0xa9, 0x9c, 0x90, 0x84, 0x79, 0x6e, 0x64, 0x5a, 0x51, 0x48,
    0x40, 0x38, 0x31, 0x2a, 0x24, 0x1e, 0x19, 0x14, 0x10, 0x0c, 0x09, 0x06, 0x04, 0x02, 0x01, 0x00,
    0x00, 0x00, 0x01, 0x02, 0x04, 0x06, 0x09, 0x0c, 0x10, 0x14, 0x19, 0x1e, 0x24, 0x2a, 0x31, 0x38,
    0x40, 0x48, 0x51, 0x5a, 0x64, 0x6e, 0x79, 0x84, 0x90, 0x9c, 0xa9, 0xb6, 0xc4, 0xd2, 0xe1, 0xf0,
    0x00, 0x10, 0x21, 0x32, 0x44, 0x56, 0x69, 0x7c, 0x90, 0xa4, 0xb9, 0xce, 0xe4, 0xfa, 0x11, 0x28,
    0x40, 0x58, 0x71, 0x8a, 0xa4, 0xbe, 0xd9, 0xf4, 0x10, 0x2c, 0x49, 0x66, 0x84, 0xa2, 0xc1, 0xe0,
    0x00, 0x20, 0x41, 0x62, 0x84, 0xa6, 0xc9, 0xec, 0x10, 0x34, 0x59, 0x7e, 0xa4, 0xca, 0xf1, 0x18,
    0x40, 0x68, 0x91, 0xba, 0xe4, 0x0e, 0x39, 0x64, 0x90, 0xbc, 0xe9, 0x16, 0x44, 0x72, 0xa1, 0xd0,
    0x00, 0x30, 0x61, 0x92, 0xc4, 0xf6, 0x29, 0x5c, 0x90, 0xc4, 0xf9, 0x2e, 0x64, 0x9a, 0xd1, 0x08,
    0x40, 0x78, 0xb1, 0xea, 0x24, 0x5e, 0x99, 0xd4, 0x10, 0x4c, 0x89, 0xc6, 0x04, 0x42, 0x81, 0xc0,
    0x00, 0x40, 0x81, 0xc2, 0x04, 0x46, 0x89, 0xcc, 0x10, 0x54, 0x99, 0xde, 0x24, 0x6a, 0xb1, 0xf8,
    0x40, 0x88, 0xd1, 0x1a, 0x64, 0xae, 0xf9, 0x44, 0x90, 0xdc, 0x29, 0x76, 0xc4, 0x12, 0x61, 0xb0,
    0x00, 0x50, 0xa1, 0xf2, 0x44, 0x96, 0xe9, 0x3c, 0x90, 0xe4, 0x39, 0x8e, 0xe4, 0x3a, 0x91, 0xe8,
    0x40, 0x98, 0xf1, 0x4a, 0xa4, 0xfe, 0x59, 0xb4, 0x10, 0x6c, 0xc9, 0x26, 0x84, 0xe2, 0x41, 0xa0,
    0x00, 0x60, 0xc1, 0x22, 0x84, 0xe6, 0x49, 0xac, 0x10, 0x74, 0xd9, 0x3e, 0xa4, 0x0a, 0x71, 0xd8,
    0x40, 0xa8, 0x11, 0x7a, 0xe4, 0x4e, 0xb9, 0x24, 0x90, 0xfc, 0x69, 0xd6, 0x44, 0xb2, 0x21, 0x90,
    0x00, 0x70, 0xe1, 0x52, 0xc4, 0x36, 0xa9, 0x1c, 0x90, 0x04, 0x79, 0xee, 0x64, 0xda, 0x51, 0xc8,
    0x40, 0xb8, 0x31, 0xaa, 0x24, 0x9e, 0x19, 0x94, 0x10, 0x8c, 0x09, 0x86, 0x04, 0x82, 0x01, 0x80,
};
static const uint8_t qs_hi[512] = {
    0x40, 0x3f, 0x3f, 0x3e, 0x3e, 0x3d, 0x3d, 0x3c, 0x3c, 0x3b, 0x3b, 0x3a, 0x3a, 0x39, 0x39, 0x38,
    0x38, 0x37, 0x37, 0x36, 0x36, 0x35, 0x35, 0x35, 0x34, 0x34, 0x33, 0x33, 0x32, 0x32, 0x31, 0x31,
    0x31, 0x30, 0x30, 0x2f, 0x2f, 0x2e, 0x2e, 0x2d, 0x2d, 0x2d, 0x2c, 0x2c, 0x2b, 0x2b, 0x2b, 0x2a,
    0x2a, 0x29, 0x29, 0x29, 0x28, 0x28, 0x27, 0x27, 0x27, 0x26, 0x26, 0x25, 0x25, 0x25, 0x24, 0x24,
    0x24, 0x23, 0x23, 0x22, 0x22, 0x22, 0x21, 0x21, 0x21, 0x20, 0x20, 0x1f, 0x1f, 0x1f, 0x1e, 0x1e,
    0x1e, 0x1d, 0x1d, 0x1d, 0x1c, 0x1c, 0x1c, 0x1b, 0x1b, 0x1b, 0x1a, 0x1a, 0x1a, 0x19, 0x19, 0x19,
    0x19, 0x18, 0x18, 0x18, 0x17, 0x17, 0x17, 0x16, 0x16, 0x16, 0x15, 0x15, 0x15, 0x15, 0x14, 0x14,
    0x14, 0x13, 0x13, 0x13, 0x13, 0x12, 0x12, 0x12, 0x12, 0x11, 0x11, 0x11, 0x11, 0x10, 0x10, 0x10,
    0x10, 0x0f, 0x0f, 0x0f, 0x0f, 0x0e, 0x0e, 0x0e, 0x0e, 0x0d, 0x0d, 0x0d, 0x0d, 0x0c, 0x0c, 0x0c,
    0x0c, 0x0c, 0x0b, 0x0b, 0x0b, 0x0b, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x09, 0x08, 0x08, 0x08, 0x08, 0x08, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0b, 0x0b, 0x0b, 0x0b, 0x0c,
    0x0c, 0x0c, 0x0c, 0x0c, 0x0d, 0x0d, 0x0d, 0x0d, 0x0e, 0x0e, 0x0e, 0x0e, 0x0f, 0x0f, 0x0f, 0x0f,
    0x10, 0x10, 0x10, 0x10, 0x11, 0x11, 0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13,
    0x14, 0x14, 0x14, 0x15, 0x15, 0x15, 0x15, 0x16, 0x16, 0x16, 0x17, 0x17, 0x17, 0x18, 0x18, 0x18,
    0x19, 0x19, 0x19, 0x19, 0x1a, 0x1a, 0x1a, 0x1b, 0x1b, 0x1b, 0x1c, 0x1c, 0x1c, 0x1d, 0x1d, 0x1d,
    0x1e, 0x1e, 0x1e, 0x1f, 0x1f, 0x1f, 0x20, 0x20, 0x21, 0x21, 0x21, 0x22, 0x22, 0x22, 0x23, 0x23,
    0x24, 0x24, 0x24, 0x25, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27, 0x27, 0x28, 0x28, 0x29, 0x29, 0x29,
    0x2a, 0x2a, 0x2b, 0x2b, 0x2b, 0x2c, 0x2c, 0x2d, 0x2d, 0x2d, 0x2e, 0x2e, 0x2f, 0x2f, 0x30, 0x30,
    0x31, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
    0x38, 0x38, 0x39, 0x39, 0x3a, 0x3a, 0x3b, 0x3b, 0x3c, 0x3c, 0x3d, 0x3d, 0x3e, 0x3e, 0x3f, 0x3f,
};
static int16_t quarter_square(unsigned index) {
    /* Every entry is <= 16384, so this conversion is representable. */
    return (int16_t)((uint16_t)qs_lo[index] |
                     ((uint16_t)qs_hi[index] << 8));
}
#endif
static int16_t mul8(int8_t a,int8_t b) {
#ifdef TINY_NATIVE_MUL
    return (int16_t)a * (int16_t)b;
#else
    unsigned u=(unsigned)((int)a+128), v=(unsigned)((int)b+128);
    return quarter_square(u+v)-quarter_square(u+256u-v);
#endif
}
static int32_t dot8(const int8_t *a,const int8_t *b,int n) {
    int i;int32_t sum=0;
    for(i=0;i<n;i++)sum+=(int32_t)mul8(a[i],b[i]);
    return sum;
}
static void axpy8(int32_t *accum,int8_t c,const int8_t *v,int n) {
    int i;for(i=0;i<n;i++)accum[i]+=(int32_t)mul8(c,v[i]);
}
#endif
static void matvec(const Table *t,const Vec *src,Vec *dst) {
    static int8_t q[MAXHID];int r,m=q8(src->v,q,t->cols);int e=src->exp+t->exp;Wide a;
    for(r=0;r<t->rows;r++){
        rw_read(&t->data,r,0,rowbuf,(unsigned)t->cols+4u);
        wi(&a,dot8((const int8_t *)(rowbuf+4),q,t->cols));wmi(&a,&a,signed32(get32(rowbuf)));wmu(&a,&a,(uint32_t)m);rdiv(&wide[r],&a,127);
    }
    pack(dst,wide,t->rows,e);
}
static void row(const Table *t,int r,Vec *dst) {
    int i;int32_t c;rw_read(&t->data,r,0,rowbuf,(unsigned)t->cols+4u);c=signed32(get32(rowbuf));
    for(i=0;i<t->cols;i++){wi(&wide[i],(int8_t)rowbuf[4+i]);wmi(&wide[i],&wide[i],c);}
    pack(dst,wide,t->cols,t->exp);
}
static void align(Wide *r,const Wide *v,int from,int to) {
    if(from>=to)wl(r,v,from-to);else rshift(r,v,to-from);
}
static void residual(Vec *a,const Vec *b,int n) {
    int i,e;Wide av,bv;
    if(!peak16(b->v,n))return;
    if(!peak16(a->v,n)){*a=*b;return;}
    e=(a->exp>b->exp?a->exp:b->exp)-16;
    for(i=0;i<n;i++){
        wi(&av,a->v[i]);wi(&bv,b->v[i]);align(&av,&av,a->exp,e);align(&bv,&bv,b->exp,e);wa(&wide[i],&av,&bv);
    }
    pack(a,wide,n,e);
}
static uint32_t isqrt_round(const Wide *src) {
    Wide n,bit,root,t;n=*src;wu(&bit,1);wl(&bit,&bit,48);wu(&root,0);
    while(wc(&bit,&n)>0)wr(&bit,&bit,2);
    while(!wz(&bit)){
        wa(&t,&root,&bit);
        if(wc(&n,&t)>=0){ws(&n,&n,&t);wr(&root,&root,1);wa(&root,&root,&bit);}
        else wr(&root,&root,1);
        wr(&bit,&bit,2);
    }
    return root.lo+(wc(&n,&root)>0);
}
static void normalize(const Vec *src,const Gain *g,Vec *dst) {
    int i,xe,hi,eh,e;Wide ss,mean,rad,t,eps,b,inv;uint32_t root;
    wu(&ss,0);eps.hi=2;eps.lo=U32(2669064476); /* 11258999068 */
    for(i=0;i<D;i++){int32_t v=src->v[i];wu(&t,(uint32_t)(v*v));wa(&ss,&ss,&t);}
    if(wz(&ss)){memset(wide,0,(size_t)D*sizeof(wide[0]));pack(dst,wide,D,0);return;}
    wl(&mean,&ss,16);wu(&b,(uint32_t)D);wrounddiv(&mean,&mean,&b);
    xe=2*src->exp-16;hi=topbit(&mean)+xe;eh=topbit(&eps)-50;
    e=(hi>eh?hi:eh)-46;if(e%2)e--;
    align(&rad,&mean,xe,e);align(&t,&eps,-50,e);wa(&rad,&rad,&t);
    root=isqrt_round(&rad);wu(&t,1);wl(&t,&t,47);wu(&b,root);wrounddiv(&inv,&t,&b);
    for(i=0;i<D;i++){wi(&t,src->v[i]);wmi(&t,&t,g->v[i]);wm(&wide[i],&t,&inv);}
    pack(dst,wide,D,src->exp+g->exp-47-e/2);
}
static void cache_store(const Region *r,int p,const Vec *src) {
    unsigned m=(unsigned)q8(src->v,(int8_t *)(rowbuf+4),D);
#if INT_MAX > 32767
    if(src->exp < -32767-1 || src->exp > 32767)die("cache exponent out of range");
#endif
    put16(rowbuf,m);put16(rowbuf+2,(uint16_t)src->exp);
    rw_write(r,p,0,rowbuf,(unsigned)D+4u);
}
static void cache_scale(const Region *r,int p,CacheScale *s) {
    rw_read(r,p,0,rowbuf,4);s->m=(uint16_t)get16(rowbuf);s->exp=signed16(get16(rowbuf+2));
}
static void cache_head(const Region *r,int p,int h) {
    rw_read(r,p,4u+(unsigned)h*HD,rowbuf,(unsigned)HD);
}
/* exp(-d), d >= 0, with a Q16 input and unsigned Q24 output.
 * Table spacing 1/64 (1024 Q16 units); linear interpolation uses 10 bits.
 * The largest adjacent difference times 1023 fits in uint32_t.
 * Beyond d=18 the true Q24 exponential rounds to zero. The table is
 * generated offline, not at program startup, and occupies 3456 bytes (24-bit entries, with entry zero implicit). */
#define EXP_ONE U32(16777216)
#define EXP_LIMIT (U32(18)<<16)
static const uint8_t exp_lut_0[1152] = {
    245, 171, 227, 96, 228, 53, 23, 81, 170, 233, 216, 65, 237, 168, 62, 125,
    50, 43, 56, 41, 207, 252, 128, 49, 224, 99, 143, 57, 55, 97, 143, 152,
    85, 160, 84, 74, 93, 107, 78, 229, 12, 161, 131, 146, 172, 178, 132, 4,
    18, 145, 100, 108, 143, 175, 176, 120, 236, 240, 107, 68, 97, 169, 4, 89,
    145, 149, 77, 163, 128, 207, 123, 109, 144, 209, 27, 90, 122, 104, 16, 97,
    71, 177, 140, 198, 80, 22, 9, 25, 51, 74, 76, 42, 213, 62, 86, 15,
    89, 39, 107, 24, 30, 114, 6, 206, 188, 196, 217, 241, 254, 245, 203, 115,
    227, 16, 239, 117, 152, 77, 138, 69, 117, 15, 9, 91, 251, 224, 1, 85,
    212, 116, 45, 248, 203, 159, 107, 41, 207, 88, 186, 239, 240, 182, 56, 114,
    91, 237, 33, 241, 87, 77, 203, 204, 74, 64, 167, 121, 177, 73, 61, 134,
    31, 3, 45, 152, 63, 28, 44, 105, 207, 89, 3, 200, 163, 145, 141, 148,
    159, 173, 185, 190, 186, 167, 131, 74, 248, 137, 250, 72, 111, 107, 58, 216,
    66, 117, 110, 41, 163, 219, 204, 115, 207, 220, 151, 254, 15, 197, 32, 27,
    182, 237, 190, 38, 36, 180, 213, 133, 192, 134, 212, 167, 254, 214, 46, 4,
    85, 33, 100, 29, 74, 234, 250, 122, 102, 190, 128, 169, 58, 47, 136, 66,
    93, 215, 174, 225, 110, 85, 147, 40, 18, 79, 223, 192, 240, 112, 60, 85,
    185, 103, 94, 156, 33, 235, 249, 74, 222, 179, 200, 29, 175, 127, 139, 210,
    84, 15, 3, 47, 145, 42, 248, 250, 48, 152, 51, 254, 251, 38, 129, 10,
    193, 164, 179, 238, 84, 227, 156, 126, 136, 186, 18, 145, 53, 255, 237, 255,
    53, 142, 8, 165, 99, 66, 65, 96, 158, 251, 118, 15, 197, 152, 136, 148,
    187, 254, 91, 211, 101, 16, 212, 177, 166, 180, 217, 21, 104, 209, 81, 231,
    146, 82, 39, 17, 15, 32, 69, 126, 201, 40, 152, 27, 175, 85, 12, 213,
    174, 151, 145, 155, 180, 221, 21, 92, 178, 22, 137, 10, 153, 53, 223, 150,
    90, 43, 9, 243, 233, 236, 250, 20, 57, 106, 166, 237, 62, 155, 1, 115,
    238, 115, 2, 155, 61, 233, 158, 92, 35, 243, 203, 172, 150, 136, 130, 132,
    142, 159, 185, 217, 2, 49, 104, 166, 235, 55, 137, 226, 66, 168, 21, 136,
    0, 127, 4, 143, 32, 182, 82, 244, 155, 71, 248, 175, 107, 44, 241, 188,
    140, 96, 57, 22, 248, 222, 201, 184, 171, 162, 158, 158, 161, 168, 180, 195,
    214, 236, 6, 36, 69, 106, 146, 189, 236, 30, 83, 139, 199, 5, 70, 139,
    210, 28, 105, 185, 11, 97, 184, 19, 112, 207, 49, 150, 253, 102, 209, 63,
    175, 34, 150, 13, 134, 1, 126, 253, 126, 1, 134, 13, 149, 32, 172, 59,
    203, 92, 240, 133, 28, 180, 78, 234, 135, 37, 197, 103, 10, 175, 85, 252,
    165, 79, 250, 167, 85, 4, 181, 103, 26, 206, 131, 58, 242, 170, 100, 31,
    219, 152, 86, 22, 214, 151, 89, 28, 224, 165, 107, 50, 249, 194, 139, 86,
    33, 237, 185, 135, 85, 36, 244, 197, 150, 104, 59, 14, 226, 183, 140, 99,
    57, 17, 233, 193, 155, 117, 79, 42, 6, 226, 191, 156, 122, 88, 55, 22,
    246, 215, 184, 153, 123, 93, 64, 35, 7, 235, 208, 180, 154, 128, 102, 76,
    51, 27, 3, 235, 211, 188, 165, 143, 121, 99, 78, 57, 36, 16, 252, 232,
    212, 193, 174, 156, 137, 119, 102, 84, 67, 50, 33, 17, 1, 241, 225, 210,
    195, 180, 165, 151, 137, 122, 109, 95, 82, 69, 56, 43, 30, 18, 6, 250,
    238, 226, 215, 204, 192, 182, 171, 160, 150, 140, 129, 119, 110, 100, 91, 81,
    72, 63, 54, 45, 37, 28, 20, 11, 3, 251, 244, 236, 228, 221, 213, 206,
    199, 192, 185, 178, 171, 165, 158, 152, 145, 139, 133, 127, 121, 115, 109, 104,
    98, 93, 87, 82, 77, 72, 67, 62, 57, 52, 47, 42, 38, 33, 29, 24,
    20, 16, 11, 7, 3, 255, 251, 247, 243, 240, 236, 232, 229, 225, 222, 218,
    215, 212, 208, 205, 202, 199, 196, 193, 190, 187, 184, 181, 178, 175, 173, 170,
    167, 165, 162, 160, 157, 155, 152, 150, 148, 145, 143, 141, 139, 137, 134, 132,
    130, 128, 126, 124, 122, 121, 119, 117, 115, 113, 111, 110, 108, 106, 105, 103,
    101, 100, 98, 97, 95, 94, 92, 91, 90, 88, 87, 85, 84, 83, 82, 80,
    79, 78, 77, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63,
    62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 53, 52, 51, 50, 49, 49,
    48, 47, 46, 46, 45, 44, 44, 43, 42, 42, 41, 40, 40, 39, 39, 38,
    37, 37, 36, 36, 35, 35, 34, 33, 33, 32, 32, 31, 31, 30, 30, 30,
    29, 29, 28, 28, 27, 27, 26, 26, 26, 25, 25, 24, 24, 24, 23, 23,
    23, 22, 22, 22, 21, 21, 21, 20, 20, 20, 19, 19, 19, 18, 18, 18,
    18, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14,
    14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 11, 11, 11, 11,
    11, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 8,
    8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static const uint8_t exp_lut_1[1152] = {
    7, 31, 70, 125, 194, 23, 122, 235, 106, 247, 146, 59, 240, 179, 131, 95,
    72, 61, 62, 75, 99, 135, 183, 242, 55, 136, 227, 73, 185, 51, 183, 69,
    221, 126, 41, 221, 154, 96, 47, 6, 231, 207, 192, 185, 186, 195, 212, 237,
    13, 52, 99, 153, 214, 26, 101, 183, 15, 110, 212, 64, 178, 42, 169, 45,
    183, 71, 221, 120, 25, 191, 107, 28, 210, 141, 78, 19, 221, 172, 128, 88,
    53, 22, 252, 230, 213, 200, 191, 186, 185, 188, 195, 206, 220, 239, 5, 31,
    60, 93, 129, 169, 212, 2, 52, 104, 160, 219, 25, 90, 158, 229, 47, 124,
    203, 30, 114, 202, 36, 129, 224, 66, 166, 13, 118, 225, 78, 190, 49, 165,
    27, 148, 15, 139, 10, 139, 14, 147, 25, 162, 44, 184, 70, 214, 104, 251,
    144, 38, 191, 88, 244, 145, 47, 207, 113, 20, 184, 94, 5, 174, 88, 3,
    176, 94, 13, 189, 111, 34, 214, 139, 65, 249, 178, 107, 38, 226, 159, 93,
    28, 220, 157, 95, 34, 230, 171, 113, 55, 255, 199, 145, 91, 38, 242, 190,
    140, 90, 41, 249, 201, 154, 108, 63, 18, 230, 187, 144, 103, 61, 21, 237,
    197, 158, 120, 83, 46, 9, 229, 194, 159, 125, 91, 58, 25, 249, 218, 187,
    156, 126, 96, 67, 38, 9, 237, 210, 183, 156, 130, 104, 79, 54, 29, 5,
    237, 213, 190, 167, 145, 123, 101, 80, 59, 38, 17, 253, 233, 214, 195, 176,
    157, 139, 121, 103, 86, 68, 51, 35, 18, 2, 242, 227, 211, 196, 181, 166,
    152, 138, 124, 110, 96, 83, 69, 56, 44, 31, 19, 6, 250, 239, 227, 216,
    204, 193, 182, 171, 161, 150, 140, 130, 120, 110, 101, 91, 82, 72, 63, 54,
    46, 37, 29, 20, 12, 4, 252, 244, 236, 228, 221, 214, 206, 199, 192, 185,
    178, 171, 165, 158, 152, 146, 139, 133, 127, 121, 115, 110, 104, 98, 93, 87,
    82, 77, 72, 67, 62, 57, 52, 47, 42, 38, 33, 29, 24, 20, 16, 11,
    7, 3, 255, 251, 247, 243, 240, 236, 232, 229, 225, 222, 218, 215, 211, 208,
    205, 202, 199, 195, 192, 189, 186, 184, 181, 178, 175, 172, 170, 167, 165, 162,
    159, 157, 155, 152, 150, 147, 145, 143, 141, 138, 136, 134, 132, 130, 128, 126,
    124, 122, 120, 118, 117, 115, 113, 111, 109, 108, 106, 104, 103, 101, 100, 98,
    97, 95, 94, 92, 91, 89, 88, 86, 85, 84, 82, 81, 80, 79, 77, 76,
    75, 74, 73, 72, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59,
    58, 57, 57, 56, 55, 54, 53, 52, 51, 51, 50, 49, 48, 48, 47, 46,
    45, 45, 44, 43, 43, 42, 41, 41, 40, 39, 39, 38, 37, 37, 36, 36,
    35, 35, 34, 34, 33, 33, 32, 31, 31, 31, 30, 30, 29, 29, 28, 28,
    27, 27, 26, 26, 26, 25, 25, 24, 24, 24, 23, 23, 23, 22, 22, 21,
    21, 21, 20, 20, 20, 20, 19, 19, 19, 18, 18, 18, 17, 17, 17, 17,
    16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13,
    13, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10,
    10, 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static const uint8_t exp_lut_2[1152] = {
    252, 248, 244, 240, 236, 233, 229, 225, 222, 218, 215, 212, 208, 205, 202, 199,
    196, 193, 190, 187, 184, 181, 178, 175, 173, 170, 167, 165, 162, 160, 157, 155,
    152, 150, 148, 145, 143, 141, 139, 137, 134, 132, 130, 128, 126, 124, 122, 120,
    119, 117, 115, 113, 111, 110, 108, 106, 105, 103, 101, 100, 98, 97, 95, 94,
    92, 91, 89, 88, 87, 85, 84, 83, 81, 80, 79, 78, 76, 75, 74, 73,
    72, 71, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 58, 57,
    56, 55, 54, 53, 52, 52, 51, 50, 49, 48, 48, 47, 46, 45, 45, 44,
    43, 43, 42, 41, 41, 40, 39, 39, 38, 38, 37, 36, 36, 35, 35, 34,
    34, 33, 33, 32, 32, 31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26,
    26, 26, 25, 25, 24, 24, 24, 23, 23, 23, 22, 22, 22, 21, 21, 21,
    20, 20, 20, 19, 19, 19, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16,
    16, 15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 13, 12, 12,
    12, 12, 12, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 9,
    9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static uint32_t exp_entry(unsigned i) {
    if(!i)return EXP_ONE;
    --i;
    return (uint32_t)exp_lut_0[i] | ((uint32_t)exp_lut_1[i]<<8) | ((uint32_t)exp_lut_2[i]<<16);
}
static uint32_t exp_neg_q16(uint32_t d) {
    uint32_t i,f,a,b;
    if(d>=EXP_LIMIT)return 0;
    i=d>>10; f=d&1023u;
    a=exp_entry((unsigned)i);b=exp_entry((unsigned)(i+1));
    return a-(((a-b)*f+512u)>>10);
}
static void update_hi(const Wide *v,int e,int *hi) {
    Wide m;int b;if(wz(v))return;wmag(&m,v);b=topbit(&m)+e;if(b>*hi)*hi=b;
}
/* Nonnegative difference -> Q16, clipped to the exponential table. */
static uint32_t delta_q16(const Wide *m,int e,unsigned mul,unsigned div) {
    Wide a,b,v,limit;int shift=e+16;
    if(wz(m))return 0;
    wmu(&a,m,mul);wu(&limit,EXP_LIMIT*(uint32_t)div);
    if(shift>=0){
        wr(&v,&limit,shift);if(shift>=63||wc(&a,&v)>0)return EXP_LIMIT;
        wl(&v,&a,shift);wu(&b,div);
    }else{
        if(shift<=-60)return 0;
        v=a;wu(&b,div);wl(&b,&b,-shift);
    }
    wrounddiv(&v,&v,&b);
    return v.hi||v.lo>=EXP_LIMIT?EXP_LIMIT:v.lo;
}
static void attention(Layer *a,int p,const Vec *q,Vec *dst) {
    int h,j,i,m,hi,e,he,shift;
    static int exps[CTX],out_exp[MAXD];
    static int8_t aq[MAXD],av[CTX];static int32_t accum[MAXD];
    static Wide scores[CTX],result[MAXD];Wide raw,big,s,t,den,scale;
    uint32_t sum,weight;int out_hi=INT_MIN;CacheScale cs;
    for(h=0;h<H;h++){
        m=q8(q->v+h*HD,aq,HD);hi=INT_MIN;
        for(j=0;j<=p;j++){
            cache_scale(&a->key,j,&cs);cache_head(&a->key,j,h);
            wi(&raw,dot8(aq,(const int8_t *)rowbuf,HD));wmu(&raw,&raw,(uint32_t)m);wmu(&raw,&raw,cs.m);
            wmag(&t,&raw);shift=wz(&t)?0:topbit(&t)-29;if(shift<0)shift=0;
            rshift(&t,&raw,shift);wmu(&scores[j],&t,att_factor);
            exps[j]=q->exp+cs.exp+shift-44;update_hi(&scores[j],exps[j],&hi);
        }
        e=hi==INT_MIN?0:hi-45;
        for(j=0;j<=p;j++){align(&scores[j],&scores[j],exps[j],e);if(j==0||wcs(&scores[j],&big)>0)big=scores[j];}
        sum=0;hi=INT_MIN;
        for(j=0;j<=p;j++){
            ws(&t,&big,&scores[j]);weight=exp_neg_q16(delta_q16(&t,e,1,1));sum+=weight;
            cache_scale(&a->value,j,&cs);
            wu(&t,weight);wmu(&scores[j],&t,cs.m);exps[j]=cs.exp;update_hi(&scores[j],exps[j],&hi);
        }
        e=hi==INT_MIN?0:hi-40;wu(&big,0);
        for(j=0;j<=p;j++){align(&scores[j],&scores[j],exps[j],e);if(wc(&scores[j],&big)>0)big=scores[j];}
        for(j=0;j<=p;j++){
            if(wz(&big))av[j]=0;else{wmu(&t,&scores[j],127);wrounddiv(&t,&t,&big);av[j]=(int8_t)t.lo;}
        }
        for(i=0;i<HD;i++)accum[i]=0;
        for(j=0;j<=p;j++)if(av[j]){cache_head(&a->value,j,h);axpy8(accum,av[j],(const int8_t *)rowbuf,HD);}
        wu(&den,sum);wmu(&den,&den,16129);wl(&t,&big,22);wrounddiv(&scale,&t,&den);he=e-22;
        for(i=0;i<HD;i++){
            wi(&s,accum[i]);wm(&s,&s,&scale);result[h*HD+i]=s;out_exp[h*HD+i]=he;update_hi(&s,he,&out_hi);
        }
    }
    e=out_hi==INT_MIN?0:out_hi-45;
    for(i=0;i<D;i++)align(&wide[i],&result[i],out_exp[i],e);
    pack(dst,wide,D,e);
}
static void step(int tok,int p,Vec *out) {
    int l,i;Layer *a;
    row(&emb,tok,&x);row(&pos,p,&tmp);residual(&x,&tmp,D);note(&x,D,&peak_residual);
    for(l=0;l<L;l++){
        a=&layers[l];fetch_gain(&a->g1);normalize(&x,&gain_work,&normed);note(&normed,D,&peak_norm);
        matvec(&a->k,&normed,&tmp);note(&tmp,D,&peak_proj);cache_store(&a->key,p,&tmp);
        matvec(&a->v,&normed,&tmp);note(&tmp,D,&peak_proj);cache_store(&a->value,p,&tmp);
        matvec(&a->q,&normed,&query);note(&query,D,&peak_proj);
        attention(a,p,&query,&att);note(&att,D,&peak_att);
        matvec(&a->o,&att,&tmp);note(&tmp,D,&peak_proj);residual(&x,&tmp,D);note(&x,D,&peak_residual);
        fetch_gain(&a->g2);normalize(&x,&gain_work,&normed);note(&normed,D,&peak_norm);
        matvec(&a->f1,&normed,&hidden);note(&hidden,HID,&peak_hidden);
        for(i=0;i<HID;i++)if(hidden.v[i]<0)hidden.v[i]=0;
        matvec(&a->f2,&hidden,&tmp);note(&tmp,D,&peak_proj);residual(&x,&tmp,D);note(&x,D,&peak_residual);
    }
    fetch_gain(&gf);normalize(&x,&gain_work,&normed);note(&normed,D,&peak_norm);
    matvec(&emb,&normed,&tmp);note(&tmp,VOCAB,&peak_logit);
    *out=tmp;
}
static int clean(const char *s,char *q){int n=0,u=0,l=0,i;for(;*s&&n<MAXQ;s++)if((unsigned char)*s>=32&&(unsigned char)*s<=126){q[n++]=*s;u+=*s>='A'&&*s<='Z';l+=*s>='a'&&*s<='z';}while(n&&q[n-1]==' ')n--;q[n]=0;if(u&&!l)for(i=0;i<n;i++)if(q[i]>='A'&&q[i]<='Z')q[i]+='a'-'A';return n;}
static int is_word(const char *s,const char *w){while(*s&&*w){if((*s|32)!=*w)return 0;s++;w++;}return !*s&&!*w;}
static int pick(void) {
    int i,best=0;static uint32_t w[VOCAB];uint32_t sum=0,r;Wide t;
    for(i=1;i<VOCAB;i++)if(score.v[i]>score.v[best])best=i;
    if(!creativity)return best;
    for(i=0;i<VOCAB;i++){
        wu(&t,(uint32_t)((int32_t)score.v[best]-score.v[i]));
        w[i]=exp_neg_q16(delta_q16(&t,score.exp,5,(unsigned)creativity));
        if(w[i]<U32(2516583))w[i]=0;
        sum+=w[i];
    }
    wu(&t,sum);wmu(&t,&t,(unsigned)(rand()%10000));wdu(&t,&t,10000);r=t.lo;
    for(i=0;i<VOCAB;i++){if(r<w[i])return i;r-=w[i];}
    return best;
}
/* Exact mantissa/scale checksum: FNV-1a over little-endian int16 logits. */
static void check_question(const char *q) {
    int p,i,t,n=(int)strlen(q);uint32_t hash;uint16_t v;
    puts("position,token,exponent,hash");
    for(p=0;p<=n;p++){
        t=p==n?1:(unsigned char)q[p]-30;step(t,p,&score);
        hash=U32(2166136261);
        for(i=0;i<VOCAB;i++){
            v=(uint16_t)score.v[i];
            hash=(hash^(v&255u))*U32(16777619);
            hash=(hash^(v>>8))*U32(16777619);
        }
        printf("%d,%d,%d,%08lX\n",p,t,score.exp,(unsigned long)hash);
        fflush(stdout);
    }
}
#ifdef TINY_APPLE2
/* ProDOS system programs have no useful argv. Use interactive commands. */
static void apple_answer(const char *q) {
    int p,t,n=(int)strlen(q),letters=0;
    puts("Thinking...");
    for(p=0;p<=n;p++){
        t=p==n?1:(unsigned char)q[p]-30;step(t,p,&score);
        putchar('.');fflush(stdout);
    }
    putchar('\n');
    for(;p<CTX;p++){
        t=pick();if(t<=1)break;putchar(t+30);fflush(stdout);letters++;
        step(t,p,&score);
    }
    printf("\n[%d characters]\n",letters);
}
int main(void) {
    static char line[96],q[MAXQ+1];
    unsigned bad;int check=0,c;
    puts("TINY 8B / Apple //e / RamWorks");
    puts("Banks 1-12 reserved for TINY.");
    bad=tiny_rw_probe();if(bad){printf("Bank probe failed: %u\n",bad);return 1;}
    srand(1);rw_report=1;load_model("TINY.INT");rw_stats("load");rw_reset_stats();
    puts(":check toggles hash trace; :banks stats; quit exits.");
    for(;;){
        fputs("> ",stdout);fflush(stdout);
        if(!fgets(line,sizeof(line),stdin))break;
        if(!strchr(line,'\n'))while((c=getchar())!='\n'&&c!=EOF){}
        if(!clean(line,q))continue;
        if(is_word(q,"quit")||is_word(q,"exit"))break;
        if(is_word(q,":check")){check=!check;puts(check?"Trace on":"Trace off");continue;}
        if(is_word(q,":banks")){rw_stats("inference");continue;}
        if(q[0]==':'&&q[1]=='c'&&q[2]>='0'&&q[2]<='9'&&!q[3]){creativity=q[2]-'0';continue;}
        if(check)check_question(q);else apple_answer(q);
    }
    return 0;
}
#else
/* Decimal scientific notation from integer mantissa and binary exponent.
 * For negative e, m*2^e = m*5^(-e)*10^e.  Keep decimal digits and emit
 * up to 10 significant digits; no floating types or printf conversions. */
static void print_number(FILE *f,int32_t m,int e) {
    static unsigned char digits[2048];unsigned n=0,i,v,carry,factor;
    uint32_t a;int k,dec_exp;
    if(!m){fputc('0',f);return;}
    if(m<0){fputc('-',f);a=0UL-(uint32_t)m;}else a=(uint32_t)m;
    while(a){digits[n++]=(unsigned char)(a%10);a/=10;}
    if(e < -2048 || e > 2048)die("diagnostic exponent out of range");
    factor=e<0?5u:2u;k=e<0?-e:e;
    while(k--){carry=0;for(i=0;i<n;i++){v=digits[i]*factor+carry;digits[i]=(unsigned char)(v%10);carry=v/10;}
        if(carry){if(n>=sizeof(digits))die("diagnostic number too long");digits[n++]=(unsigned char)carry;}}
    dec_exp=(int)n-1+(e<0?e:0);
    fputc('0'+digits[n-1],f);
    if(n>1){fputc('.',f);for(i=1;i<n&&i<10;i++)fputc('0'+digits[n-1-i],f);}
    fprintf(f,"e%d",dec_exp);
}
static void answer(const char *q,int trace) {
    int p,i,t,n=(int)strlen(q),letters=0;clock_t start=clock(),end;uint32_t ticks,sec,frac;Wide wt;
    if(trace){printf("position,input_token");for(i=0;i<VOCAB;i++)printf(",logit_%d",i);putchar('\n');}
    for(p=0;p<=n;p++){
        t=p==n?1:(unsigned char)q[p]-30;step(t,p,&score);
        if(trace){printf("%d,%d",p,t);for(i=0;i<VOCAB;i++){putchar(',');print_number(stdout,score.v[i],score.exp);}putchar('\n');}
    }
    if(trace)return;
    for(;p<CTX;p++){t=pick();if(t<=1)break;putchar(t+30);fflush(stdout);letters++;step(t,p,&score);}
    putchar('\n');end=clock();
    if(start==(clock_t)-1||end==(clock_t)-1||end<start)fprintf(stderr,"[%d characters]\n",letters);
    else{ticks=(uint32_t)end-(uint32_t)start;sec=ticks/(uint32_t)CLOCKS_PER_SEC;
        wu(&wt,ticks%(uint32_t)CLOCKS_PER_SEC);wmu(&wt,&wt,10);wdu(&wt,&wt,(uint32_t)CLOCKS_PER_SEC);frac=wt.lo;
        fprintf(stderr,"[%d characters, %lu.%lu CPU seconds]\n",letters,(unsigned long)sec,(unsigned long)frac);}
}
static void profile(void) {
    if(!profiling)return;
    fprintf(stderr,"Stage 8B profile (absolute peaks, cumulative):\n residual ");print_number(stderr,peak_residual.m,peak_residual.exp);
    fputs(", norm ",stderr);print_number(stderr,peak_norm.m,peak_norm.exp);
    fputs(", projection ",stderr);print_number(stderr,peak_proj.m,peak_proj.exp);
    fputs(", hidden ",stderr);print_number(stderr,peak_hidden.m,peak_hidden.exp);
    fputs(", attention ",stderr);print_number(stderr,peak_att.m,peak_att.exp);
    fputs(", logits ",stderr);print_number(stderr,peak_logit.m,peak_logit.exp);fputc('\n',stderr);
    if(min_exp!=INT_MAX)fprintf(stderr," vector binary exponent range %d..%d\n",min_exp,max_exp);
    fprintf(stderr," packs=%lu zero_vectors=%lu rounded_to_zero=%lu saturations=%lu\n",(unsigned long)packs,(unsigned long)zeros,(unsigned long)rounded_zero,(unsigned long)saturation);
}
static int uintarg(const char *s,unsigned *v){char *e;unsigned long a;if(*s<'0'||*s>'9')return 0;errno=0;a=strtoul(s,&e,10);if(errno||*e||a>UINT_MAX)return 0;*v=(unsigned)a;return 1;}
static void usage(const char *s){fprintf(stderr,"Usage: %s [-m model] [-q question] [-c 0..9] [-s seed] [--logits|--check] [--profile] [--banks]\n",s);}
int main(int argc,char **argv){int i,trace=0,c;unsigned seed=1,v;const char *model="TINY.INT",*question=NULL;char line[256],q[MAXQ+1];
    if(CHAR_BIT!=8)die("requires 8-bit bytes");
    for(i=1;i<argc;i++){if(!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")){usage(argv[0]);return 0;}else if(!strcmp(argv[i],"--profile")){profiling=1;rw_report=1;}else if(!strcmp(argv[i],"--banks"))rw_report=1;else if(!strcmp(argv[i],"--logits"))trace=1;else if(!strcmp(argv[i],"--check"))trace=2;else if(!strcmp(argv[i],"-m")&&i+1<argc)model=argv[++i];else if(!strcmp(argv[i],"-q")&&i+1<argc)question=argv[++i];else if(!strcmp(argv[i],"-c")&&i+1<argc&&uintarg(argv[++i],&v)&&v<=9)creativity=v;else if(!strcmp(argv[i],"-s")&&i+1<argc&&uintarg(argv[++i],&seed)){}else{usage(argv[0]);return 2;}}
    if(trace&&!question){usage(argv[0]);return 2;}if(question&&!clean(question,q))die("no printable question");
    srand(seed);load_model(model);rw_stats("load");rw_reset_stats();fprintf(stderr,"TINY Stage 8B: %d layers, width %d, %d heads, hidden %d, creativity %d\n",L,D,H,HID,creativity);
    if(question){if(trace==2)check_question(q);else answer(q,trace);}else{while(fputs("> ",stderr),fflush(stderr),fgets(line,sizeof(line),stdin)){if(!strchr(line,'\n'))while((c=getchar())!='\n'&&c!=EOF){}if(!clean(line,q))continue;if(is_word(q,"quit")||is_word(q,"exit"))break;answer(q,0);}}
    profile();rw_stats("inference");return ferror(stdout)||ferror(stdin)?1:0;
}

#endif /* TINY_APPLE2 */
