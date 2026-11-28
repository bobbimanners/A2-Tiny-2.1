/* TINY for the Apple II (LLVM-MOS): integer-only transformer inference and diagnostics.
 * Model architecture from WollyMonjac's Tiny 2.1.
 * https://github.com/WollyMonjac/Tiny-2.1 (MIT)
 * Convert original TNY2 file offline using tiny-convert.py.
 * Build: see Makefile and README.md
 * Linux reference or native 65C02 MAC backend (TINY_ASM_MAC).
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
#if defined(TINY_MOS)
#define TINY_APPLE2
#endif
#if defined(TINY_MOS) || defined(TINY_TEST_READER)
#include "tiny-prodos.h"
typedef TinyModel ModelFile;
#define model_open(p) tiny_model_open(p)
#define model_read(f,p,n) tiny_model_read(f,p,n)
#define model_getc(f) tiny_model_getc(f)
#define model_error(f) tiny_model_error(f)
#define model_close(f) tiny_model_close(f)
#else
typedef FILE ModelFile;
#define model_open(p) fopen(p,"rb")
#define model_read(f,p,n) fread(p,1,n,f)
#define model_getc(f) fgetc(f)
#define model_error(f) ferror(f)
#define model_close(f) fclose(f)
#endif
#ifdef TINY_MOS
/* Small unbuffered console frontend; do not pull in file stdio. */
#define console_flush() ((void)0)
static void out_str(const char *s) { while(*s) putchar((unsigned char)*s++); }
static void out_u32(uint32_t n) {
    char b[10]; unsigned i=0;
    do { b[i++]=(char)('0'+n%10u); n/=10u; } while(n);
    while(i) putchar(b[--i]);
}
static void out_int(int n) {
    if(n<0){putchar('-');out_u32((uint32_t)(-(int32_t)n));}
    else out_u32((uint32_t)n);
}
static void out_hex(uint32_t n,unsigned digits) {
    static const char hex[]="0123456789ABCDEF";
    while(digits){--digits;putchar(hex[(n>>(digits*4u))&15u]);}
}
static int read_line(char *s,unsigned size) {
    unsigned n=0; int c;
    for(;;){
        c=getchar();
        if(c==EOF){s[n]=0;return n!=0;}
        if(c=='\n'||c=='\r'){putchar('\n');s[n]=0;return 1;}
        if(c==8||c==127){if(n){--n;putchar(8);putchar(' ');putchar(8);}continue;}
        if(c>=32&&c<=126&&n+1<size){s[n++]=(char)c;putchar(c);}
    }
}
#else
#define console_flush() fflush(stdout)
#endif
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
#if defined(TINY_MOS) && !defined(TINY_COPY_MAC)
#define TINY_DIRECT_MAC
#endif
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
#if defined(TINY_MOS) && !defined(TINY_PRODOS_HOST_TEST)
/* Standard model has HD=24; shrink per-head accumulators below so the
 * top 512 bytes of the existing reserved scratch arena can hold gains
 * and the mutually exclusive row staging buffer. */
typedef char gain_fits_scratch[(sizeof(Gain)<=0x200u)?1:-1];
#define gain_work (*(Gain *)(uintptr_t)0x1e00u)
#else
static Gain gain_work;
#endif
static int D,L,H,HD,HID, creativity=0, profiling=0;
static uint32_t att_factor; /* 2^44 / (127^2 * sqrt(HD)) */
/* These lifetimes do not overlap: logits are consumed before the next
 * step, Q dies after attention, and normed dies before attention writes. */
#if defined(TINY_AUX0_LUT) && (!defined(TINY_MOS) || !defined(TINY_COMPACT_LUT))
#error TINY_AUX0_LUT requires TINY_MOS and TINY_COMPACT_LUT
#endif
#ifdef TINY_AUX0_LUT
#define AUX_LUT_SIZE 3456u
#define AUX_LUT_ADDR 0x4000u
#define AUX_EXP_ADDR (AUX_LUT_ADDR+AUX_LUT_SIZE)
#define AUX_EXP_RESERVE 256u
#define AUX_BORROW_SIZE (AUX_LUT_SIZE+AUX_EXP_RESERVE)
#define AUX_LUT_BACKUP (0xc000u-AUX_BORROW_SIZE)
#if !defined(TINY_PRODOS_HOST_TEST)
extern const uint8_t tiny_aux_exp_image[],tiny_aux_exp_image_end[];
extern const uint8_t tiny_aux_exp_gate_end[];
void tiny_aux_exp(uint32_t *d);
#endif
/* The native MAC workspace starts with 352 packed exponential entries.
 * Copy these bytes to auxiliary bank 0 before initializing the MAC. */
#if !defined(TINY_PRODOS_HOST_TEST)
#define TINY_LUT_SPLIT
#define AUX_LUT_HEAD 1056u
extern const uint8_t tiny_exp_lut_head[AUX_LUT_HEAD];
#else
#define AUX_LUT_HEAD 0u
#endif
static uint8_t aux_lut_active;
static void aux_lut_restore(void);
#endif
static Vec work_x,work_normed,work_tmp,work_hidden;
#define score work_tmp
#define query work_hidden
#define att work_normed
/* Wide scratch is declared after the arithmetic helpers. */
static uint32_t packs, zeros, rounded_zero, saturation;
static int min_exp=INT_MAX,max_exp=INT_MIN;
typedef struct { uint16_t m; int exp; } Peak;
static Peak peak_residual,peak_norm,peak_proj,peak_hidden,peak_att,peak_logit;

static void die(const char *s) {
#ifdef TINY_MOS
    out_str("TINY: ");puts(s);
    if(tiny_model_last_error){out_str("ProDOS error $");out_hex(tiny_model_last_error,2);putchar('\n');}
#ifdef TINY_AUX0_LUT
    aux_lut_restore();
#endif
    tiny_model_abort();
#if defined(TINY_WIDE_ASM) && !defined(TINY_PRODOS_HOST_TEST)
    tiny_release_rescale_code();
#endif
    out_str("Press a key."); (void)getchar();
#else
    fprintf(stderr,"TINY: %s\n",s);
#endif
    exit(1);
}
/* Keep bank 0 out of the allocator; 12B temporarily borrows a LUT region.
 * Only $4000..$BFFF of banks 1..127 is available to the allocator.
 * This is a conservative proposed layout, not an Apple II MMU emulator.
 * No inference code may dereference simulated auxiliary memory directly. */
#define RW_BANKS 128u
#define RW_LO 0x4000u
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
    if(rw_next_bank>15u)die("model exceeds 15 verified banks");
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
/* Transfer an already resolved address; both read paths keep the same stats. */
static void rw_read_at(unsigned bank,unsigned off,void *dst,unsigned n) {
    rw_select(bank);
#ifdef TINY_APPLE2
    tiny_rw_read(bank,off,dst,n);
#else
    memcpy(dst,&rw_mem[bank][off],n);
#endif
    ++rw_reads;rw_read_bytes+=n;
}
static void rw_read(const Region *r,unsigned row,unsigned start,void *dst,unsigned n) {
    unsigned bank,off;rw_addr(r,row,start,n,&bank,&off);
    rw_read_at(bank,off,dst,n);
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
#ifdef TINY_MOS
    out_str("RamWorks ");out_str(phase);out_str(": banks=");out_u32(rw_last_bank);
    out_str("/127 payload=");out_u32(rw_payload);out_str(" padding=");out_u32(rw_padding);putchar('\n');
    out_str(" reads=");out_u32(rw_reads);out_str(" bytes=");out_u32(rw_read_bytes);
    out_str(" writes=");out_u32(rw_writes);out_str(" bytes=");out_u32(rw_write_bytes);
    out_str(" bank changes=");out_u32(rw_switches);putchar('\n');
#else
    fprintf(stderr,"RamWorks %s: banks=%u/%u payload=%lu padding=%lu window=%u bytes/bank\n",phase,rw_last_bank,RW_BANKS-1,(unsigned long)rw_payload,(unsigned long)rw_padding,RW_SIZE);
    fprintf(stderr,"  reads=%lu (%lu bytes), writes=%lu (%lu bytes), bank changes=%lu\n",(unsigned long)rw_reads,(unsigned long)rw_read_bytes,(unsigned long)rw_writes,(unsigned long)rw_write_bytes,(unsigned long)rw_switches);
#endif
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
#ifdef TINY_MOS
/* Row transfers and normalization never need the gain vector concurrently.
 * Access through uint8_t is legal; the native target is little-endian. */
typedef char rowbuf_fits_gain[(ROWBUF_SIZE <= sizeof(gain_work.v)) ? 1 : -1];
#define rowbuf ((uint8_t *)gain_work.v)
#else
static uint8_t rowbuf[ROWBUF_SIZE];
#endif
static void read_exact(ModelFile *f,void *p,size_t n) { if(model_read(f,p,n)!=n) die("truncated model"); }
static unsigned read16(ModelFile *f) { unsigned char b[2]; read_exact(f,b,2); return b[0]+256u*b[1]; }
static uint32_t read32(ModelFile *f) {
    unsigned char b[4]; read_exact(f,b,4);
    return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
}
static int read_exp(ModelFile *f) {
    unsigned v=read16(f); int e=v<32768u?(int)v:(int)((int32_t)v-65536);
    if(e < -171 || e > 105)die("invalid model exponent");
    return e;
}
static int32_t read_coeff(ModelFile *f) {
    uint32_t u=read32(f);
    int32_t v=u<=INT32_MAX?(int32_t)u:-1-(int32_t)(~u);
    if(v < -8388608 || v > 8388608)die("invalid model coefficient");
    return (int32_t)v;
}
#ifdef TINY_DIRECT_MAC
/* Store banked weights and KV bytes as signed value + 128. Applying the
 * same transform to a staged embedding restores its original signed bytes.
 * The four scale bytes stay in their original format. */
static void bias_row(unsigned cols) {
    uint8_t i=0;
    do { rowbuf[4u+i]^=0x80; } while(++i!=(uint8_t)cols);
}
#endif
static void load_table(ModelFile *f,Table *t,int rows,int cols) {
    int r;int32_t c;t->rows=rows;t->cols=cols;t->exp=read_exp(f);
    rw_alloc(&t->data,(unsigned)rows,(unsigned)cols+4u);
    for(r=0;r<rows;r++){
        c=read_coeff(f);if(c<0)die("negative row scale");
        put32(rowbuf,(uint32_t)c);read_exact(f,rowbuf+4,cols);
#ifdef TINY_DIRECT_MAC
        bias_row((unsigned)cols);
#endif
        rw_write(&t->data,r,0,rowbuf,(unsigned)cols+4u);
    }
}
static void gains(ModelFile *f,BankGain *g) {
    int i;g->exp=read_exp(f);rw_alloc(&g->data,1,4u*(unsigned)D);
    for(i=0;i<D;i++)put32(rowbuf+4*i,(uint32_t)read_coeff(f));
    rw_write(&g->data,0,0,rowbuf,4u*(unsigned)D);
}
static void fetch_gain(const BankGain *g) {
    gain_work.exp=g->exp;rw_read(&g->data,0,0,rowbuf,4u*(unsigned)D);
#ifndef TINY_MOS
    {int i;for(i=0;i<D;i++)gain_work.v[i]=signed32(get32(rowbuf+4*i));}
#endif
}
static void load_model(const char *path) {
    ModelFile *f=model_open(path);char magic[4];int l;
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
        putchar('.');console_flush();
#endif
    }
#ifdef TINY_APPLE2
    putchar('\n');
#endif
    gains(f,&gf);if(model_getc(f)!=EOF||model_error(f))die("unexpected model length");if(model_close(f))die("model close failed");
    for(l=0;l<L;l++){
        rw_alloc(&layers[l].key,CTX,(unsigned)D+4u);
        rw_alloc(&layers[l].value,CTX,(unsigned)D+4u);
    }
}
/* Two-word, unsigned storage for signed two's-complement or unsigned
 * wide intermediates. All operations use at most 32-bit native types.
 * Pointer API: cc65 cannot pass or return 8-byte structs by value.
 * Arithmetic wraps modulo 2^64; callers retain Stage 6's range bounds. */
/* Optional native unsigned 64-bit backend. Signed values are represented
 * modulo 2^64, just as with the original two-word backend. */
#if defined(TINY_WIDE_ASM)
#ifdef TINY_U64
#error TINY_WIDE_ASM and TINY_U64 are alternative backends
#endif
#include "tiny-wide.h"
#define WLO(a) ((a).lo)
#define WHI(a) ((a).hi)
static void wu(Wide *r,uint32_t a) {r->lo=a;r->hi=0;}
static void wi(Wide *r,int32_t a) {r->lo=(uint32_t)a;r->hi=a<0?UINT32_MAX:0;}
static void wmu(Wide *r,const Wide *a,uint32_t b) {Wide v;wu(&v,b);wm(r,a,&v);}
static void wmi(Wide *r,const Wide *a,int32_t b) {Wide v;wi(&v,b);wm(r,a,&v);}
static void wd(Wide *r,const Wide *a,const Wide *b) {
    if(wz(b))die("wide divide by zero");
    tiny_wdiv(r,a,b);
}
static void wdu(Wide *r,const Wide *a,uint32_t b) {Wide v;wu(&v,b);wd(r,a,&v);}
static void wrounddiv(Wide *r,const Wide *a,const Wide *b) {
    if(wz(b))die("wide divide by zero");
    tiny_wrounddiv(r,a,b);
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
#elif defined(TINY_U64)
typedef uint64_t Wide;
#define WLO(a) ((uint32_t)(a))
#define WHI(a) ((uint32_t)((a)>>32))
static void wu(Wide *r,uint32_t a) {*r=a;}
static void wi(Wide *r,int32_t a) {*r=(uint64_t)(int64_t)a;}
static int wz(const Wide *a) {return !*a;}
static int wn(const Wide *a) {return (*a>>63)!=0;}
static int wc(const Wide *a,const Wide *b) {return *a==*b?0:*a>*b?1:-1;}
static int wcs(const Wide *a,const Wide *b) {
    if(wn(a)!=wn(b))return wn(a)?-1:1;
    return wc(a,b);
}
static void wa(Wide *r,const Wide *a,const Wide *b) {*r=*a+*b;}
static void ws(Wide *r,const Wide *a,const Wide *b) {*r=*a-*b;}
static void wneg(Wide *r,const Wide *a) {*r=0-*a;}
static void wmag(Wide *r,const Wide *a) {*r=wn(a)?0-*a:*a;}
static void wl(Wide *r,const Wide *a,int n) {*r=n<=0?*a:n>=64?0:*a<<n;}
static void wr(Wide *r,const Wide *a,int n) {*r=n<=0?*a:n>=64?0:*a>>n;}
static void winc(Wide *a) {++*a;}
static unsigned wbit(const Wide *a,int n) {return n<0||n>=64?0:(unsigned)((*a>>n)&1);}
static int topbit(const Wide *a) {
    uint32_t v=WHI(*a);int n=v?32:0;
    if(!v)v=WLO(*a);
    while(v>>=1)n++;
    return n;
}
static void wm(Wide *r,const Wide *a,const Wide *b) {*r=*a * *b;}
static void wmu(Wide *r,const Wide *a,uint32_t b) {*r=*a*b;}
static void wmi(Wide *r,const Wide *a,int32_t b) {*r=*a*(uint64_t)(int64_t)b;}
static void wd(Wide *r,const Wide *a,const Wide *b) {
    if(!*b)die("wide divide by zero");
    *r=*a / *b;
}
static void wdu(Wide *r,const Wide *a,uint32_t b) {Wide v=b;wd(r,a,&v);}
/* Use one division and recover remainder; avoids overflow on rounding. */
static void wrounddiv(Wide *r,const Wide *a,const Wide *b) {
    Wide q,rem;wd(&q,a,b);rem=*a-q* *b;
    *r=q+(rem>=((*b>>1)+(*b&1)));
}
static void rshift(Wide *r,const Wide *a,int n) {
    Wide m;int neg=wn(a);
    if(n<=0){*r=*a;return;}
    wmag(&m,a);wr(r,&m,n);if(wbit(&m,n-1))winc(r);
    if(neg)wneg(r,r);
}
static void rdiv(Wide *r,const Wide *a,uint32_t d) {
    Wide m,b=d;int neg=wn(a);wmag(&m,a);wrounddiv(r,&m,&b);if(neg)wneg(r,r);
}
static int32_t wtoi(const Wide *a) {
    if(*a<=INT32_MAX)return (int32_t)*a;
    if(*a>=UINT64_MAX-INT32_MAX)return -1-(int32_t)(uint32_t)~*a;
    die("wide result does not fit signed 32 bits");return 0;
}
#else
#define WLO(a) ((a).lo)
#define WHI(a) ((a).hi)
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
#endif
/* Scratch phases are mutually exclusive. Attention keeps its per-head
 * scores here until the last head is done, then reuses it for final packing.
 * pick() runs after step() and needs neither of those arrays. */
typedef struct { uint16_t m; int16_t exp; } PackedScale;
typedef struct {
    union {
        Wide wide_values[MAXHID];
        Wide attention_scores[CTX];
        uint32_t pick_weights[VOCAB];
        /* While heads are scored the space above the scores is idle.
         * It holds each cached position's key and value scale, read
         * once per attention call instead of once per head. */
        struct { Wide scores[CTX]; PackedScale key[CTX], value[CTX]; } attn;
    } phase;
    union {
        Wide attention_result[MAXD];
#ifdef TINY_MOS
        /* Prepared MAC uses two 256-byte planes inside this union. */
        int8_t matvec_q[2*MAXHID];
#else
        int8_t matvec_q[MAXHID];
#endif
    } projection;
#ifdef TINY_MOS
    /* Native load_model enforces D=96,H=4, hence HD=24. */
    int32_t attention_accum[24];
#else
    int32_t attention_accum[MAXD];
#endif
    int attention_exps[CTX], attention_out_exp[MAXD];
    int8_t attention_aq[MAXD], attention_av[CTX];
} Scratch;
#if defined(TINY_MOS) && !defined(TINY_PRODOS_HOST_TEST)
/* SYS file begins at $2000. The model reader uses $0800-$0FFF.
 * Scratch is explicitly assigned $1000-$1FFF, separate from CRT BSS.
 * $1000-$1DFF is Scratch; $1E00-$1FFF holds Gain/rowbuf.
 * Nothing relies on zero initialization of this arena. */
typedef char scratch_must_fit[(sizeof(Scratch) <= 0x0e00u) ? 1 : -1];
#define scratch (*(Scratch *)(uintptr_t)0x1000u)
static void reserve_scratch(void) {
    volatile uint8_t *bitmap = (volatile uint8_t *)(uintptr_t)0xbf5au;
    if (bitmap[0] || bitmap[1]) die("$1000-$1FFF unavailable");
    bitmap[0] = bitmap[1] = 0xff;
}
#else
static Scratch scratch;
#endif
#define wide (scratch.phase.wide_values)
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
    Wide m,a;int i,shift;
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    int16_t pargs[2];
    (void)i;
    tiny_wmaxmag(&m,src,(unsigned)n);
#else
    Wide v;int32_t val;
    wu(&m,0);
    for(i=0;i<n;i++){wmag(&a,&src[i]);if(wc(&a,&m)>0)m=a;}
#endif
    if(profiling)packs++;
    if(wz(&m)){memset(dst->v,0,(size_t)n*sizeof(dst->v[0]));dst->exp=0;if(profiling)zeros++;return;}
    /* Match the old repeated ceiling shifts: ceil(m / 2^shift) <= 32767.
     * topbit gives the candidate; only the upper boundary needs correction. */
    shift=topbit(&m)-14;
    if(shift>0){
        wu(&a,32767UL);wl(&a,&a,shift);
        if(wc(&m,&a)>0)shift++;
    }
    dst->exp=exp+shift;
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    /* The peak calculation above guarantees -32767..32767. */
    pargs[0]=(int16_t)n;pargs[1]=(int16_t)shift;
    tiny_pack_rows(dst->v,src,pargs);
#else
    for(i=0;i<n;i++){
        if(shift>=0)rshift(&v,&src[i],shift);else wl(&v,&src[i],-shift);
        val=wtoi(&v);
        if(val>32767L||val< -32767L){if(profiling)saturation++;die("fixed-point packing overflow");}
        if(profiling&&!wz(&src[i])&&!val)rounded_zero++;
        dst->v[i]=(int16_t)val;
    }
#endif
}
static int peak16(const int16_t *v,int n) {int i;int32_t m=0,a;for(i=0;i<n;i++){a=v[i];if(a<0)a=-a;if(a>m)m=a;}if(m>32767L)die("invalid packed vector");return (int)m;}
static int q8(const int16_t *v,int8_t *out,int n) {
    int m=peak16(v,n);
    if(!m){memset(out,0,(size_t)n);return 0;}
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    /* Native dimensions are bounded by MAXHID=256. A nonzero peak
     * implies n>0. The packed-vector peak bounds every magnitude. */
    uint16_t args[2];
    args[0]=(uint16_t)n;args[1]=(uint16_t)m;
    tiny_q8_vector(v,out,args);
#else
    int i;int32_t a;
    for(i=0;i<n;i++){a=(int32_t)v[i]*127L;out[i]=(int8_t)(a<0?-((-a+m/2)/m):(a+m/2)/m);}
#endif
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
uint16_t tiny_mac_stride;
uint8_t tiny_mac_rows;
int32_t *tiny_mac_acc;
extern void tiny_mac65_dot(void);
extern void tiny_mac65_axpy(void);
#ifdef TINY_MOS
extern void tiny_mac65_prepare(void);
extern void tiny_mac65_prepared_dot(void);
extern void tiny_mac65_aux_dot(void);
extern void tiny_mac65_aux_rows(void);
extern void tiny_mac65_axpy_begin(void);
extern void tiny_mac65_aux_axpy(void);
extern void tiny_mac65_axpy_end(void);
extern const uint8_t tiny_mac65_aux_end[], ql[];
extern int8_t tiny_mac65_work[512];
extern void tiny_mac65_init(void);
#endif
/* Native matrix rows and attention scores both use the prepared kernel,
 * so the general dot is only needed by other assembly-MAC targets. */
#ifndef TINY_MOS
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
#endif
#ifndef TINY_DIRECT_MAC
static void axpy8(int32_t *accum,int8_t c,const int8_t *v,int n) {
    tiny_mac_acc=accum;tiny_mac_coeff=c;tiny_mac_b=v;
    tiny_mac_n=(uint16_t)n;tiny_mac65_axpy();
}
#endif
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
#ifdef TINY_DIRECT_MAC
#define projection_q tiny_mac65_work
#else
#define projection_q scratch.projection.matvec_q
#endif
/* These describe projection_q until the next prepare or attention head.
 * cache_store uses rowbuf, so K, V and Q can consume the same input. */
static int projection_peak, projection_exp;
static void prepare_projection(const Vec *src,int cols) {
    projection_peak=q8(src->v,projection_q,cols);
    projection_exp=src->exp-7; /* absorb the rescaling divisor (128) */
#ifdef TINY_MOS
    tiny_mac_a=(const int8_t *)(rowbuf+4);
    tiny_mac_b=projection_q;
    tiny_mac_n=(uint16_t)cols;
    tiny_mac65_prepare();
#ifdef TINY_DIRECT_MAC
    tiny_mac_b=(const int8_t *)rowbuf;
#endif
#endif
}
static void matvec_prepared(const Table *t,Vec *dst) {
    int r,m=projection_peak,e=projection_exp+t->exp;
    unsigned bank=t->data.bank,off=t->data.off,n=t->data.stride;
    unsigned last=RW_HI-n;
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    TinyRescaleArgs args;
    args.peak=(uint16_t)m;
#else
    Wide a;
#endif
    /* rw_alloc packs complete records consecutively, skipping only the
     * unusable tail of a bank. Walk that layout without mul/div per row. */
#ifdef TINY_DIRECT_MAC
    /* Up to eight rows per kernel call; a batch never spans two banks. */
    unsigned room=t->data.first;
    (void)last;
    tiny_mac_stride=(uint16_t)n;
#if defined(TINY_WIDE_ASM)
    /* The native loader enforces 96/4/4/256. F1 is the only projection
     * with MAXHID rows, and its packed negative results feed only ReLU. */
    if(t->rows==MAXHID)tiny_f1_begin();
#endif
    for(r=0;r<t->rows;){
        unsigned k=(unsigned)(t->rows-r);
#if defined(TINY_WIDE_ASM)
        uint16_t rargs[2];
#else
        unsigned i;
#endif
        if(!room){++bank;off=RW_LO;room=t->data.per;}
        if(k>8u)k=8u;
        if(k>room)k=room;
        tiny_mac_a=(const int8_t *)(uintptr_t)off;
        tiny_mac_coeff=(int8_t)bank;
        tiny_mac_rows=(uint8_t)k;
        tiny_mac65_aux_rows();
        rw_select(bank);
#if defined(TINY_WIDE_ASM)
        rargs[0]=(uint16_t)k;rargs[1]=args.peak;
        tiny_rescale_rows(&wide[r],rowbuf,rargs);
#else
        for(i=0;i<k;i++){
            const uint8_t *res=rowbuf+8u*i;
            wi(&a,signed32(get32(res+4)));
            wmi(&a,&a,signed32(get32(res)));wmu(&wide[r+(int)i],&a,(uint32_t)m);
        }
#endif
        off+=k*n;r+=(int)k;room-=k;
    }
#if defined(TINY_WIDE_ASM)
    if(t->rows==MAXHID)tiny_f1_finish();
#endif
    /* Count logical reads, not bytes copied into main memory. */
    rw_reads+=(uint32_t)t->rows;rw_read_bytes+=(uint32_t)t->rows*n;
#else
    for(r=0;r<t->rows;r++){
        if(off>last){++bank;off=RW_LO;}
        rw_read_at(bank,off,rowbuf,n);
#ifdef TINY_MOS
        tiny_mac65_prepared_dot();
#endif
        off+=n;
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
        args.dot=tiny_mac_result;
        tiny_rescale(&wide[r],rowbuf,&args);
#else
#ifdef TINY_MOS
        wi(&a,tiny_mac_result);
#else
        wi(&a,dot8((const int8_t *)(rowbuf+4),projection_q,t->cols));
#endif
        wmi(&a,&a,signed32(get32(rowbuf)));wmu(&wide[r],&a,(uint32_t)m);
#endif
    }
#endif
    pack(dst,wide,t->rows,e);
}
static void matvec(const Table *t,const Vec *src,Vec *dst) {
    prepare_projection(src,t->cols);
    matvec_prepared(t,dst);
}
static void row(const Table *t,int r,Vec *dst) {
    int i;int32_t c;rw_read(&t->data,r,0,rowbuf,(unsigned)t->cols+4u);c=signed32(get32(rowbuf));
#ifdef TINY_DIRECT_MAC
    bias_row((unsigned)t->cols);
#endif
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
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    {
        /* One shift is 16; the other is at most that. A 16-bit value
         * shifted right by 16 or more rounds to zero. */
        TinyResidualArgs ra;int sa=a->exp-e,sb=b->exp-e;
        (void)i;(void)av;(void)bv;
        ra.b=b->v;ra.n=(uint16_t)n;
        ra.sa=(int8_t)(sa< -16?-16:sa);ra.sb=(int8_t)(sb< -16?-16:sb);
        tiny_residual_rows(wide,a->v,&ra);
    }
#else
    for(i=0;i<n;i++){
        wi(&av,a->v[i]);wi(&bv,b->v[i]);align(&av,&av,a->exp,e);align(&bv,&bv,b->exp,e);wa(&wide[i],&av,&bv);
    }
#endif
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
    return WLO(root)+(wc(&n,&root)>0);
}
static void normalize(const Vec *src,const Gain *g,Vec *dst) {
    int i,xe,hi,eh,e;Wide ss,mean,rad,t,eps,b,inv;uint32_t root;
#ifdef TINY_U64
    eps=UINT64_C(11258999068);
#else
    eps.hi=2;eps.lo=U32(2669064476);
#endif
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    tiny_sumsq(&ss,src->v,(unsigned)D);
#else
    wu(&ss,0);
    for(i=0;i<D;i++){int32_t v=src->v[i];wu(&t,(uint32_t)(v*v));wa(&ss,&ss,&t);}
#endif
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
#ifdef TINY_DIRECT_MAC
    bias_row((unsigned)D);
#endif
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
 * generated offline, not at program startup, and occupies 4612 bytes. */
#define EXP_ONE U32(16777216)
#define EXP_LIMIT (U32(18)<<16)
/* The first 354 stored entries need three bytes, the next 355 need two,
 * and the last 443 fit in one. Both representations use the same values. */
#define EXP_COUNT24 354u
#define EXP_COUNT16 355u
#define EXP_COUNT8 443u
#ifdef TINY_COMPACT_LUT
#define E24(v) (uint8_t)(v),(uint8_t)((v)>>8),(uint8_t)((v)>>16)
#define E16(v) (uint8_t)(v),(uint8_t)((v)>>8)
static const uint8_t exp_lut_compact[] = {
#else
#define E24(v) (v)
#define E16(v) (v)
static const uint32_t exp_lut[] = {
    EXP_ONE,
#endif
#ifndef TINY_LUT_SPLIT
    E24(16517109), E24(16261035), E24(16008931), E24(15760736), E24(15516388), E24(15275829),
    E24(15038999), E24(14805841), E24(14576298), E24(14350313), E24(14127832), E24(13908801),
    E24(13693165), E24(13480872), E24(13271870), E24(13066109), E24(12863538), E24(12664107),
    E24(12467768), E24(12274473), E24(12084175), E24(11896828), E24(11712384), E24(11530801),
    E24(11352032), E24(11176035), E24(11002767), E24(10832185), E24(10664247), E24(10498913),
    E24(10336143), E24(10175896), E24(10018133), E24(9862816), E24(9709908), E24(9559370),
    E24(9411165), E24(9265259), E24(9121614), E24(8980197), E24(8840972), E24(8703905),
    E24(8568963), E24(8436114), E24(8305324), E24(8176562), E24(8049796), E24(7924996),
    E24(7802130), E24(7681169), E24(7562084), E24(7444844), E24(7329423), E24(7215791),
    E24(7103920), E24(6993784), E24(6885356), E24(6778608), E24(6673515), E24(6570052),
    E24(6468193), E24(6367913), E24(6269188), E24(6171993), E24(6076305), E24(5982101),
    E24(5889357), E24(5798051), E24(5708160), E24(5619663), E24(5532539), E24(5446765),
    E24(5362320), E24(5279185), E24(5197339), E24(5116762), E24(5037434), E24(4959336),
    E24(4882448), E24(4806753), E24(4732231), E24(4658865), E24(4586636), E24(4515526),
    E24(4445520), E24(4376598), E24(4308745), E24(4241945), E24(4176179), E24(4111434),
    E24(4047692), E24(3984938), E24(3923157), E24(3862334), E24(3802454), E24(3743503),
    E24(3685465), E24(3628327), E24(3572075), E24(3516696), E24(3462174), E24(3408498),
    E24(3355654), E24(3303630), E24(3252412), E24(3201988), E24(3152345), E24(3103473),
    E24(3055358), E24(3007989), E24(2961355), E24(2915443), E24(2870243), E24(2825744),
    E24(2781935), E24(2738805), E24(2696344), E24(2654541), E24(2613386), E24(2572869),
    E24(2532981), E24(2493711), E24(2455049), E24(2416987), E24(2379515), E24(2342624),
    E24(2306305), E24(2270549), E24(2235348), E24(2200692), E24(2166573), E24(2132984),
    E24(2099915), E24(2067359), E24(2035307), E24(2003753), E24(1972687), E24(1942104),
    E24(1911994), E24(1882351), E24(1853168), E24(1824438), E24(1796152), E24(1768306),
    E24(1740891), E24(1713901), E24(1687329), E24(1661169), E24(1635415), E24(1610061),
    E24(1585099), E24(1560524), E24(1536330), E24(1512512), E24(1489063), E24(1465977),
    E24(1443249), E24(1420873), E24(1398845), E24(1377158), E24(1355807), E24(1334787),
    E24(1314093), E24(1293720), E24(1273663), E24(1253916), E24(1234476), E24(1215337),
    E24(1196495), E24(1177945), E24(1159683), E24(1141704), E24(1124003), E24(1106577),
    E24(1089421), E24(1072532), E24(1055903), E24(1039533), E24(1023417), E24(1007550),
    E24(991930), E24(976551), E24(961411), E24(946506), E24(931832), E24(917385),
    E24(903162), E24(889160), E24(875375), E24(861803), E24(848442), E24(835288),
    E24(822338), E24(809589), E24(797038), E24(784681), E24(772515), E24(760539),
    E24(748748), E24(737139), E24(725711), E24(714460), E24(703383), E24(692478),
    E24(681743), E24(671173), E24(660768), E24(650523), E24(640438), E24(630509),
    E24(620734), E24(611110), E24(601636), E24(592308), E24(583125), E24(574085),
    E24(565184), E24(556422), E24(547796), E24(539303), E24(530942), E24(522710),
    E24(514606), E24(506628), E24(498773), E24(491041), E24(483428), E24(475933),
    E24(468554), E24(461290), E24(454138), E24(447098), E24(440166), E24(433342),
    E24(426624), E24(420009), E24(413498), E24(407087), E24(400776), E24(394562),
    E24(388445), E24(382423), E24(376494), E24(370657), E24(364910), E24(359253),
    E24(353683), E24(348200), E24(342802), E24(337487), E24(332255), E24(327104),
    E24(322032), E24(317040), E24(312124), E24(307285), E24(302521), E24(297831),
    E24(293214), E24(288668), E24(284193), E24(279787), E24(275449), E24(271178),
    E24(266974), E24(262835), E24(258760), E24(254749), E24(250799), E24(246911),
    E24(243083), E24(239314), E24(235604), E24(231951), E24(228355), E24(224815),
    E24(221329), E24(217898), E24(214520), E24(211194), E24(207920), E24(204696),
    E24(201523), E24(198398), E24(195323), E24(192294), E24(189313), E24(186378),
    E24(183489), E24(180644), E24(177843), E24(175086), E24(172372), E24(169699),
    E24(167068), E24(164478), E24(161928), E24(159418), E24(156946), E24(154513),
    E24(152117), E24(149759), E24(147437), E24(145151), E24(142901), E24(140686),
    E24(138504), E24(136357), E24(134243), E24(132162), E24(130113), E24(128096),
    E24(126110), E24(124155), E24(122230), E24(120335), E24(118469), E24(116632),
    E24(114824), E24(113044), E24(111291), E24(109566), E24(107867), E24(106195),
    E24(104549), E24(102928), E24(101332), E24(99761), E24(98214), E24(96692),
    E24(95193), E24(93717), E24(92264), E24(90833), E24(89425), E24(88039),
    E24(86674), E24(85330), E24(84007), E24(82705), E24(81423), E24(80160),
    E24(78917), E24(77694), E24(76489), E24(75304), E24(74136), E24(72987),
    E24(71855), E24(70741), E24(69644), E24(68565),
#endif
    E24(67502), E24(66455),
    E16(65425), E16(64411), E16(63412), E16(62429), E16(61461), E16(60508), E16(59570), E16(58646),
    E16(57737), E16(56842), E16(55961), E16(55093), E16(54239), E16(53398), E16(52570), E16(51755),
    E16(50953), E16(50163), E16(49385), E16(48620), E16(47866), E16(47124), E16(46393), E16(45674),
    E16(44966), E16(44269), E16(43582), E16(42907), E16(42241), E16(41587), E16(40942), E16(40307),
    E16(39682), E16(39067), E16(38461), E16(37865), E16(37278), E16(36700), E16(36131), E16(35571),
    E16(35019), E16(34476), E16(33942), E16(33416), E16(32898), E16(32388), E16(31886), E16(31391),
    E16(30905), E16(30425), E16(29954), E16(29489), E16(29032), E16(28582), E16(28139), E16(27703),
    E16(27273), E16(26850), E16(26434), E16(26024), E16(25621), E16(25224), E16(24832), E16(24447),
    E16(24068), E16(23695), E16(23328), E16(22966), E16(22610), E16(22260), E16(21915), E16(21575),
    E16(21240), E16(20911), E16(20587), E16(20268), E16(19953), E16(19644), E16(19340), E16(19040),
    E16(18745), E16(18454), E16(18168), E16(17886), E16(17609), E16(17336), E16(17067), E16(16802),
    E16(16542), E16(16286), E16(16033), E16(15784), E16(15540), E16(15299), E16(15062), E16(14828),
    E16(14598), E16(14372), E16(14149), E16(13930), E16(13714), E16(13501), E16(13292), E16(13086),
    E16(12883), E16(12683), E16(12487), E16(12293), E16(12102), E16(11915), E16(11730), E16(11548),
    E16(11369), E16(11193), E16(11019), E16(10849), E16(10680), E16(10515), E16(10352), E16(10191),
    E16(10033), E16(9878), E16(9725), E16(9574), E16(9425), E16(9279), E16(9135), E16(8994),
    E16(8854), E16(8717), E16(8582), E16(8449), E16(8318), E16(8189), E16(8062), E16(7937),
    E16(7814), E16(7693), E16(7573), E16(7456), E16(7340), E16(7227), E16(7115), E16(7004),
    E16(6896), E16(6789), E16(6684), E16(6580), E16(6478), E16(6378), E16(6279), E16(6181),
    E16(6085), E16(5991), E16(5898), E16(5807), E16(5717), E16(5628), E16(5541), E16(5455),
    E16(5370), E16(5287), E16(5205), E16(5124), E16(5045), E16(4967), E16(4890), E16(4814),
    E16(4739), E16(4666), E16(4594), E16(4522), E16(4452), E16(4383), E16(4315), E16(4248),
    E16(4182), E16(4118), E16(4054), E16(3991), E16(3929), E16(3868), E16(3808), E16(3749),
    E16(3691), E16(3634), E16(3577), E16(3522), E16(3467), E16(3414), E16(3361), E16(3309),
    E16(3257), E16(3207), E16(3157), E16(3108), E16(3060), E16(3013), E16(2966), E16(2920),
    E16(2875), E16(2830), E16(2786), E16(2743), E16(2700), E16(2659), E16(2617), E16(2577),
    E16(2537), E16(2497), E16(2459), E16(2421), E16(2383), E16(2346), E16(2310), E16(2274),
    E16(2239), E16(2204), E16(2170), E16(2136), E16(2103), E16(2070), E16(2038), E16(2007),
    E16(1976), E16(1945), E16(1915), E16(1885), E16(1856), E16(1827), E16(1799), E16(1771),
    E16(1744), E16(1716), E16(1690), E16(1664), E16(1638), E16(1612), E16(1587), E16(1563),
    E16(1539), E16(1515), E16(1491), E16(1468), E16(1445), E16(1423), E16(1401), E16(1379),
    E16(1358), E16(1337), E16(1316), E16(1296), E16(1276), E16(1256), E16(1236), E16(1217),
    E16(1198), E16(1180), E16(1161), E16(1143), E16(1126), E16(1108), E16(1091), E16(1074),
    E16(1057), E16(1041), E16(1025), E16(1009), E16(993), E16(978), E16(963), E16(948),
    E16(933), E16(919), E16(905), E16(890), E16(877), E16(863), E16(850), E16(837),
    E16(824), E16(811), E16(798), E16(786), E16(774), E16(762), E16(750), E16(738),
    E16(727), E16(716), E16(704), E16(694), E16(683), E16(672), E16(662), E16(652),
    E16(641), E16(631), E16(622), E16(612), E16(603), E16(593), E16(584), E16(575),
    E16(566), E16(557), E16(549), E16(540), E16(532), E16(523), E16(515), E16(507),
    E16(500), E16(492), E16(484), E16(477), E16(469), E16(462), E16(455), E16(448),
    E16(441), E16(434), E16(427), E16(421), E16(414), E16(408), E16(401), E16(395),
    E16(389), E16(383), E16(377), E16(371), E16(365), E16(360), E16(354), E16(349),
    E16(343), E16(338), E16(333), E16(328), E16(323), E16(318), E16(313), E16(308),
    E16(303), E16(298), E16(294), E16(289), E16(285), E16(280), E16(276), E16(272),
    E16(267), E16(263), E16(259),
    255, 251, 247, 243, 240, 236, 232, 229, 225, 222, 218, 215, 212, 208, 205, 202,
    199, 196, 193, 190, 187, 184, 181, 178, 175, 173, 170, 167, 165, 162, 160, 157,
    155, 152, 150, 148, 145, 143, 141, 139, 137, 134, 132, 130, 128, 126, 124, 122,
    121, 119, 117, 115, 113, 111, 110, 108, 106, 105, 103, 101, 100, 98, 97, 95,
    94, 92, 91, 90, 88, 87, 85, 84, 83, 82, 80, 79, 78, 77, 75, 74,
    73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58,
    57, 56, 55, 54, 53, 53, 52, 51, 50, 49, 49, 48, 47, 46, 46, 45,
    44, 44, 43, 42, 42, 41, 40, 40, 39, 39, 38, 37, 37, 36, 36, 35,
    35, 34, 33, 33, 32, 32, 31, 31, 30, 30, 30, 29, 29, 28, 28, 27,
    27, 26, 26, 26, 25, 25, 24, 24, 24, 23, 23, 23, 22, 22, 22, 21,
    21, 21, 20, 20, 20, 19, 19, 19, 18, 18, 18, 18, 17, 17, 17, 17,
    16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13,
    13, 13, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 10, 10, 10,
    10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
#undef E24
#undef E16
#ifdef TINY_COMPACT_LUT
#ifdef TINY_AUX0_LUT
/* Save the borrowed auxiliary region in the unused tail of model bank 15.
 * No ProDOS /RAM disk access is allowed while the overlay is active.
 * Restore on normal exit (including EOF) and application errors. */
static void aux_lut_copy(unsigned from,unsigned src,unsigned to,unsigned dst) {
    unsigned i,n;
    for(i=0;i<AUX_BORROW_SIZE;i+=n){
        n=AUX_BORROW_SIZE-i;
        if(n>ROWBUF_SIZE)n=ROWBUF_SIZE;
        tiny_rw_read(from,src+i,rowbuf,n);
        tiny_rw_write(to,dst+i,rowbuf,n);
    }
}
/* The wide scratch arena is idle during startup. Expand the narrow runs
 * here so the auxiliary exponent kernel keeps its three-byte indexing. */
static void aux_lut_expand(unsigned addr,const uint8_t *src,unsigned count,unsigned width) {
    uint8_t *out=(uint8_t *)scratch.phase.wide_values;
    unsigned i;
    for(i=0;i<count;i++){
        *out++=*src++;
        *out++=width==2?*src++:0;
        *out++=0;
    }
    tiny_rw_write(0,addr,scratch.phase.wide_values,3u*count);
}
_Static_assert(sizeof(scratch.phase.wide_values)>=3u*EXP_COUNT8,"exponential LUT scratch");
static void aux_lut_begin(void) {
#if !defined(TINY_PRODOS_HOST_TEST)
    unsigned n=(unsigned)((uintptr_t)tiny_aux_exp_image_end-(uintptr_t)tiny_aux_exp_image);
    if((uintptr_t)tiny_aux_exp<0x2000u ||
       (uintptr_t)tiny_aux_exp_gate_end>0x4000u || !n || n>AUX_EXP_RESERVE)
        die("Aux code placement");
#endif
    if(rw_last_bank>=15u && rw_next_off>AUX_LUT_BACKUP)
        die("No aux LUT backup space");
    aux_lut_copy(0,AUX_LUT_ADDR,15,AUX_LUT_BACKUP);
#ifdef TINY_LUT_SPLIT
    tiny_rw_write(0,AUX_LUT_ADDR,tiny_exp_lut_head,AUX_LUT_HEAD);
#endif
    tiny_rw_write(0,AUX_LUT_ADDR+AUX_LUT_HEAD,exp_lut_compact,3u*EXP_COUNT24-AUX_LUT_HEAD);
    aux_lut_expand(AUX_LUT_ADDR+3u*EXP_COUNT24,
                   exp_lut_compact+3u*EXP_COUNT24-AUX_LUT_HEAD,EXP_COUNT16,2);
    aux_lut_expand(AUX_LUT_ADDR+3u*(EXP_COUNT24+EXP_COUNT16),
                   exp_lut_compact+3u*EXP_COUNT24-AUX_LUT_HEAD+2u*EXP_COUNT16,EXP_COUNT8,1);
    aux_lut_active=1;
#if !defined(TINY_PRODOS_HOST_TEST)
    tiny_rw_write(0,AUX_EXP_ADDR,tiny_aux_exp_image,n);
#endif
}
static void aux_lut_restore(void) {
    if(aux_lut_active){
        aux_lut_copy(15,AUX_LUT_BACKUP,0,AUX_LUT_ADDR);
        aux_lut_active=0;
    }
}
#endif
static uint32_t exp_entry_bytes(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16);
}
#ifndef TINY_AUX0_LUT
static uint32_t exp_compact_entry(unsigned i) {
    const uint8_t *p=exp_lut_compact;
    if(i<EXP_COUNT24)return exp_entry_bytes(p+3u*i);
    i-=EXP_COUNT24;p+=3u*EXP_COUNT24;
    if(i<EXP_COUNT16)return get16(p+2u*i);
    return p[2u*EXP_COUNT16+i-EXP_COUNT16];
}
#endif
#endif
static uint32_t exp_neg_q16(uint32_t d) {
#if defined(TINY_AUX0_LUT) && !defined(TINY_PRODOS_HOST_TEST)
    if(d>=EXP_LIMIT)return 0;
    tiny_aux_exp(&d);
    return d;
#else
    uint32_t i,f,a,b;
    if(d>=EXP_LIMIT)return 0;
    i=d>>10; f=d&1023u;
#ifdef TINY_COMPACT_LUT
#ifdef TINY_AUX0_LUT
    /* Fetch adjacent entries together, in one bank switch. Entry 0 is
     * implicit since its exact value requires 25 bits. */
    { uint8_t pair[6];
      if(!i){
          tiny_rw_read(0,AUX_LUT_ADDR,pair,3);a=EXP_ONE;b=exp_entry_bytes(pair);
      }else{
          tiny_rw_read(0,AUX_LUT_ADDR+3u*((unsigned)i-1u),pair,6);
          a=exp_entry_bytes(pair);b=exp_entry_bytes(pair+3);
      }
    }
#else
    a=i?exp_compact_entry((unsigned)i-1u):EXP_ONE;
    b=exp_compact_entry((unsigned)i);
#endif
#else
    a=exp_lut[i];b=exp_lut[i+1];
#endif
    return a-(((a-b)*f+512u)>>10);
#endif
}
static void update_hi(const Wide *v,int e,int *hi) {
    Wide m;int b;if(wz(v))return;wmag(&m,v);b=topbit(&m)+e;if(b>*hi)*hi=b;
}
/* Nonnegative difference -> Q16, clipped to the exponential table. */
static uint32_t delta_q16(const Wide *m,int e,unsigned mul,unsigned div) {
    Wide a,b,v,limit;int shift=e+16;
    if(wz(m))return 0;
    if(mul==1)a=*m;else wmu(&a,m,mul);
    wu(&limit,EXP_LIMIT*(uint32_t)div);
    if(shift>=0){
        wr(&v,&limit,shift);if(shift>=63||wc(&a,&v)>0)return EXP_LIMIT;
        wl(&v,&a,shift);
        if(div==1)return WLO(v); /* already clipped to EXP_LIMIT */
        wu(&b,div);
    }else{
        if(shift<=-60)return 0;
        if(div==1){ /* exact unsigned round-to-nearest power-of-two division */
            wr(&v,&a,-shift);if(wbit(&a,-shift-1))winc(&v);
        }else{
            v=a;wu(&b,div);wl(&b,&b,-shift);
        }
    }
    if(div!=1)wrounddiv(&v,&v,&b);
    return WHI(v)||WLO(v)>=EXP_LIMIT?EXP_LIMIT:WLO(v);
}
/* One attention score: raw = dot*m*csm, normalized to at most 30 bits with
 * rounding, times the attention factor. Returns the normalizing shift;
 * *top is the top bit of the magnitude, or -1 for a zero score. */
#if !(defined(TINY_DIRECT_MAC) && defined(TINY_WIDE_ASM))
static int att_score(Wide *out,int32_t dot,int m,unsigned csm,int *top) {
    Wide raw,t;int shift;
    wi(&raw,dot);wmu(&raw,&raw,(uint32_t)m);wmu(&raw,&raw,csm);
    wmag(&t,&raw);shift=wz(&t)?0:topbit(&t)-29;if(shift<0)shift=0;
    rshift(&t,&raw,shift);wmu(out,&t,att_factor);
    if(wz(out))*top=-1;else{wmag(&t,out);*top=topbit(&t);}
    return shift;
}
#endif
#if defined(TINY_DIRECT_MAC) && defined(TINY_WIDE_ASM)
/* Shared with tiny_att_batch in tiny-mac65-mos.s. */
TinyScoreArgs tiny_att_args;
Wide *tiny_att_out;
int16_t *tiny_att_exps;
PackedScale *tiny_att_ks;
int16_t tiny_att_base, tiny_att_hi;
uint8_t tiny_att_first;
extern void tiny_att_batch(void);
#endif
#ifdef TINY_DIRECT_MAC
/* Score every cached key against the prepared query head. Walks the key
 * records in place, up to eight per kernel call, as matvec walks matrix
 * rows. Each result is the four bytes ahead of the weights and the dot; for
 * head 0 those bytes are the record's scale, kept for the other heads.
 * Returns the highest magnitude bit over all scores, or INT_MIN. */
static int att_key_scores(const Layer *a,int p,int h,int qexp,int m) {
    PackedScale *ks=scratch.phase.attn.key;
    unsigned kbank=a->key.bank,koff=a->key.off,room=a->key.first,kstride=a->key.stride;
    int j;
#if defined(TINY_WIDE_ASM)
    typedef char exps_are_16_bit[(sizeof(int)==2)?1:-1];
    tiny_att_args.m=(uint16_t)m;tiny_att_args.factor=att_factor;
    tiny_att_out=scratch.phase.attention_scores;tiny_att_exps=(int16_t *)scratch.attention_exps;
    tiny_att_ks=ks;tiny_att_base=(int16_t)(qexp-44);tiny_att_hi=INT16_MIN;tiny_att_first=(uint8_t)!h;
#else
    int shift,top,hi=INT_MIN;
#endif
    tiny_mac_b=(const int8_t *)rowbuf;tiny_mac_stride=(uint16_t)kstride;
    for(j=0;j<=p;){
        unsigned k=(unsigned)(p+1-j);
        if(!room){++kbank;koff=RW_LO;room=a->key.per;}
        if(k>8u)k=8u;
        if(k>room)k=room;
        tiny_mac_a=(const int8_t *)(uintptr_t)(koff+(unsigned)h*(unsigned)HD);
        tiny_mac_coeff=(int8_t)kbank;tiny_mac_rows=(uint8_t)k;
        tiny_mac65_aux_rows();
        koff+=k*kstride;room-=k;
#if defined(TINY_WIDE_ASM)
        tiny_att_batch();j+=(int)k;
#else
        {unsigned i;
        for(i=0;i<k;i++,j++){
            const uint8_t *res=rowbuf+8u*i;
            if(!h){ks[j].m=(uint16_t)get16(res);ks[j].exp=(int16_t)signed16(get16(res+2));}
            shift=att_score(&scratch.phase.attention_scores[j],signed32(get32(res+4)),m,ks[j].m,&top);
            scratch.attention_exps[j]=qexp+ks[j].exp+shift-44;
            if(top>=0&&top+scratch.attention_exps[j]>hi)hi=top+scratch.attention_exps[j];
        }}
#endif
    }
#if defined(TINY_WIDE_ASM)
    return tiny_att_hi==INT16_MIN?INT_MIN:tiny_att_hi;
#else
    return hi;
#endif
}
/* Sum the cached values of one head, weighted by attention_av, in place.
 * The kernel is fixed at 24 values per head. */
static void att_value_sums(const Layer *a,int p,int h) {
    unsigned head=4u+(unsigned)h*(unsigned)HD;
    unsigned bank=a->value.bank,off=a->value.off+head,room=a->value.first;
    int j;
    tiny_mac65_axpy_begin();
    for(j=0;j<=p;j++){
        if(!room){++bank;off=RW_LO+head;room=a->value.per;}
        if(scratch.attention_av[j]){
            tiny_mac_a=(const int8_t *)(uintptr_t)off;
            tiny_mac_rows=(uint8_t)bank;tiny_mac_coeff=scratch.attention_av[j];
            tiny_mac65_aux_axpy();
        }
        off+=a->value.stride;--room;
    }
    tiny_mac_acc=scratch.attention_accum;tiny_mac65_axpy_end();
}
#endif
typedef char att_scales_fit[(sizeof(scratch.phase.attn)<=sizeof(scratch.phase.wide_values))?1:-1];
static void attention(Layer *a,int p,const Vec *q,Vec *dst) {
    PackedScale *ks=scratch.phase.attn.key,*vs=scratch.phase.attn.value;
    int h,j,i,m,hi,e,he,shift,top;
    Wide big,s,t,den,scale;
    uint32_t sum,weight;int out_hi=INT_MIN;CacheScale cs;
    for(h=0;h<H;h++){
#ifdef TINY_MOS
        /* The query head is fixed while the cached keys vary, the same shape
         * as a matrix row loop. Prepare it once in the MAC workspace (free
         * between projections) and reuse the matrix kernels for every key. */
        m=q8(q->v+h*HD,tiny_mac65_work,HD);hi=INT_MIN;
        tiny_mac_a=(const int8_t *)rowbuf;tiny_mac_b=tiny_mac65_work;
        tiny_mac_n=(uint16_t)HD;tiny_mac65_prepare();
#else
        m=q8(q->v+h*HD,scratch.attention_aq,HD);hi=INT_MIN;
#endif
#ifdef TINY_DIRECT_MAC
        hi=att_key_scores(a,p,h,q->exp,m);
#else
        for(j=0;j<=p;j++){
            if(!h){cache_scale(&a->key,j,&cs);ks[j].m=cs.m;ks[j].exp=(int16_t)cs.exp;}
            cache_head(&a->key,j,h);
#ifdef TINY_MOS
            tiny_mac65_prepared_dot();
            shift=att_score(&scratch.phase.attention_scores[j],tiny_mac_result,m,ks[j].m,&top);
#else
            shift=att_score(&scratch.phase.attention_scores[j],dot8(scratch.attention_aq,(const int8_t *)rowbuf,HD),m,ks[j].m,&top);
#endif
            scratch.attention_exps[j]=q->exp+ks[j].exp+shift-44;
            if(top>=0&&top+scratch.attention_exps[j]>hi)hi=top+scratch.attention_exps[j];
        }
#endif
        e=hi==INT_MIN?0:hi-45;
        for(j=0;j<=p;j++){align(&scratch.phase.attention_scores[j],&scratch.phase.attention_scores[j],scratch.attention_exps[j],e);if(j==0||wcs(&scratch.phase.attention_scores[j],&big)>0)big=scratch.phase.attention_scores[j];}
        sum=0;hi=INT_MIN;
        for(j=0;j<=p;j++){
            ws(&t,&big,&scratch.phase.attention_scores[j]);weight=exp_neg_q16(delta_q16(&t,e,1,1));sum+=weight;
            if(!h){cache_scale(&a->value,j,&cs);vs[j].m=cs.m;vs[j].exp=(int16_t)cs.exp;}
            cs.m=vs[j].m;cs.exp=vs[j].exp;
            wu(&t,weight);wmu(&scratch.phase.attention_scores[j],&t,cs.m);scratch.attention_exps[j]=cs.exp;update_hi(&scratch.phase.attention_scores[j],scratch.attention_exps[j],&hi);
        }
        e=hi==INT_MIN?0:hi-40;wu(&big,0);
        for(j=0;j<=p;j++){align(&scratch.phase.attention_scores[j],&scratch.phase.attention_scores[j],scratch.attention_exps[j],e);if(wc(&scratch.phase.attention_scores[j],&big)>0)big=scratch.phase.attention_scores[j];}
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
        tiny_attention_coeffs(scratch.phase.attention_scores,&big,scratch.attention_av,(uint8_t)(p+1));
#else
        for(j=0;j<=p;j++){
            if(wz(&big))scratch.attention_av[j]=0;else{wmu(&t,&scratch.phase.attention_scores[j],127);wrounddiv(&t,&t,&big);scratch.attention_av[j]=(int8_t)WLO(t);}
        }
#endif
#ifdef TINY_DIRECT_MAC
        att_value_sums(a,p,h);
#else
        for(i=0;i<HD;i++)scratch.attention_accum[i]=0;
        for(j=0;j<=p;j++)if(scratch.attention_av[j]){cache_head(&a->value,j,h);axpy8(scratch.attention_accum,scratch.attention_av[j],(const int8_t *)rowbuf,HD);}
#endif
        wu(&den,sum);wmu(&den,&den,16129);wl(&t,&big,22);wrounddiv(&scale,&t,&den);he=e-22;
        for(i=0;i<HD;i++){
            wi(&s,scratch.attention_accum[i]);wm(&s,&s,&scale);scratch.projection.attention_result[h*HD+i]=s;scratch.attention_out_exp[h*HD+i]=he;update_hi(&s,he,&out_hi);
        }
    }
    e=out_hi==INT_MIN?0:out_hi-45;
    for(i=0;i<D;i++)align(&wide[i],&scratch.projection.attention_result[i],scratch.attention_out_exp[i],e);
    pack(dst,wide,D,e);
}
/* A NULL output requests KV-cache updates only (non-final prompt token). */
static void step(int tok,int p,Vec *out) {
    int l,i;Layer *a;
    row(&emb,tok,&work_x);row(&pos,p,&work_tmp);residual(&work_x,&work_tmp,D);note(&work_x,D,&peak_residual);
    for(l=0;l<L;l++){
        a=&layers[l];fetch_gain(&a->g1);normalize(&work_x,&gain_work,&work_normed);note(&work_normed,D,&peak_norm);
        prepare_projection(&work_normed,D);
        matvec_prepared(&a->k,&work_tmp);note(&work_tmp,D,&peak_proj);cache_store(&a->key,p,&work_tmp);
        matvec_prepared(&a->v,&work_tmp);note(&work_tmp,D,&peak_proj);cache_store(&a->value,p,&work_tmp);
        /* No later layer consumes the last layer's output in cache-only mode. */
        if(!out && l==L-1)return;
        matvec_prepared(&a->q,&query);note(&query,D,&peak_proj);
        attention(a,p,&query,&att);note(&att,D,&peak_att);
        matvec(&a->o,&att,&work_tmp);note(&work_tmp,D,&peak_proj);residual(&work_x,&work_tmp,D);note(&work_x,D,&peak_residual);
        fetch_gain(&a->g2);normalize(&work_x,&gain_work,&work_normed);note(&work_normed,D,&peak_norm);
        matvec(&a->f1,&work_normed,&work_hidden);note(&work_hidden,HID,&peak_hidden);
        for(i=0;i<HID;i++)if(work_hidden.v[i]<0)work_hidden.v[i]=0;
        matvec(&a->f2,&work_hidden,&work_tmp);note(&work_tmp,D,&peak_proj);residual(&work_x,&work_tmp,D);note(&work_x,D,&peak_residual);
    }
    fetch_gain(&gf);normalize(&work_x,&gain_work,&work_normed);note(&work_normed,D,&peak_norm);
    matvec(&emb,&work_normed,out);note(out,VOCAB,&peak_logit);
}
static int clean(const char *s,char *q){int n=0,u=0,l=0,i;for(;*s&&n<MAXQ;s++)if((unsigned char)*s>=32&&(unsigned char)*s<=126){q[n++]=*s;u+=*s>='A'&&*s<='Z';l+=*s>='a'&&*s<='z';}while(n&&q[n-1]==' ')n--;q[n]=0;if(u&&!l)for(i=0;i<n;i++)if(q[i]>='A'&&q[i]<='Z')q[i]+='a'-'A';return n;}
static int is_word(const char *s,const char *w){while(*s&&*w){if((*s|32)!=*w)return 0;s++;w++;}return !*s&&!*w;}
static int pick(void) {
    int i,best=0;uint32_t sum=0,r;Wide t;
    for(i=1;i<VOCAB;i++)if(score.v[i]>score.v[best])best=i;
    if(!creativity)return best;
    for(i=0;i<VOCAB;i++){
        wu(&t,(uint32_t)((int32_t)score.v[best]-score.v[i]));
        scratch.phase.pick_weights[i]=exp_neg_q16(delta_q16(&t,score.exp,5,(unsigned)creativity));
        if(scratch.phase.pick_weights[i]<U32(2516583))scratch.phase.pick_weights[i]=0;
        sum+=scratch.phase.pick_weights[i];
    }
    wu(&t,sum);wmu(&t,&t,(unsigned)(rand()%10000));wdu(&t,&t,10000);r=WLO(t);
    for(i=0;i<VOCAB;i++){if(r<scratch.phase.pick_weights[i])return i;r-=scratch.phase.pick_weights[i];}
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
#ifdef TINY_MOS
        out_int(p);putchar(',');out_int(t);putchar(',');out_int(score.exp);putchar(',');out_hex(hash,8);putchar('\n');
#else
        printf("%d,%d,%d,%08lX\n",p,t,score.exp,(unsigned long)hash);
#endif
        console_flush();
    }
}
#ifdef TINY_APPLE2
/* ProDOS system programs have no useful argv. Use interactive commands. */
static void apple_answer(const char *q) {
    int p,t,n=(int)strlen(q),letters=0;
    puts("Thinking...");
    for(p=0;p<=n;p++){
        t=p==n?1:(unsigned char)q[p]-30;step(t,p,p==n?&score:NULL);
        putchar('.');console_flush();
    }
    putchar('\n');
    for(;p<CTX;p++){
        t=pick();if(t<=1)break;putchar(t+30);console_flush();letters++;
        step(t,p,&score);
    }
#ifdef TINY_MOS
    out_str("\n[");out_int(letters);puts(" characters]");
#else
    printf("\n[%d characters]\n",letters);
#endif
}
int main(void) {
    static char line[96],q[MAXQ+1];
    unsigned bad;int check=0;
#ifndef TINY_MOS
    int c;
#endif
#if !defined(TINY_PRODOS_HOST_TEST)
    __asm__ volatile ("jsr $fc58" ::: "a", "x", "y", "p", "memory"); // HOME
#endif
    puts("TINY v0.2 / Runtime kernels");
    puts("Apple //e / LLVM-MOS / RamWorks");
#ifdef TINY_MOS
#if !defined(TINY_PRODOS_HOST_TEST)
    reserve_scratch();
#ifdef TINY_DIRECT_MAC
    if((uintptr_t)tiny_mac65_prepare<0x2000u ||
       (uintptr_t)tiny_mac65_aux_end>0x4000u ||
       (uintptr_t)ql<0x2000u ||
       (uintptr_t)tiny_mac65_work+512u>0x4000u)
        die("MAC outside HGR page");
#endif
#endif
#endif
    puts("Banks 1-15 reserved for TINY.");
    bad=tiny_rw_probe();if(bad==255u)die("RAM copy outside HGR page");if(bad){
#ifdef TINY_MOS
        out_str("Bank probe failed: ");out_u32(bad);putchar('\n');die("RamWorks probe");
#else
        printf("Bank probe failed: %u\n",bad);return 1;
#endif
    }
    srand(1);rw_report=1;load_model("TINY.INT");
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM) && !defined(TINY_PRODOS_HOST_TEST)
    {
        void *code=tiny_claim_rescale_code();
        if(!code)die("Rescale workspace busy");
        tiny_rescale_init(code);
    }
#endif
#ifdef TINY_AUX0_LUT
    aux_lut_begin();
    if(exp_neg_q16(0)!=EXP_ONE || exp_neg_q16(1024)!=U32(16517109) ||
       exp_neg_q16(512)!=U32(16647162) || exp_neg_q16(EXP_LIMIT-1)!=0)
        die("Aux exp self-test");
    puts("Aux0 exp active; quit restores aux0.");
#endif
#ifdef TINY_MOS
    /* The start of the exp table is in auxiliary memory now; its main-memory
     * bytes become the MAC workspace and the biased table plane. */
    tiny_mac65_init();
#endif
    rw_stats("load");rw_reset_stats();
    puts(":check toggles hash trace; :banks stats; quit exits.");
    for(;;){
#ifdef TINY_MOS
        out_str("> ");if(!read_line(line,sizeof(line)))break;
#else
        fputs("> ",stdout);console_flush();
        if(!fgets(line,sizeof(line),stdin))break;
        if(!strchr(line,'\n'))while((c=getchar())!='\n'&&c!=EOF){}
#endif
        if(!clean(line,q))continue;
        if(is_word(q,"quit")||is_word(q,"exit"))break;
        if(is_word(q,":check")){check=!check;puts(check?"Trace on":"Trace off");continue;}
        if(is_word(q,":banks")){rw_stats("inference");continue;}
        if(q[0]==':'&&q[1]=='c'&&q[2]>='0'&&q[2]<='9'&&!q[3]){creativity=q[2]-'0';continue;}
        if(check)check_question(q);else apple_answer(q);
    }
#ifdef TINY_AUX0_LUT
    aux_lut_restore();
#endif
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM) && !defined(TINY_PRODOS_HOST_TEST)
    tiny_release_rescale_code();
#endif
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
        t=p==n?1:(unsigned char)q[p]-30;step(t,p,(p==n||trace||profiling)?&score:NULL);
        if(trace){printf("%d,%d",p,t);for(i=0;i<VOCAB;i++){putchar(',');print_number(stdout,score.v[i],score.exp);}putchar('\n');}
    }
    if(trace)return;
    for(;p<CTX;p++){t=pick();if(t<=1)break;putchar(t+30);fflush(stdout);letters++;step(t,p,&score);}
    putchar('\n');end=clock();
    if(start==(clock_t)-1||end==(clock_t)-1||end<start)fprintf(stderr,"[%d characters]\n",letters);
    else{ticks=(uint32_t)end-(uint32_t)start;sec=ticks/(uint32_t)CLOCKS_PER_SEC;
        wu(&wt,ticks%(uint32_t)CLOCKS_PER_SEC);wmu(&wt,&wt,10);wdu(&wt,&wt,(uint32_t)CLOCKS_PER_SEC);frac=WLO(wt);
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
    srand(seed);load_model(model);rw_stats("load");rw_reset_stats();fprintf(stderr,"TINY Stage 10B: %d layers, width %d, %d heads, hidden %d, creativity %d\n",L,D,H,HID,creativity);
    if(question){if(trace==2)check_question(q);else answer(q,trace);}else{while(fputs("> ",stderr),fflush(stderr),fgets(line,sizeof(line),stdin)){if(!strchr(line,'\n'))while((c=getchar())!='\n'&&c!=EOF){}if(!clean(line,q))continue;if(is_word(q,"quit")||is_word(q,"exit"))break;answer(q,0);}}
    profile();rw_stats("inference");return ferror(stdout)||ferror(stdin)?1:0;
}

#endif /* TINY_APPLE2 */
