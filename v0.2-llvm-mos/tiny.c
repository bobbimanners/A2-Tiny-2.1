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
/* Initialized as the compact LUT in the SYS file. After the model is
 * loaded, move the table to aux bank zero, then reuse its main RAM. */
/* On the native build the first AUX_LUT_HEAD bytes of the table live in
 * tiny-qsquare-mos.s, as the initial contents of memory the MAC kernels
 * take over once the table is in auxiliary memory. That keeps the load
 * image under the ProDOS SYS limit. */
#if !defined(TINY_PRODOS_HOST_TEST)
#define TINY_LUT_SPLIT
#define AUX_LUT_HEAD 1056u
extern const uint8_t tiny_exp_lut_head[AUX_LUT_HEAD];
#else
#define AUX_LUT_HEAD 0u
#endif
typedef union {
    uint8_t lut[(AUX_LUT_SIZE-AUX_LUT_HEAD)/3u][3];
    struct { Vec vx,vnorm,vtmp,vhidden; } vec;
} AuxOverlay;
static AuxOverlay overlay;
typedef char overlay_fits[(sizeof(overlay.vec)<=AUX_LUT_SIZE)?1:-1];
typedef char lut_split_exact[((AUX_LUT_SIZE-AUX_LUT_HEAD)%3u==0)?1:-1];
static uint8_t aux_lut_active;
static void aux_lut_restore(void);
#define work_x (overlay.vec.vx)
#define work_normed (overlay.vec.vnorm)
#define work_tmp (overlay.vec.vtmp)
#define work_hidden (overlay.vec.vhidden)
#else
static Vec work_x,work_normed,work_tmp,work_hidden;
#endif
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
static void load_table(ModelFile *f,Table *t,int rows,int cols) {
    int r;int32_t c;t->rows=rows;t->cols=cols;t->exp=read_exp(f);
    rw_alloc(&t->data,(unsigned)rows,(unsigned)cols+4u);
    for(r=0;r<rows;r++){
        c=read_coeff(f);if(c<0)die("negative row scale");
        put32(rowbuf,(uint32_t)c);read_exact(f,rowbuf+4,cols);
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
#if !defined(TINY_COPY_MAC)
#define TINY_DIRECT_MAC
#endif
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
static void matvec(const Table *t,const Vec *src,Vec *dst) {
#ifdef TINY_DIRECT_MAC
    int8_t *q=tiny_mac65_work;
#else
    int8_t *q=scratch.projection.matvec_q;
#endif
    /* Absorb the matrix rescaling divisor (128) into the exponent. */
    int r,m=q8(src->v,q,t->cols);int e=src->exp+t->exp-7;
    unsigned bank=t->data.bank,off=t->data.off,n=t->data.stride;
    unsigned last=RW_HI-n;
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
    TinyRescaleArgs args;
    args.peak=(uint16_t)m;
#else
    Wide a;
#endif
#ifdef TINY_MOS
    /* q8 emits [-127,127], so the biased offsets are both nonzero.
     * Prepare once; rowbuf and the vector length stay fixed across rows. */
    tiny_mac_a=(const int8_t *)(rowbuf+4);
    tiny_mac_b=q;
    tiny_mac_n=(uint16_t)t->cols;
    tiny_mac65_prepare();
#ifdef TINY_DIRECT_MAC
    tiny_mac_b=(const int8_t *)rowbuf; /* four scale bytes only */
#endif
#endif
    /* rw_alloc packs complete records consecutively, skipping only the
     * unusable tail of a bank. Walk that layout without mul/div per row. */
#ifdef TINY_DIRECT_MAC
    /* Up to eight rows per kernel call; a batch never spans two banks. */
    unsigned room=t->data.first;
    (void)last;
    tiny_mac_stride=(uint16_t)n;
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
        wi(&a,dot8((const int8_t *)(rowbuf+4),q,t->cols));
#endif
        wmi(&a,&a,signed32(get32(rowbuf)));wmu(&wide[r],&a,(uint32_t)m);
#endif
    }
#endif
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
#ifdef TINY_COMPACT_LUT
#ifdef TINY_AUX0_LUT
static AuxOverlay overlay = { .lut = {
#else
static const uint8_t exp_lut_compact[1152][3] = {
#endif
#ifndef TINY_LUT_SPLIT
    {245,7,252}, {171,31,248}, {227,70,244}, {96,125,240}, {228,194,236}, {53,23,233}, {23,122,229}, {81,235,225},
    {170,106,222}, {233,247,218}, {216,146,215}, {65,59,212}, {237,240,208}, {168,179,205}, {62,131,202}, {125,95,199},
    {50,72,196}, {43,61,193}, {56,62,190}, {41,75,187}, {207,99,184}, {252,135,181}, {128,183,178}, {49,242,175},
    {224,55,173}, {99,136,170}, {143,227,167}, {57,73,165}, {55,185,162}, {97,51,160}, {143,183,157}, {152,69,155},
    {85,221,152}, {160,126,150}, {84,41,148}, {74,221,145}, {93,154,143}, {107,96,141}, {78,47,139}, {229,6,137},
    {12,231,134}, {161,207,132}, {131,192,130}, {146,185,128}, {172,186,126}, {178,195,124}, {132,212,122}, {4,237,120},
    {18,13,119}, {145,52,117}, {100,99,115}, {108,153,113}, {143,214,111}, {175,26,110}, {176,101,108}, {120,183,106},
    {236,15,105}, {240,110,103}, {107,212,101}, {68,64,100}, {97,178,98}, {169,42,97}, {4,169,95}, {89,45,94},
    {145,183,92}, {149,71,91}, {77,221,89}, {163,120,88}, {128,25,87}, {207,191,85}, {123,107,84}, {109,28,83},
    {144,210,81}, {209,141,80}, {27,78,79}, {90,19,78}, {122,221,76}, {104,172,75}, {16,128,74}, {97,88,73},
    {71,53,72}, {177,22,71}, {140,252,69}, {198,230,68}, {80,213,67}, {22,200,66}, {9,191,65}, {25,186,64},
    {51,185,63}, {74,188,62}, {76,195,61}, {42,206,60}, {213,220,59}, {62,239,58}, {86,5,58}, {15,31,57},
    {89,60,56}, {39,93,55}, {107,129,54}, {24,169,53}, {30,212,52}, {114,2,52}, {6,52,51}, {206,104,50},
    {188,160,49}, {196,219,48}, {217,25,48}, {241,90,47}, {254,158,46}, {245,229,45}, {203,47,45}, {115,124,44},
    {227,203,43}, {16,30,43}, {239,114,42}, {117,202,41}, {152,36,41}, {77,129,40}, {138,224,39}, {69,66,39},
    {117,166,38}, {15,13,38}, {9,118,37}, {91,225,36}, {251,78,36}, {224,190,35}, {1,49,35}, {85,165,34},
    {212,27,34}, {116,148,33}, {45,15,33}, {248,139,32}, {203,10,32}, {159,139,31}, {107,14,31}, {41,147,30},
    {207,25,30}, {88,162,29}, {186,44,29}, {239,184,28}, {240,70,28}, {182,214,27}, {56,104,27}, {114,251,26},
    {91,144,26}, {237,38,26}, {33,191,25}, {241,88,25}, {87,244,24}, {77,145,24}, {203,47,24}, {204,207,23},
    {74,113,23}, {64,20,23}, {167,184,22}, {121,94,22}, {177,5,22}, {73,174,21}, {61,88,21}, {134,3,21},
    {31,176,20}, {3,94,20}, {45,13,20}, {152,189,19}, {63,111,19}, {28,34,19}, {44,214,18}, {105,139,18},
    {207,65,18}, {89,249,17}, {3,178,17}, {200,107,17}, {163,38,17}, {145,226,16}, {141,159,16}, {148,93,16},
    {159,28,16}, {173,220,15}, {185,157,15}, {190,95,15}, {186,34,15}, {167,230,14}, {131,171,14}, {74,113,14},
    {248,55,14}, {137,255,13}, {250,199,13}, {72,145,13}, {111,91,13}, {107,38,13}, {58,242,12}, {216,190,12},
    {66,140,12}, {117,90,12}, {110,41,12}, {41,249,11}, {163,201,11}, {219,154,11}, {204,108,11}, {115,63,11},
    {207,18,11}, {220,230,10}, {151,187,10}, {254,144,10}, {15,103,10}, {197,61,10}, {32,21,10}, {27,237,9},
    {182,197,9}, {237,158,9}, {190,120,9}, {38,83,9}, {36,46,9}, {180,9,9}, {213,229,8}, {133,194,8},
    {192,159,8}, {134,125,8}, {212,91,8}, {167,58,8}, {254,25,8}, {214,249,7}, {46,218,7}, {4,187,7},
    {85,156,7}, {33,126,7}, {100,96,7}, {29,67,7}, {74,38,7}, {234,9,7}, {250,237,6}, {122,210,6},
    {102,183,6}, {190,156,6}, {128,130,6}, {169,104,6}, {58,79,6}, {47,54,6}, {136,29,6}, {66,5,6},
    {93,237,5}, {215,213,5}, {174,190,5}, {225,167,5}, {110,145,5}, {85,123,5}, {147,101,5}, {40,80,5},
    {18,59,5}, {79,38,5}, {223,17,5}, {192,253,4}, {240,233,4}, {112,214,4}, {60,195,4}, {85,176,4},
    {185,157,4}, {103,139,4}, {94,121,4}, {156,103,4}, {33,86,4}, {235,68,4}, {249,51,4}, {74,35,4},
    {222,18,4}, {179,2,4}, {200,242,3}, {29,227,3}, {175,211,3}, {127,196,3}, {139,181,3}, {210,166,3},
    {84,152,3}, {15,138,3}, {3,124,3}, {47,110,3}, {145,96,3}, {42,83,3}, {248,69,3}, {250,56,3},
    {48,44,3}, {152,31,3}, {51,19,3}, {254,6,3}, {251,250,2}, {38,239,2}, {129,227,2}, {10,216,2},
    {193,204,2}, {164,193,2}, {179,182,2}, {238,171,2}, {84,161,2}, {227,150,2}, {156,140,2}, {126,130,2},
    {136,120,2}, {186,110,2}, {18,101,2}, {145,91,2}, {53,82,2}, {255,72,2}, {237,63,2}, {255,54,2},
    {53,46,2}, {142,37,2}, {8,29,2}, {165,20,2}, {99,12,2}, {66,4,2}, {65,252,1}, {96,244,1},
    {158,236,1}, {251,228,1}, {118,221,1}, {15,214,1}, {197,206,1}, {152,199,1}, {136,192,1}, {148,185,1},
    {187,178,1}, {254,171,1}, {91,165,1}, {211,158,1}, {101,152,1}, {16,146,1}, {212,139,1}, {177,133,1},
    {166,127,1}, {180,121,1}, {217,115,1}, {21,110,1}, {104,104,1}, {209,98,1}, {81,93,1}, {231,87,1},
    {146,82,1}, {82,77,1}, {39,72,1}, {17,67,1}, {15,62,1}, {32,57,1}, {69,52,1}, {126,47,1},
    {201,42,1}, {40,38,1}, {152,33,1}, {27,29,1}, {175,24,1}, {85,20,1}, {12,16,1}, {213,11,1},
#endif
    {174,7,1}, {151,3,1}, {145,255,0}, {155,251,0}, {180,247,0}, {221,243,0}, {21,240,0}, {92,236,0},
    {178,232,0}, {22,229,0}, {137,225,0}, {10,222,0}, {153,218,0}, {53,215,0}, {223,211,0}, {150,208,0},
    {90,205,0}, {43,202,0}, {9,199,0}, {243,195,0}, {233,192,0}, {236,189,0}, {250,186,0}, {20,184,0},
    {57,181,0}, {106,178,0}, {166,175,0}, {237,172,0}, {62,170,0}, {155,167,0}, {1,165,0}, {115,162,0},
    {238,159,0}, {115,157,0}, {2,155,0}, {155,152,0}, {61,150,0}, {233,147,0}, {158,145,0}, {92,143,0},
    {35,141,0}, {243,138,0}, {203,136,0}, {172,134,0}, {150,132,0}, {136,130,0}, {130,128,0}, {132,126,0},
    {142,124,0}, {159,122,0}, {185,120,0}, {217,118,0}, {2,117,0}, {49,115,0}, {104,113,0}, {166,111,0},
    {235,109,0}, {55,108,0}, {137,106,0}, {226,104,0}, {66,103,0}, {168,101,0}, {21,100,0}, {136,98,0},
    {0,97,0}, {127,95,0}, {4,94,0}, {143,92,0}, {32,91,0}, {182,89,0}, {82,88,0}, {244,86,0},
    {155,85,0}, {71,84,0}, {248,82,0}, {175,81,0}, {107,80,0}, {44,79,0}, {241,77,0}, {188,76,0},
    {140,75,0}, {96,74,0}, {57,73,0}, {22,72,0}, {248,70,0}, {222,69,0}, {201,68,0}, {184,67,0},
    {171,66,0}, {162,65,0}, {158,64,0}, {158,63,0}, {161,62,0}, {168,61,0}, {180,60,0}, {195,59,0},
    {214,58,0}, {236,57,0}, {6,57,0}, {36,56,0}, {69,55,0}, {106,54,0}, {146,53,0}, {189,52,0},
    {236,51,0}, {30,51,0}, {83,50,0}, {139,49,0}, {199,48,0}, {5,48,0}, {70,47,0}, {139,46,0},
    {210,45,0}, {28,45,0}, {105,44,0}, {185,43,0}, {11,43,0}, {97,42,0}, {184,41,0}, {19,41,0},
    {112,40,0}, {207,39,0}, {49,39,0}, {150,38,0}, {253,37,0}, {102,37,0}, {209,36,0}, {63,36,0},
    {175,35,0}, {34,35,0}, {150,34,0}, {13,34,0}, {134,33,0}, {1,33,0}, {126,32,0}, {253,31,0},
    {126,31,0}, {1,31,0}, {134,30,0}, {13,30,0}, {149,29,0}, {32,29,0}, {172,28,0}, {59,28,0},
    {203,27,0}, {92,27,0}, {240,26,0}, {133,26,0}, {28,26,0}, {180,25,0}, {78,25,0}, {234,24,0},
    {135,24,0}, {37,24,0}, {197,23,0}, {103,23,0}, {10,23,0}, {175,22,0}, {85,22,0}, {252,21,0},
    {165,21,0}, {79,21,0}, {250,20,0}, {167,20,0}, {85,20,0}, {4,20,0}, {181,19,0}, {103,19,0},
    {26,19,0}, {206,18,0}, {131,18,0}, {58,18,0}, {242,17,0}, {170,17,0}, {100,17,0}, {31,17,0},
    {219,16,0}, {152,16,0}, {86,16,0}, {22,16,0}, {214,15,0}, {151,15,0}, {89,15,0}, {28,15,0},
    {224,14,0}, {165,14,0}, {107,14,0}, {50,14,0}, {249,13,0}, {194,13,0}, {139,13,0}, {86,13,0},
    {33,13,0}, {237,12,0}, {185,12,0}, {135,12,0}, {85,12,0}, {36,12,0}, {244,11,0}, {197,11,0},
    {150,11,0}, {104,11,0}, {59,11,0}, {14,11,0}, {226,10,0}, {183,10,0}, {140,10,0}, {99,10,0},
    {57,10,0}, {17,10,0}, {233,9,0}, {193,9,0}, {155,9,0}, {117,9,0}, {79,9,0}, {42,9,0},
    {6,9,0}, {226,8,0}, {191,8,0}, {156,8,0}, {122,8,0}, {88,8,0}, {55,8,0}, {22,8,0},
    {246,7,0}, {215,7,0}, {184,7,0}, {153,7,0}, {123,7,0}, {93,7,0}, {64,7,0}, {35,7,0},
    {7,7,0}, {235,6,0}, {208,6,0}, {180,6,0}, {154,6,0}, {128,6,0}, {102,6,0}, {76,6,0},
    {51,6,0}, {27,6,0}, {3,6,0}, {235,5,0}, {211,5,0}, {188,5,0}, {165,5,0}, {143,5,0},
    {121,5,0}, {99,5,0}, {78,5,0}, {57,5,0}, {36,5,0}, {16,5,0}, {252,4,0}, {232,4,0},
    {212,4,0}, {193,4,0}, {174,4,0}, {156,4,0}, {137,4,0}, {119,4,0}, {102,4,0}, {84,4,0},
    {67,4,0}, {50,4,0}, {33,4,0}, {17,4,0}, {1,4,0}, {241,3,0}, {225,3,0}, {210,3,0},
    {195,3,0}, {180,3,0}, {165,3,0}, {151,3,0}, {137,3,0}, {122,3,0}, {109,3,0}, {95,3,0},
    {82,3,0}, {69,3,0}, {56,3,0}, {43,3,0}, {30,3,0}, {18,3,0}, {6,3,0}, {250,2,0},
    {238,2,0}, {226,2,0}, {215,2,0}, {204,2,0}, {192,2,0}, {182,2,0}, {171,2,0}, {160,2,0},
    {150,2,0}, {140,2,0}, {129,2,0}, {119,2,0}, {110,2,0}, {100,2,0}, {91,2,0}, {81,2,0},
    {72,2,0}, {63,2,0}, {54,2,0}, {45,2,0}, {37,2,0}, {28,2,0}, {20,2,0}, {11,2,0},
    {3,2,0}, {251,1,0}, {244,1,0}, {236,1,0}, {228,1,0}, {221,1,0}, {213,1,0}, {206,1,0},
    {199,1,0}, {192,1,0}, {185,1,0}, {178,1,0}, {171,1,0}, {165,1,0}, {158,1,0}, {152,1,0},
    {145,1,0}, {139,1,0}, {133,1,0}, {127,1,0}, {121,1,0}, {115,1,0}, {109,1,0}, {104,1,0},
    {98,1,0}, {93,1,0}, {87,1,0}, {82,1,0}, {77,1,0}, {72,1,0}, {67,1,0}, {62,1,0},
    {57,1,0}, {52,1,0}, {47,1,0}, {42,1,0}, {38,1,0}, {33,1,0}, {29,1,0}, {24,1,0},
    {20,1,0}, {16,1,0}, {11,1,0}, {7,1,0}, {3,1,0}, {255,0,0}, {251,0,0}, {247,0,0},
    {243,0,0}, {240,0,0}, {236,0,0}, {232,0,0}, {229,0,0}, {225,0,0}, {222,0,0}, {218,0,0},
    {215,0,0}, {212,0,0}, {208,0,0}, {205,0,0}, {202,0,0}, {199,0,0}, {196,0,0}, {193,0,0},
    {190,0,0}, {187,0,0}, {184,0,0}, {181,0,0}, {178,0,0}, {175,0,0}, {173,0,0}, {170,0,0},
    {167,0,0}, {165,0,0}, {162,0,0}, {160,0,0}, {157,0,0}, {155,0,0}, {152,0,0}, {150,0,0},
    {148,0,0}, {145,0,0}, {143,0,0}, {141,0,0}, {139,0,0}, {137,0,0}, {134,0,0}, {132,0,0},
    {130,0,0}, {128,0,0}, {126,0,0}, {124,0,0}, {122,0,0}, {121,0,0}, {119,0,0}, {117,0,0},
    {115,0,0}, {113,0,0}, {111,0,0}, {110,0,0}, {108,0,0}, {106,0,0}, {105,0,0}, {103,0,0},
    {101,0,0}, {100,0,0}, {98,0,0}, {97,0,0}, {95,0,0}, {94,0,0}, {92,0,0}, {91,0,0},
    {90,0,0}, {88,0,0}, {87,0,0}, {85,0,0}, {84,0,0}, {83,0,0}, {82,0,0}, {80,0,0},
    {79,0,0}, {78,0,0}, {77,0,0}, {75,0,0}, {74,0,0}, {73,0,0}, {72,0,0}, {71,0,0},
    {70,0,0}, {69,0,0}, {68,0,0}, {67,0,0}, {66,0,0}, {65,0,0}, {64,0,0}, {63,0,0},
    {62,0,0}, {61,0,0}, {60,0,0}, {59,0,0}, {58,0,0}, {57,0,0}, {56,0,0}, {55,0,0},
    {54,0,0}, {53,0,0}, {53,0,0}, {52,0,0}, {51,0,0}, {50,0,0}, {49,0,0}, {49,0,0},
    {48,0,0}, {47,0,0}, {46,0,0}, {46,0,0}, {45,0,0}, {44,0,0}, {44,0,0}, {43,0,0},
    {42,0,0}, {42,0,0}, {41,0,0}, {40,0,0}, {40,0,0}, {39,0,0}, {39,0,0}, {38,0,0},
    {37,0,0}, {37,0,0}, {36,0,0}, {36,0,0}, {35,0,0}, {35,0,0}, {34,0,0}, {33,0,0},
    {33,0,0}, {32,0,0}, {32,0,0}, {31,0,0}, {31,0,0}, {30,0,0}, {30,0,0}, {30,0,0},
    {29,0,0}, {29,0,0}, {28,0,0}, {28,0,0}, {27,0,0}, {27,0,0}, {26,0,0}, {26,0,0},
    {26,0,0}, {25,0,0}, {25,0,0}, {24,0,0}, {24,0,0}, {24,0,0}, {23,0,0}, {23,0,0},
    {23,0,0}, {22,0,0}, {22,0,0}, {22,0,0}, {21,0,0}, {21,0,0}, {21,0,0}, {20,0,0},
    {20,0,0}, {20,0,0}, {19,0,0}, {19,0,0}, {19,0,0}, {18,0,0}, {18,0,0}, {18,0,0},
    {18,0,0}, {17,0,0}, {17,0,0}, {17,0,0}, {17,0,0}, {16,0,0}, {16,0,0}, {16,0,0},
    {16,0,0}, {15,0,0}, {15,0,0}, {15,0,0}, {15,0,0}, {14,0,0}, {14,0,0}, {14,0,0},
    {14,0,0}, {14,0,0}, {13,0,0}, {13,0,0}, {13,0,0}, {13,0,0}, {13,0,0}, {12,0,0},
    {12,0,0}, {12,0,0}, {12,0,0}, {12,0,0}, {11,0,0}, {11,0,0}, {11,0,0}, {11,0,0},
    {11,0,0}, {11,0,0}, {10,0,0}, {10,0,0}, {10,0,0}, {10,0,0}, {10,0,0}, {10,0,0},
    {9,0,0}, {9,0,0}, {9,0,0}, {9,0,0}, {9,0,0}, {9,0,0}, {9,0,0}, {8,0,0},
    {8,0,0}, {8,0,0}, {8,0,0}, {8,0,0}, {8,0,0}, {8,0,0}, {8,0,0}, {7,0,0},
    {7,0,0}, {7,0,0}, {7,0,0}, {7,0,0}, {7,0,0}, {7,0,0}, {7,0,0}, {7,0,0},
    {6,0,0}, {6,0,0}, {6,0,0}, {6,0,0}, {6,0,0}, {6,0,0}, {6,0,0}, {6,0,0},
    {6,0,0}, {6,0,0}, {6,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0},
    {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0}, {5,0,0},
    {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0},
    {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0}, {4,0,0},
    {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0},
    {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0},
    {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {3,0,0}, {2,0,0}, {2,0,0},
    {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0},
    {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0},
    {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0},
    {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {2,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0},
    {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
#ifdef TINY_AUX0_LUT
}};
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
    tiny_rw_write(0,AUX_LUT_ADDR+AUX_LUT_HEAD,overlay.lut,AUX_LUT_SIZE-AUX_LUT_HEAD);
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
#else
};
#endif
static uint32_t exp_entry_bytes(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16);
}
#else
static const uint32_t exp_lut[1153] = {
    16777216UL, 16517109UL, 16261035UL, 16008931UL, 15760736UL, 15516388UL, 15275829UL, 15038999UL,
    14805841UL, 14576298UL, 14350313UL, 14127832UL, 13908801UL, 13693165UL, 13480872UL, 13271870UL,
    13066109UL, 12863538UL, 12664107UL, 12467768UL, 12274473UL, 12084175UL, 11896828UL, 11712384UL,
    11530801UL, 11352032UL, 11176035UL, 11002767UL, 10832185UL, 10664247UL, 10498913UL, 10336143UL,
    10175896UL, 10018133UL, 9862816UL, 9709908UL, 9559370UL, 9411165UL, 9265259UL, 9121614UL,
    8980197UL, 8840972UL, 8703905UL, 8568963UL, 8436114UL, 8305324UL, 8176562UL, 8049796UL,
    7924996UL, 7802130UL, 7681169UL, 7562084UL, 7444844UL, 7329423UL, 7215791UL, 7103920UL,
    6993784UL, 6885356UL, 6778608UL, 6673515UL, 6570052UL, 6468193UL, 6367913UL, 6269188UL,
    6171993UL, 6076305UL, 5982101UL, 5889357UL, 5798051UL, 5708160UL, 5619663UL, 5532539UL,
    5446765UL, 5362320UL, 5279185UL, 5197339UL, 5116762UL, 5037434UL, 4959336UL, 4882448UL,
    4806753UL, 4732231UL, 4658865UL, 4586636UL, 4515526UL, 4445520UL, 4376598UL, 4308745UL,
    4241945UL, 4176179UL, 4111434UL, 4047692UL, 3984938UL, 3923157UL, 3862334UL, 3802454UL,
    3743503UL, 3685465UL, 3628327UL, 3572075UL, 3516696UL, 3462174UL, 3408498UL, 3355654UL,
    3303630UL, 3252412UL, 3201988UL, 3152345UL, 3103473UL, 3055358UL, 3007989UL, 2961355UL,
    2915443UL, 2870243UL, 2825744UL, 2781935UL, 2738805UL, 2696344UL, 2654541UL, 2613386UL,
    2572869UL, 2532981UL, 2493711UL, 2455049UL, 2416987UL, 2379515UL, 2342624UL, 2306305UL,
    2270549UL, 2235348UL, 2200692UL, 2166573UL, 2132984UL, 2099915UL, 2067359UL, 2035307UL,
    2003753UL, 1972687UL, 1942104UL, 1911994UL, 1882351UL, 1853168UL, 1824438UL, 1796152UL,
    1768306UL, 1740891UL, 1713901UL, 1687329UL, 1661169UL, 1635415UL, 1610061UL, 1585099UL,
    1560524UL, 1536330UL, 1512512UL, 1489063UL, 1465977UL, 1443249UL, 1420873UL, 1398845UL,
    1377158UL, 1355807UL, 1334787UL, 1314093UL, 1293720UL, 1273663UL, 1253916UL, 1234476UL,
    1215337UL, 1196495UL, 1177945UL, 1159683UL, 1141704UL, 1124003UL, 1106577UL, 1089421UL,
    1072532UL, 1055903UL, 1039533UL, 1023417UL, 1007550UL, 991930UL, 976551UL, 961411UL,
    946506UL, 931832UL, 917385UL, 903162UL, 889160UL, 875375UL, 861803UL, 848442UL,
    835288UL, 822338UL, 809589UL, 797038UL, 784681UL, 772515UL, 760539UL, 748748UL,
    737139UL, 725711UL, 714460UL, 703383UL, 692478UL, 681743UL, 671173UL, 660768UL,
    650523UL, 640438UL, 630509UL, 620734UL, 611110UL, 601636UL, 592308UL, 583125UL,
    574085UL, 565184UL, 556422UL, 547796UL, 539303UL, 530942UL, 522710UL, 514606UL,
    506628UL, 498773UL, 491041UL, 483428UL, 475933UL, 468554UL, 461290UL, 454138UL,
    447098UL, 440166UL, 433342UL, 426624UL, 420009UL, 413498UL, 407087UL, 400776UL,
    394562UL, 388445UL, 382423UL, 376494UL, 370657UL, 364910UL, 359253UL, 353683UL,
    348200UL, 342802UL, 337487UL, 332255UL, 327104UL, 322032UL, 317040UL, 312124UL,
    307285UL, 302521UL, 297831UL, 293214UL, 288668UL, 284193UL, 279787UL, 275449UL,
    271178UL, 266974UL, 262835UL, 258760UL, 254749UL, 250799UL, 246911UL, 243083UL,
    239314UL, 235604UL, 231951UL, 228355UL, 224815UL, 221329UL, 217898UL, 214520UL,
    211194UL, 207920UL, 204696UL, 201523UL, 198398UL, 195323UL, 192294UL, 189313UL,
    186378UL, 183489UL, 180644UL, 177843UL, 175086UL, 172372UL, 169699UL, 167068UL,
    164478UL, 161928UL, 159418UL, 156946UL, 154513UL, 152117UL, 149759UL, 147437UL,
    145151UL, 142901UL, 140686UL, 138504UL, 136357UL, 134243UL, 132162UL, 130113UL,
    128096UL, 126110UL, 124155UL, 122230UL, 120335UL, 118469UL, 116632UL, 114824UL,
    113044UL, 111291UL, 109566UL, 107867UL, 106195UL, 104549UL, 102928UL, 101332UL,
    99761UL, 98214UL, 96692UL, 95193UL, 93717UL, 92264UL, 90833UL, 89425UL,
    88039UL, 86674UL, 85330UL, 84007UL, 82705UL, 81423UL, 80160UL, 78917UL,
    77694UL, 76489UL, 75304UL, 74136UL, 72987UL, 71855UL, 70741UL, 69644UL,
    68565UL, 67502UL, 66455UL, 65425UL, 64411UL, 63412UL, 62429UL, 61461UL,
    60508UL, 59570UL, 58646UL, 57737UL, 56842UL, 55961UL, 55093UL, 54239UL,
    53398UL, 52570UL, 51755UL, 50953UL, 50163UL, 49385UL, 48620UL, 47866UL,
    47124UL, 46393UL, 45674UL, 44966UL, 44269UL, 43582UL, 42907UL, 42241UL,
    41587UL, 40942UL, 40307UL, 39682UL, 39067UL, 38461UL, 37865UL, 37278UL,
    36700UL, 36131UL, 35571UL, 35019UL, 34476UL, 33942UL, 33416UL, 32898UL,
    32388UL, 31886UL, 31391UL, 30905UL, 30425UL, 29954UL, 29489UL, 29032UL,
    28582UL, 28139UL, 27703UL, 27273UL, 26850UL, 26434UL, 26024UL, 25621UL,
    25224UL, 24832UL, 24447UL, 24068UL, 23695UL, 23328UL, 22966UL, 22610UL,
    22260UL, 21915UL, 21575UL, 21240UL, 20911UL, 20587UL, 20268UL, 19953UL,
    19644UL, 19340UL, 19040UL, 18745UL, 18454UL, 18168UL, 17886UL, 17609UL,
    17336UL, 17067UL, 16802UL, 16542UL, 16286UL, 16033UL, 15784UL, 15540UL,
    15299UL, 15062UL, 14828UL, 14598UL, 14372UL, 14149UL, 13930UL, 13714UL,
    13501UL, 13292UL, 13086UL, 12883UL, 12683UL, 12487UL, 12293UL, 12102UL,
    11915UL, 11730UL, 11548UL, 11369UL, 11193UL, 11019UL, 10849UL, 10680UL,
    10515UL, 10352UL, 10191UL, 10033UL, 9878UL, 9725UL, 9574UL, 9425UL,
    9279UL, 9135UL, 8994UL, 8854UL, 8717UL, 8582UL, 8449UL, 8318UL,
    8189UL, 8062UL, 7937UL, 7814UL, 7693UL, 7573UL, 7456UL, 7340UL,
    7227UL, 7115UL, 7004UL, 6896UL, 6789UL, 6684UL, 6580UL, 6478UL,
    6378UL, 6279UL, 6181UL, 6085UL, 5991UL, 5898UL, 5807UL, 5717UL,
    5628UL, 5541UL, 5455UL, 5370UL, 5287UL, 5205UL, 5124UL, 5045UL,
    4967UL, 4890UL, 4814UL, 4739UL, 4666UL, 4594UL, 4522UL, 4452UL,
    4383UL, 4315UL, 4248UL, 4182UL, 4118UL, 4054UL, 3991UL, 3929UL,
    3868UL, 3808UL, 3749UL, 3691UL, 3634UL, 3577UL, 3522UL, 3467UL,
    3414UL, 3361UL, 3309UL, 3257UL, 3207UL, 3157UL, 3108UL, 3060UL,
    3013UL, 2966UL, 2920UL, 2875UL, 2830UL, 2786UL, 2743UL, 2700UL,
    2659UL, 2617UL, 2577UL, 2537UL, 2497UL, 2459UL, 2421UL, 2383UL,
    2346UL, 2310UL, 2274UL, 2239UL, 2204UL, 2170UL, 2136UL, 2103UL,
    2070UL, 2038UL, 2007UL, 1976UL, 1945UL, 1915UL, 1885UL, 1856UL,
    1827UL, 1799UL, 1771UL, 1744UL, 1716UL, 1690UL, 1664UL, 1638UL,
    1612UL, 1587UL, 1563UL, 1539UL, 1515UL, 1491UL, 1468UL, 1445UL,
    1423UL, 1401UL, 1379UL, 1358UL, 1337UL, 1316UL, 1296UL, 1276UL,
    1256UL, 1236UL, 1217UL, 1198UL, 1180UL, 1161UL, 1143UL, 1126UL,
    1108UL, 1091UL, 1074UL, 1057UL, 1041UL, 1025UL, 1009UL, 993UL,
    978UL, 963UL, 948UL, 933UL, 919UL, 905UL, 890UL, 877UL,
    863UL, 850UL, 837UL, 824UL, 811UL, 798UL, 786UL, 774UL,
    762UL, 750UL, 738UL, 727UL, 716UL, 704UL, 694UL, 683UL,
    672UL, 662UL, 652UL, 641UL, 631UL, 622UL, 612UL, 603UL,
    593UL, 584UL, 575UL, 566UL, 557UL, 549UL, 540UL, 532UL,
    523UL, 515UL, 507UL, 500UL, 492UL, 484UL, 477UL, 469UL,
    462UL, 455UL, 448UL, 441UL, 434UL, 427UL, 421UL, 414UL,
    408UL, 401UL, 395UL, 389UL, 383UL, 377UL, 371UL, 365UL,
    360UL, 354UL, 349UL, 343UL, 338UL, 333UL, 328UL, 323UL,
    318UL, 313UL, 308UL, 303UL, 298UL, 294UL, 289UL, 285UL,
    280UL, 276UL, 272UL, 267UL, 263UL, 259UL, 255UL, 251UL,
    247UL, 243UL, 240UL, 236UL, 232UL, 229UL, 225UL, 222UL,
    218UL, 215UL, 212UL, 208UL, 205UL, 202UL, 199UL, 196UL,
    193UL, 190UL, 187UL, 184UL, 181UL, 178UL, 175UL, 173UL,
    170UL, 167UL, 165UL, 162UL, 160UL, 157UL, 155UL, 152UL,
    150UL, 148UL, 145UL, 143UL, 141UL, 139UL, 137UL, 134UL,
    132UL, 130UL, 128UL, 126UL, 124UL, 122UL, 121UL, 119UL,
    117UL, 115UL, 113UL, 111UL, 110UL, 108UL, 106UL, 105UL,
    103UL, 101UL, 100UL, 98UL, 97UL, 95UL, 94UL, 92UL,
    91UL, 90UL, 88UL, 87UL, 85UL, 84UL, 83UL, 82UL,
    80UL, 79UL, 78UL, 77UL, 75UL, 74UL, 73UL, 72UL,
    71UL, 70UL, 69UL, 68UL, 67UL, 66UL, 65UL, 64UL,
    63UL, 62UL, 61UL, 60UL, 59UL, 58UL, 57UL, 56UL,
    55UL, 54UL, 53UL, 53UL, 52UL, 51UL, 50UL, 49UL,
    49UL, 48UL, 47UL, 46UL, 46UL, 45UL, 44UL, 44UL,
    43UL, 42UL, 42UL, 41UL, 40UL, 40UL, 39UL, 39UL,
    38UL, 37UL, 37UL, 36UL, 36UL, 35UL, 35UL, 34UL,
    33UL, 33UL, 32UL, 32UL, 31UL, 31UL, 30UL, 30UL,
    30UL, 29UL, 29UL, 28UL, 28UL, 27UL, 27UL, 26UL,
    26UL, 26UL, 25UL, 25UL, 24UL, 24UL, 24UL, 23UL,
    23UL, 23UL, 22UL, 22UL, 22UL, 21UL, 21UL, 21UL,
    20UL, 20UL, 20UL, 19UL, 19UL, 19UL, 18UL, 18UL,
    18UL, 18UL, 17UL, 17UL, 17UL, 17UL, 16UL, 16UL,
    16UL, 16UL, 15UL, 15UL, 15UL, 15UL, 14UL, 14UL,
    14UL, 14UL, 14UL, 13UL, 13UL, 13UL, 13UL, 13UL,
    12UL, 12UL, 12UL, 12UL, 12UL, 11UL, 11UL, 11UL,
    11UL, 11UL, 11UL, 10UL, 10UL, 10UL, 10UL, 10UL,
    10UL, 9UL, 9UL, 9UL, 9UL, 9UL, 9UL, 9UL,
    8UL, 8UL, 8UL, 8UL, 8UL, 8UL, 8UL, 8UL,
    7UL, 7UL, 7UL, 7UL, 7UL, 7UL, 7UL, 7UL,
    7UL, 6UL, 6UL, 6UL, 6UL, 6UL, 6UL, 6UL,
    6UL, 6UL, 6UL, 6UL, 5UL, 5UL, 5UL, 5UL,
    5UL, 5UL, 5UL, 5UL, 5UL, 5UL, 5UL, 5UL,
    5UL, 4UL, 4UL, 4UL, 4UL, 4UL, 4UL, 4UL,
    4UL, 4UL, 4UL, 4UL, 4UL, 4UL, 4UL, 4UL,
    4UL, 3UL, 3UL, 3UL, 3UL, 3UL, 3UL, 3UL,
    3UL, 3UL, 3UL, 3UL, 3UL, 3UL, 3UL, 3UL,
    3UL, 3UL, 3UL, 3UL, 3UL, 3UL, 3UL, 2UL,
    2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL,
    2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL,
    2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL,
    2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 2UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 1UL,
    1UL, 1UL, 1UL, 1UL, 1UL, 1UL, 0UL, 0UL,
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL,
    0UL
};
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
    a=i?exp_entry_bytes(exp_lut_compact[i-1]):EXP_ONE;
    b=exp_entry_bytes(exp_lut_compact[i]);
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
        for(j=0;j<=p;j++){
#if defined(TINY_MOS) && defined(TINY_WIDE_ASM)
            scratch.attention_av[j]=(int8_t)tiny_attention_coeff(&scratch.phase.attention_scores[j],&big);
#else
            if(wz(&big))scratch.attention_av[j]=0;else{wmu(&t,&scratch.phase.attention_scores[j],127);wrounddiv(&t,&t,&big);scratch.attention_av[j]=(int8_t)WLO(t);}
#endif
        }
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
        matvec(&a->k,&work_normed,&work_tmp);note(&work_tmp,D,&peak_proj);cache_store(&a->key,p,&work_tmp);
        matvec(&a->v,&work_normed,&work_tmp);note(&work_tmp,D,&peak_proj);cache_store(&a->value,p,&work_tmp);
        /* No later layer consumes the last layer's output in cache-only mode. */
        if(!out && l==L-1)return;
        matvec(&a->q,&work_normed,&query);note(&query,D,&peak_proj);
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
    puts("TINY v0.2 / Stage 12E");
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
