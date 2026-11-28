/* Minimal sequential ProDOS 8 reader, with a 1 KiB read-ahead buffer.
 * Native SYS image starts at $2000: $0800-$0BFF is the OPEN buffer,
 * $0C00-$0FFF is the read-ahead buffer. No heap or file stdio needed.
 * The OPEN buffer must be FREE in the bitmap before OPEN; ProDOS claims it.
 * No other file is opened while the read-ahead buffer is in use.
 * Host test backend exercises the same buffering/EOF logic. */
#include "tiny-prodos.h"
#include <string.h>
#define CHUNK 1024u
#define PD_EOF 0x4c
struct TinyModel { unsigned pos, len; uint8_t ref, error, eof; };
static TinyModel file;
uint8_t tiny_model_last_error;
#ifdef TINY_PRODOS_HOST_TEST
#include <stdio.h>
static FILE *host;
static uint8_t buffer[CHUNK];
#define READ_BUFFER buffer
static uint8_t pd_open(const char *path) {
    host=fopen(path,"rb"); if(!host)return 0x46;file.ref=1;return 0;
}
static uint8_t pd_read(unsigned *n) {
    *n=(unsigned)fread(buffer,1,CHUNK,host);
    if(ferror(host))return 0x27;
    return *n?0:PD_EOF;
}
static uint8_t pd_close(void) { int r=fclose(host);host=NULL;return r?0x27:0; }
#else
#define READ_BUFFER ((const uint8_t *)(uintptr_t)0x0c00u)
#define BITMAP_08 (*(volatile uint8_t *)(uintptr_t)0xbf59u)
/* Pages $0C-$0F are the four low bits of this ProDOS bitmap byte.
 * After CLOSE, inference reuses the model read-ahead area for code. */
static uint8_t code_claimed;
void *tiny_claim_rescale_code(void) {
    if(file.ref || (BITMAP_08 & 0x0f))return NULL;
    BITMAP_08 |= 0x0f;
    code_claimed=1;
    return (void *)(uintptr_t)0x0c00u;
}
void tiny_release_rescale_code(void) {
    if(code_claimed){BITMAP_08 &= 0xf0;code_claimed=0;}
}
/* Explicit byte arrays avoid any struct-layout/packing dependency. */
extern uint8_t tiny_pd_open_pb[6], tiny_pd_read_pb[8], tiny_pd_close_pb[2];
extern uint8_t tiny_pd_pathname[65];
extern uint8_t tiny_mli_open(void),tiny_mli_read(void),tiny_mli_close(void);
static void put16(uint8_t *p,unsigned n){p[0]=(uint8_t)n;p[1]=(uint8_t)(n>>8);}
static uint8_t pd_open(const char *path) {
    unsigned n=(unsigned)strlen(path);uint8_t e;
    if(!n||n>64u)return 0x40;
    /* Refuse to overwrite anything a previous system program owns. */
    if(BITMAP_08)return 0x56;
    tiny_pd_pathname[0]=(uint8_t)n;memcpy(tiny_pd_pathname+1,path,n);
    tiny_pd_open_pb[0]=3;
    put16(tiny_pd_open_pb+1,(unsigned)(uintptr_t)tiny_pd_pathname);
    put16(tiny_pd_open_pb+3,0x0800u);tiny_pd_open_pb[5]=0;
    e=tiny_mli_open();if(!e)file.ref=tiny_pd_open_pb[5];return e;
}
static uint8_t pd_read(unsigned *n) {
    uint8_t e;tiny_pd_read_pb[0]=4;tiny_pd_read_pb[1]=file.ref;
    put16(tiny_pd_read_pb+2,0x0c00u);put16(tiny_pd_read_pb+4,CHUNK);
    tiny_pd_read_pb[6]=tiny_pd_read_pb[7]=0;
    e=tiny_mli_read();*n=tiny_pd_read_pb[6]+256u*tiny_pd_read_pb[7];return e;
}
static uint8_t pd_close(void) {
    tiny_pd_close_pb[0]=1;tiny_pd_close_pb[1]=file.ref;return tiny_mli_close();
}
#endif
TinyModel *tiny_model_open(const char *path) {
    uint8_t e;if(file.ref){tiny_model_last_error=0x50;return NULL;}
    memset(&file,0,sizeof(file));tiny_model_last_error=0;
    e=pd_open(path);if(e){tiny_model_last_error=e;return NULL;}return &file;
}
static int refill(TinyModel *f) {
    unsigned n=0;uint8_t e;
    if(f->error||f->eof||!f->ref)return 0;
    e=pd_read(&n);
    f->pos=f->len=0;
    if(e==PD_EOF && n==0){f->eof=1;return 0;}
    if(e||n>CHUNK){f->error=tiny_model_last_error=e?e:0x27;return 0;}
    if(!n){f->eof=1;return 0;}
    f->len=n;return 1;
}
size_t tiny_model_read(TinyModel *f,void *dest,size_t n) {
    uint8_t *p=dest;size_t done=0;unsigned take;
    while(done<n){
        if(f->pos==f->len&&!refill(f))break;
        take=f->len-f->pos;if(take>n-done)take=(unsigned)(n-done);
        memcpy(p+done,READ_BUFFER+f->pos,take);f->pos+=take;done+=take;
    }
    return done;
}
int tiny_model_getc(TinyModel *f) {uint8_t b;return tiny_model_read(f,&b,1)==1?b:-1;}
int tiny_model_error(TinyModel *f) {return f->error;}
int tiny_model_close(TinyModel *f) {
    uint8_t e;if(!f->ref)return 0;e=pd_close();f->ref=0;
    if(e)tiny_model_last_error=e;
    return e;
}
void tiny_model_abort(void) {if(file.ref)(void)tiny_model_close(&file);}
