#ifndef TINY_WIDE_H
#define TINY_WIDE_H
#include <stdint.h>
/* Shared by the inference engine and the standalone ABI test. */
typedef struct { uint32_t lo, hi; } Wide;
_Static_assert(sizeof(Wide)==8, "wide ABI requires eight bytes");
void wa(Wide *,const Wide *,const Wide *);
void ws(Wide *,const Wide *,const Wide *);
void wneg(Wide *,const Wide *);
void wmag(Wide *,const Wide *);
void winc(Wide *);
int wz(const Wide *);
int wn(const Wide *);
int wc(const Wide *,const Wide *);
int wcs(const Wide *,const Wide *);
void wl(Wide *,const Wide *,int);
void wr(Wide *,const Wide *,int);
unsigned wbit(const Wide *,int);
int topbit(const Wide *);
void wm(Wide *,const Wide *,const Wide *);
/* Nonzero unsigned divisor required; public C wrappers check this. */
void tiny_wdiv(Wide *,const Wide *,const Wide *);
void tiny_wrounddiv(Wide *,const Wide *,const Wide *);
#endif
