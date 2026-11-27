#ifndef TINY_WIDE_H
#define TINY_WIDE_H
#include <stdint.h>
#include <stddef.h>
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
/* Native matrix rescaling: signed 24-bit dot, unsigned 24-bit row
 * coefficient, peak <= 32767. The coefficient is little-endian. */
typedef struct { int32_t dot; uint16_t peak; } TinyRescaleArgs;
_Static_assert(offsetof(TinyRescaleArgs, peak)==4, "rescale ABI layout");
void tiny_rescale(Wide *,const uint8_t *,const TinyRescaleArgs *);
/* Packing-only conversion. Shift in [-14,49], rounded result must fit
 * [-32767,32767] (guaranteed by pack's peak calculation). */
int16_t tiny_pack16(const Wide *,int shift);
/* Exact round(127*a/b), or zero if b==0. Attention alignment
 * guarantees 0 <= a <= b <= 2^41. Return range 0..127. */
int tiny_attention_coeff(const Wide *a,const Wide *b);
/* Quantize n=1..256 packed mantissas with nonzero peak <=32767.
 * args[0]=n, args[1]=peak; each abs(input[i])<=peak. No overlap. */
void tiny_q8_vector(const int16_t *,int8_t *,const uint16_t *);
/* Attention score for one cached position: raw = dot*m*csm,
 * shift = max(0, topbit(|raw|)-29), out = round(raw/2^shift)*factor.
 * Returns shift; top = topbit(|out|), or 0xff when out is zero. */
typedef struct { int32_t dot; uint16_t m, csm; uint32_t factor; uint8_t top; } TinyScoreArgs;
_Static_assert(offsetof(TinyScoreArgs, factor)==8 && offsetof(TinyScoreArgs, top)==12, "score ABI layout");
int tiny_att_score(Wide *,TinyScoreArgs *);
/* Rescale count kernel results: eight-byte records {scale, dot}; args = {count, peak}. */
void tiny_rescale_rows(Wide *,const uint8_t *,const uint16_t *);
/* Largest magnitude of n >= 1 signed values. */
void tiny_wmaxmag(Wide *,const Wide *,unsigned n);
/* Pack args[0] >= 1 values with shift args[1]; see tiny_pack16. */
void tiny_pack_rows(int16_t *,const Wide *,const int16_t *args);
/* out[i] = shift(a[i],sa) + shift(b[i],sb): 0..16 shifts left exactly,
 * -1..-16 shifts right, rounded to nearest with ties away from zero. */
typedef struct { const int16_t *b; uint16_t n; int8_t sa, sb; } TinyResidualArgs;
_Static_assert(offsetof(TinyResidualArgs, n)==sizeof(void *) && offsetof(TinyResidualArgs, sa)==sizeof(void *)+2, "residual ABI layout");
void tiny_residual_rows(Wide *,const int16_t *,const TinyResidualArgs *);
/* Sum of squares of n=1..256 packed mantissas. */
void tiny_sumsq(Wide *,const int16_t *,unsigned n);
/* Nonzero unsigned divisor required; public C wrappers check this. */
void tiny_wdiv(Wide *,const Wide *,const Wide *);
void tiny_wrounddiv(Wide *,const Wide *,const Wide *);
#endif
