#ifndef TINY_PRODOS_H
#define TINY_PRODOS_H
#include <stddef.h>
#include <stdint.h>
/* Read-only, single open file. Returns 0 on close success; getc returns -1
 * at EOF/error. Distinguish EOF from errors using tiny_model_error(). */
typedef struct TinyModel TinyModel;
extern uint8_t tiny_model_last_error;
TinyModel *tiny_model_open(const char *path);
size_t tiny_model_read(TinyModel *f, void *dest, size_t n);
int tiny_model_getc(TinyModel *f);
int tiny_model_error(TinyModel *f);
int tiny_model_close(TinyModel *f);
void tiny_model_abort(void);
#endif
