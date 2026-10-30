#ifndef TOY_EMIT_H
#define TOY_EMIT_H
#include <stdint.h>

void toy_putc(int c);  /* per platform: bridge.c (Qwenputer putchar ecall) or the host harness */
void emit_str(const char *s);
void emit_hex(uint64_t v, unsigned digits);  /* uppercase, most significant digit first */

#endif
