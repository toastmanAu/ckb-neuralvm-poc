#include "emit.h"

void emit_str(const char *s) {
    while (*s) toy_putc((unsigned char)*s++);
}

void emit_hex(uint64_t v, unsigned digits) {
    static const char h[] = "0123456789ABCDEF";
    for (unsigned i = digits; i-- > 0;) toy_putc(h[(v >> (i * 4)) & 15u]);
}
