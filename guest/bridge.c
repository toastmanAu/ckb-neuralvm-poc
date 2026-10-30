/*
 * CKB-NeuralVM PoC v0.2: a CKB-INSPIRED toy validation firmware.
 * WARNING: Not a CKB lock, not consensus-valid, not signature verification.
 * Runs as an RV32 firmware inside Qwenputer's neural CPU.
 *
 * This file is the platform: Qwenputer ecalls and the 80-token text bridge. The payload comes
 * from the tool call (the upstream calc bridge's text-token capture, not CKB syscalls); the
 * transaction context is host-injected read-only data (ctx.c).
 */
#include <stdint.h>

#include "contract.h"
#include "emit.h"

#define TEXT ((volatile const uint8_t *)0x8000u)
#define TEXT_ROWS 80u
#define TEXT_ROW_BYTES 16u
#define PAYLOAD_LEN 160u
#define TOOL_RESPONSE 256
#define TOOL_RESPONSE_END 257
#define THINK 258
#define THINK_END 259

static char payload[PAYLOAD_LEN];

static inline long rv_ecall2(long n, long a0, long a1, long a2) {
    register long r0 __asm__("a0") = a0;
    register long r1 __asm__("a1") = a1;
    register long r2 __asm__("a2") = a2;
    register long r7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r0) : "r"(r1), "r"(r2), "r"(r7) : "memory");
    return r0;
}

void toy_putc(int c) { (void)rv_ecall2(1, c, 0, 0); }

/* Qwenputer 'exit' is a HALT/park; GO resumes immediately afterwards. */
__attribute__((noreturn)) void exit(int code) {
    for (;;) (void)rv_ecall2(93, code, 0, 0);
}

/* The bytes after the second '>' (of <parameter=payload>) up to the next '<', minus newlines. */
static int gather_payload(void) {
    unsigned n = 0;
    unsigned gt = 0;
    for (unsigned row = 1; row <= TEXT_ROWS; ++row) {
        const volatile uint8_t *r = TEXT + row * TEXT_ROW_BYTES;
        if (r[0] == 0) break;
        for (unsigned col = 0; col < TEXT_ROW_BYTES && r[col] != 0; ++col) {
            const uint8_t c = r[col];
            if (gt < 2) { if (c == '>') ++gt; continue; }
            if (c == '<') {
                payload[n] = 0;
                return n > 0 ? 0 : -1;
            }
            if (c == '\n' || c == '\r') continue;
            if (n + 1 >= PAYLOAD_LEN) return -2;
            payload[n++] = (char)c;
        }
    }
    payload[n] = 0;
    return -1;
}

int main(void) {
    for (;;) {
        (void)rv_ecall2(93, 0, 0, 0); /* Initial park and subsequent parks. */
        const int status = gather_payload();
        toy_putc('\n'); toy_putc(TOOL_RESPONSE); toy_putc('\n');
        if (status == 0) contract_dispatch(payload);
        else if (status == -2) emit_str("FAIL;R=TOO_LONG\n");
        else emit_str("FAIL;R=BRIDGE\n");
        toy_putc(TOOL_RESPONSE_END); toy_putc('\n');
        toy_putc(THINK); toy_putc('\n'); toy_putc('\n'); toy_putc(THINK_END); toy_putc('\n');
    }
}
