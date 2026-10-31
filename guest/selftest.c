/*
 * Bridge-free build of the contracts for upstream's strict circuit check (hf.check drives the
 * text-free chip core from GO, so it cannot run the tool-call bridge). main runs one payload over
 * the injected context, prints the verdict and exits. Not the firmware the neural runs use.
 */
#include "contract.h"
#include "emit.h"

#ifndef SELFTEST_PAYLOAD
#define SELFTEST_PAYLOAD "V3"
#endif

static inline long rv_ecall(long n, long a0) {
    register long r0 __asm__("a0") = a0;
    register long r7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r0) : "r"(r7) : "memory");
    return r0;
}

void toy_putc(int c) { (void)rv_ecall(1, c); }

__attribute__((noreturn)) void exit(int code) {
    for (;;) (void)rv_ecall(93, code);
}

int main(void) {
    contract_dispatch(SELFTEST_PAYLOAD);
    return 0;
}
