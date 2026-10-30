#include "ctx.h"

#ifdef TOY_HOST
uint8_t ckb_ctx[TOY_CTX_SIZE];  /* filled by the native harness */
#else
/* In DROM (.rodata). The host overwrites these bytes in the ELF file; unpatched, the magic is
 * wrong and the context is invalid. A plain const object (a const volatile one lands in GCC
 * 10's writable .sdata), read only through a volatile pointer so every read is a real load. */
__attribute__((used, aligned(8))) const uint8_t ckb_ctx[TOY_CTX_SIZE] = {'N', 'O', 'N', 'E'};
#endif
#define CTX ((const volatile uint8_t *)ckb_ctx)

enum { N_IN = 4, N_OUT = 5, ARGS_LEN = 6, WIT_LEN = 7, IN_CAP = 8, OUT_CAP = 24, ARGS = 40, WIT = 72 };

static uint64_t rd64(unsigned off) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | CTX[off + (unsigned)i];
    return v;
}

int toy_ctx_valid(void) {
    const unsigned w = CTX[WIT_LEN];
    return CTX[0] == 'C' && CTX[1] == 'K' && CTX[2] == 'X' && CTX[3] == '1' &&
           CTX[N_IN] <= TOY_MAX_CELLS && CTX[N_OUT] <= TOY_MAX_CELLS &&
           CTX[ARGS_LEN] <= TOY_MAX_BYTES && (w <= TOY_MAX_BYTES || w == TOY_WITNESS_MISSING);
}

int toy_partial_load(void *dst, uint64_t *len, uint64_t offset, uint64_t source) {
    unsigned base, n;
    if (!toy_ctx_valid()) return TOY_ITEM_MISSING;
    if (source == TOY_ITEM_ARGS) {
        base = ARGS; n = CTX[ARGS_LEN];
    } else if (source == TOY_ITEM_WITNESS) {
        if (CTX[WIT_LEN] == TOY_WITNESS_MISSING) return TOY_ITEM_MISSING;
        base = WIT; n = CTX[WIT_LEN];
    } else {
        return TOY_INDEX_OUT_OF_BOUND;
    }
    const uint64_t off = offset < n ? offset : n;
    const uint64_t full = n - off;
    const uint64_t real = *len < full ? *len : full;
    uint8_t *d = dst;
    for (uint64_t i = 0; i < real; ++i) d[i] = CTX[base + (unsigned)(off + i)];
    *len = full;
    return TOY_SUCCESS;
}

int toy_load_script_args(void *dst, uint64_t *len, uint64_t offset) {
    return toy_partial_load(dst, len, offset, TOY_ITEM_ARGS);
}

int toy_load_witness(void *dst, uint64_t *len, uint64_t offset) {
    return toy_partial_load(dst, len, offset, TOY_ITEM_WITNESS);
}

int toy_load_cell_capacity(uint64_t *capacity, uint64_t index, uint64_t source) {
    unsigned base, n;
    if (!toy_ctx_valid()) return TOY_ITEM_MISSING;
    if (source == TOY_SOURCE_INPUT) {
        base = IN_CAP; n = CTX[N_IN];
    } else if (source == TOY_SOURCE_OUTPUT) {
        base = OUT_CAP; n = CTX[N_OUT];
    } else {
        return TOY_INDEX_OUT_OF_BOUND;
    }
    if (index >= n) return TOY_INDEX_OUT_OF_BOUND;
    *capacity = rd64(base + 8u * (unsigned)index);
    return TOY_SUCCESS;
}
