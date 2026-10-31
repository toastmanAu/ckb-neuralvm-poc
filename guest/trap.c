/* The handler for the trapped CKB-numbered ecalls (guest/syscall.h), over the toy context. */
#include "ctx.h"
#include "syscall.h"

static int load_cell(void *addr, uint64_t *len, uint64_t offset, uint64_t index, uint64_t source) {
    uint64_t cap;
    uint8_t le[8];
    const int rc = toy_load_cell_capacity(&cap, index, source);
    if (rc != TOY_SUCCESS) return rc;
    for (unsigned i = 0; i < 8; ++i) le[i] = (uint8_t)(cap >> (8 * i));
    return toy_store_bytes(addr, len, offset, le, 8);
}

static int load_witness(void *addr, uint64_t *len, uint64_t offset, uint64_t index, uint64_t source) {
    if (!toy_ctx_valid()) return TOY_ITEM_MISSING;
    if (source != TOY_SOURCE_INPUT && source != TOY_SOURCE_OUTPUT) return TOY_INDEX_OUT_OF_BOUND;
    if (index != 0) return TOY_INDEX_OUT_OF_BOUND;
    const int rc = toy_load_witness(addr, len, offset);
    return rc == TOY_ITEM_MISSING ? TOY_INDEX_OUT_OF_BOUND : rc;  /* no witness at index 0 */
}

long ckb_trap_dispatch(long a0, long a1, long a2, long a3, long a4, long a5, long n) {
    (void)a5;
    void *addr = (void *)a0;
    uint64_t *len = (uint64_t *)a1;
    const uint64_t offset = (unsigned long)a2, index = (unsigned long)a3, source = (unsigned long)a4;
    if (n == SYS_LOAD_SCRIPT) return toy_load_script_args(addr, len, offset);
    if (n == SYS_LOAD_CELL) return load_cell(addr, len, offset, index, source);
    if (n == SYS_LOAD_WITNESS) return load_witness(addr, len, offset, index, source);
    return TOY_INDEX_OUT_OF_BOUND;  /* not reached on the patched circuit: other numbers fault */
}
