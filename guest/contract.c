/*
 * Explicitly toy validation: no Molecule parsing, no hashes, no signatures. The args/witness
 * byte equality is a toy tag check, NOT authentication.
 */
#include "contract.h"
#include "ctx.h"
#include "emit.h"
#include "parse.h"
#include "syscall.h"

#define CHUNK 4u  /* the lock compares args and witness through 4-byte partial loads */

static void fail(const char *reason) {
    emit_str("FAIL;R=");
    emit_str(reason);
    toy_putc('\n');
}

static void pass_fee(uint64_t fee) {
    emit_str("PASS;FEE=");
    emit_hex(fee, 16);
    toy_putc('\n');
}

/* v0.1 semantics: PASS iff L == W and I >= O; fee = I - O. */
static void contract_v1(const char *payload) {
    struct v1_tx tx;
    if (!parse_v1(payload, &tx)) { fail("FMT"); return; }
    if (tx.lock_tag != tx.witness_tag) { fail("LOCK"); return; }
    if (tx.input_capacity < tx.output_capacity) { fail("CAP"); return; }
    pass_fee(tx.input_capacity - tx.output_capacity);
}

/* Sum of the capacities of one source, iterating until TOY_INDEX_OUT_OF_BOUND as a CKB script
 * does. Returns 0 on u64 overflow, else 1, with the count in *n. */
static int sum_capacity(uint64_t source, uint64_t *sum, unsigned *n) {
    uint64_t cap;
    *sum = 0;
    *n = 0;
    while (toy_load_cell_capacity(&cap, *n, source) == TOY_SUCCESS) {
        if (*sum + cap < *sum) return 0;
        *sum += cap;
        ++*n;
    }
    return 1;
}

static int lock_matches(void) {
    uint8_t a[CHUNK], w[CHUNK];
    uint64_t args_len = 0, wit_len = 0;  /* *len = 0: a length query */
    (void)toy_load_script_args(a, &args_len, 0);
    (void)toy_load_witness(w, &wit_len, 0);
    if (args_len != wit_len) return 0;
    for (uint64_t off = 0; off < args_len; off += CHUNK) {
        uint64_t la = CHUNK, lw = CHUNK;
        (void)toy_load_script_args(a, &la, off);
        (void)toy_load_witness(w, &lw, off);
        const unsigned got = la < CHUNK ? (unsigned)la : CHUNK;  /* *len out: bytes remaining */
        for (unsigned i = 0; i < got; ++i)
            if (a[i] != w[i]) return 0;
    }
    return 1;
}

/* v0.2 toy lock: a valid context, a witness present, args == witness byte for byte, at least
 * one input, no u64 overflow in either sum, inputs >= outputs; fee = inputs - outputs. */
static void contract_v2(void) {
    uint64_t probe_len = 0, sum_in, sum_out;
    unsigned n_in, n_out;
    uint8_t probe[1];
    if (!toy_ctx_valid()) { fail("CTX"); return; }
    if (toy_load_witness(probe, &probe_len, 0) == TOY_ITEM_MISSING) { fail("NOWIT"); return; }
    if (!lock_matches()) { fail("LOCK"); return; }
    if (!sum_capacity(TOY_SOURCE_INPUT, &sum_in, &n_in)) { fail("OVF"); return; }
    if (n_in == 0) { fail("NOIN"); return; }
    if (!sum_capacity(TOY_SOURCE_OUTPUT, &sum_out, &n_out)) { fail("OVF"); return; }
    if (sum_in < sum_out) { fail("CAP"); return; }
    pass_fee(sum_in - sum_out);
}

/* ";<rc>,<*len out, 2 hex>,<copied bytes, hex>" */
static void probe_load(uint64_t item, uint64_t buf_len, uint64_t offset) {
    uint8_t buf[CHUNK];
    uint64_t len = buf_len;
    const int rc = toy_partial_load(buf, &len, offset, item);
    toy_putc(';');
    emit_hex((uint64_t)rc, 1);
    toy_putc(',');
    emit_hex(len, 2);
    toy_putc(',');
    if (rc == TOY_SUCCESS)
        for (uint64_t i = 0; i < buf_len && i < len; ++i) emit_hex(buf[i], 2);
}

/* ";<rc>[,<capacity, 16 hex>]" */
static void probe_cap(uint64_t index, uint64_t source) {
    uint64_t cap;
    const int rc = toy_load_cell_capacity(&cap, index, source);
    toy_putc(';');
    emit_hex((uint64_t)rc, 1);
    if (rc == TOY_SUCCESS) { toy_putc(','); emit_hex(cap, 16); }
}

/* A fixed list of API calls, mirrored by scripts/reference_v2.py PROBES. */
static void probe_v2(void) {
    toy_putc('T');
    probe_load(TOY_ITEM_ARGS, 0, 0);
    probe_load(TOY_ITEM_ARGS, CHUNK, 0);
    probe_load(TOY_ITEM_ARGS, CHUNK, 30);
    probe_load(TOY_ITEM_ARGS, CHUNK, UINT64_MAX);
    probe_load(TOY_ITEM_WITNESS, 0, 0);
    probe_load(TOY_ITEM_WITNESS, CHUNK, 1);
    probe_load(3, CHUNK, 0);
    probe_cap(0, TOY_SOURCE_INPUT);
    probe_cap(1, TOY_SOURCE_INPUT);
    probe_cap(2, TOY_SOURCE_INPUT);
    probe_cap(UINT64_MAX, TOY_SOURCE_INPUT);
    probe_cap(0, TOY_SOURCE_OUTPUT);
    probe_cap(0, 3);
    toy_putc('\n');
}

/* ---- v0.3: the same toy lock, every read a CKB-numbered ecall (syscall.h) ---- */

static int cell_capacity(uint64_t index, uint64_t source, uint64_t *cap) {
    uint8_t le[8];
    uint64_t len = 8;
    const int rc = ckb_load_cell(le, &len, 0, (uint32_t)index, (uint32_t)source);
    if (rc != TOY_SUCCESS) return rc;
    *cap = 0;
    for (int i = 7; i >= 0; --i) *cap = (*cap << 8) | le[i];
    return TOY_SUCCESS;
}

static int sum_capacity_v3(uint64_t source, uint64_t *sum, unsigned *n) {
    uint64_t cap;
    *sum = 0;
    *n = 0;
    while (cell_capacity(*n, source, &cap) == TOY_SUCCESS) {
        if (*sum + cap < *sum) return 0;
        *sum += cap;
        ++*n;
    }
    return 1;
}

static int lock_matches_v3(void) {
    uint8_t a[CHUNK], w[CHUNK];
    uint64_t args_len = 0, wit_len = 0;
    (void)ckb_load_script(a, &args_len, 0);
    (void)ckb_load_witness(w, &wit_len, 0, 0, TOY_SOURCE_INPUT);
    if (args_len != wit_len) return 0;
    for (uint32_t off = 0; off < args_len; off += CHUNK) {
        uint64_t la = CHUNK, lw = CHUNK;
        (void)ckb_load_script(a, &la, off);
        (void)ckb_load_witness(w, &lw, off, 0, TOY_SOURCE_INPUT);
        const unsigned got = la < CHUNK ? (unsigned)la : CHUNK;
        for (unsigned i = 0; i < got; ++i)
            if (a[i] != w[i]) return 0;
    }
    return 1;
}

/* Same verdicts as contract_v2 on every context. */
static void contract_v3(void) {
    uint64_t probe_len = 0, sum_in, sum_out;
    unsigned n_in, n_out;
    uint8_t probe[1];
    if (!toy_ctx_valid()) { fail("CTX"); return; }
    if (ckb_load_witness(probe, &probe_len, 0, 0, TOY_SOURCE_INPUT) != TOY_SUCCESS) { fail("NOWIT"); return; }
    if (!lock_matches_v3()) { fail("LOCK"); return; }
    if (!sum_capacity_v3(TOY_SOURCE_INPUT, &sum_in, &n_in)) { fail("OVF"); return; }
    if (n_in == 0) { fail("NOIN"); return; }
    if (!sum_capacity_v3(TOY_SOURCE_OUTPUT, &sum_out, &n_out)) { fail("OVF"); return; }
    if (sum_in < sum_out) { fail("CAP"); return; }
    pass_fee(sum_in - sum_out);
}

/* ";<rc>,<*len out, 2 hex>,<copied bytes, hex>" for one syscall */
static void probe_sys(long n, uint64_t buf_len, uint32_t offset, uint32_t index, uint32_t source) {
    uint8_t buf[8];
    uint64_t len = buf_len;
    const int rc = (int)ckb_syscall((long)buf, (long)&len, (long)offset, (long)index, (long)source, 0, n);
    toy_putc(';');
    emit_hex((uint64_t)rc, 1);
    toy_putc(',');
    emit_hex(len, 2);
    toy_putc(',');
    if (rc == TOY_SUCCESS)
        for (uint64_t i = 0; i < buf_len && i < len; ++i) emit_hex(buf[i], 2);
}

/* A fixed list of syscalls, mirrored by scripts/reference_v2.py SYS_PROBES. */
static void probe_v3(void) {
    toy_putc('S');
    probe_sys(SYS_LOAD_SCRIPT, 0, 0, 0, 0);
    probe_sys(SYS_LOAD_SCRIPT, 4, 30, 0, 0);
    probe_sys(SYS_LOAD_WITNESS, 4, 1, 0, TOY_SOURCE_INPUT);
    probe_sys(SYS_LOAD_WITNESS, 4, 0, 1, TOY_SOURCE_INPUT);
    probe_sys(SYS_LOAD_WITNESS, 4, 0, 0, 3);
    probe_sys(SYS_LOAD_CELL, 8, 0, 0, TOY_SOURCE_INPUT);
    probe_sys(SYS_LOAD_CELL, 4, 6, 0, TOY_SOURCE_INPUT);
    probe_sys(SYS_LOAD_CELL, 8, 0, 2, TOY_SOURCE_INPUT);
    probe_sys(SYS_LOAD_CELL, 8, 0, 0, 3);
    toy_putc('\n');
}

void contract_dispatch(const char *payload) {
    if (str_eq(payload, "V2")) contract_v2();
    else if (str_eq(payload, "V2;T")) probe_v2();
    else if (str_eq(payload, "V3")) contract_v3();
    else if (str_eq(payload, "V3;T")) probe_v3();
    else contract_v1(payload);
}
