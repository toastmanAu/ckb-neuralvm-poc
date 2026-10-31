/*
 * Toy host-injected transaction context (HANDOFF §10). A SHIM modelled on a few CKB syscalls,
 * NOT CKB-VM syscalls: no ecall numbers, no Molecule, no script groups, no hashes.
 *
 * Return codes and the partial-load rule follow ckb-script 1.1.1 (syscalls/mod.rs and
 * syscalls/utils.rs store_data, i.e. RFC 0009): the offset clamps to the item length, *len
 * returns the full size remaining after the offset, and min(*len in, remaining) bytes are copied.
 * Toy differences: an unknown source returns TOY_INDEX_OUT_OF_BOUND (CKB aborts the VM), and on
 * an invalid context every loader returns TOY_ITEM_MISSING.
 *
 * Layout of the 128-byte context (little-endian), written into the ELF by scripts/ctx_v2.py:
 *   0  "CKX1"   4 n_inputs   5 n_outputs   6 args_len   7 witness_len (0xFF = missing)
 *   8  u64 input capacity[2]   24 u64 output capacity[2]   40 args[32]   72 witness[32]
 */
#ifndef TOY_CTX_H
#define TOY_CTX_H
#include <stdint.h>

#define TOY_SUCCESS 0
#define TOY_INDEX_OUT_OF_BOUND 1
#define TOY_ITEM_MISSING 2

#define TOY_SOURCE_INPUT 1  /* toy_load_cell_capacity */
#define TOY_SOURCE_OUTPUT 2
#define TOY_ITEM_ARGS 1     /* toy_partial_load */
#define TOY_ITEM_WITNESS 2

#define TOY_CTX_SIZE 128
#define TOY_MAX_CELLS 2
#define TOY_MAX_BYTES 32
#define TOY_WITNESS_MISSING 0xFF

int toy_ctx_valid(void);
int toy_partial_load(void *dst, uint64_t *len, uint64_t offset, uint64_t source);
/* store_data over any n bytes at src (the rule above); the syscall layer uses it too */
int toy_store_bytes(void *dst, uint64_t *len, uint64_t offset, const volatile uint8_t *src, uint64_t n);
int toy_load_script_args(void *dst, uint64_t *len, uint64_t offset);
int toy_load_witness(void *dst, uint64_t *len, uint64_t offset);
int toy_load_cell_capacity(uint64_t *capacity, uint64_t index, uint64_t source);

#endif
