/*
 * CKB-numbered syscalls over the toy context (v0.3). The guest issues a real `ecall` with the CKB
 * syscall number in a7; the PATCHED Qwenputer circuit (patches/qwenputer-ecall-trap.patch) turns
 * 2052/2071/2074 into a jump to ckb_trap_entry (guest/start.S), and ckb_trap_dispatch (trap.c)
 * serves them. The stock circuit faults on these numbers.
 *
 * Still a toy, NOT CKB-VM syscalls: RV32 (pointers and offsets are 32 bits), the trap saves no
 * return address (calls must go through ckb_syscall), sources are only 1 (input) and 2 (output),
 * and the data returned are toy encodings:
 *   2052 load_script(addr, len*, offset)                  the script args bytes (not a Molecule Script)
 *   2071 load_cell(addr, len*, offset, index, source)     the cell's capacity, 8 bytes LE (not a CellOutput)
 *   2074 load_witness(addr, len*, offset, index, source)  the single witness at index 0
 * Partial loading and return codes follow ckb-script 1.1.1 store_data; an absent witness index,
 * including a missing witness, is INDEX_OUT_OF_BOUND as in ckb-script.
 */
#ifndef TOY_SYSCALL_H
#define TOY_SYSCALL_H
#include <stdint.h>

#define SYS_LOAD_SCRIPT 2052
#define SYS_LOAD_CELL 2071
#define SYS_LOAD_WITNESS 2074

long ckb_syscall(long a0, long a1, long a2, long a3, long a4, long a5, long n);
long ckb_trap_dispatch(long a0, long a1, long a2, long a3, long a4, long a5, long n);

static inline int ckb_load_script(void *addr, uint64_t *len, uint32_t offset) {
    return (int)ckb_syscall((long)addr, (long)len, (long)offset, 0, 0, 0, SYS_LOAD_SCRIPT);
}
static inline int ckb_load_cell(void *addr, uint64_t *len, uint32_t offset, uint32_t index, uint32_t source) {
    return (int)ckb_syscall((long)addr, (long)len, (long)offset, (long)index, (long)source, 0, SYS_LOAD_CELL);
}
static inline int ckb_load_witness(void *addr, uint64_t *len, uint32_t offset, uint32_t index, uint32_t source) {
    return (int)ckb_syscall((long)addr, (long)len, (long)offset, (long)index, (long)source, 0, SYS_LOAD_WITNESS);
}

#endif
