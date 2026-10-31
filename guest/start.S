.section .text.init
.globl _start
_start:
.weak __tls_base
la tp, __tls_base
call main
call exit
1: j 1b

/* ckb-neuralvm trap patch: on the patched circuit, ecall with a7 in {2052, 2071, 2074} jumps
 * here, to ROM word 0x10 (spec.TRAP_WORD), without saving a return address; the stock circuit
 * faults instead. The call came through ckb_syscall, so ra still returns to its caller, and the
 * tail call below returns there with the result in a0. This file links first, so the .org puts
 * the entry at 0x40040 (the build checks). */
.org 0x40
.globl ckb_trap_entry
ckb_trap_entry:
mv a6, a7
j ckb_trap_dispatch

/* long ckb_syscall(a0, a1, a2, a3, a4, a5, number in a6): a CKB-numbered ecall. */
.text
.globl ckb_syscall
ckb_syscall:
mv a7, a6
ecall
ret /* not reached: a trapped ecall jumps to ckb_trap_entry, an unknown one faults */
