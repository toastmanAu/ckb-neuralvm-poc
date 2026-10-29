.section .text.init
.globl _start
_start:
.weak __tls_base
la tp, __tls_base
call main
call exit
1: j 1b
