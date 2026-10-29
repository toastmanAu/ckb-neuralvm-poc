ENTRY(_start)
MEMORY { RAM (rw) : ORIGIN = 0x100, LENGTH = 0x7F00
  DROM (r) : ORIGIN = 0x20000, LENGTH = 128K
  ROM (rx) : ORIGIN = 0x40000, LENGTH = 256K }
SECTIONS { .text : { *(.text.init) *(.text .text.*) } > ROM
.rodata : { *(.rodata .rodata.* .srodata .srodata.*) *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr) } > DROM
.data : { *(.data .data.* .sdata .sdata.*) } > RAM
.tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
.tbss : { *(.tbss .tbss.*) } > RAM
.bss : { . = . + SIZEOF(.tbss); *(.bss .bss.* .sbss .sbss.*) *(COMMON) . = ALIGN(16); __heap_start = .; } > RAM
__heap_end = 0x7000; _end = __heap_start; }
