/*
 * Native (x86-64) build of the same contract and context sources, for differential testing.
 * NOT the RV32 firmware and NOT neural execution. Built with -DTOY_HOST by scripts/test_host_c.py.
 *
 *   host_toy run <256 hex context> <payload>   prints the contract's verdict line
 *   host_toy api                               reads API calls on stdin, one result per line:
 *       C <256 hex>            set the context
 *       L <source> <len> <off> toy_partial_load  -> "<rc> <len out> <hex bytes copied or ->"
 *       K <index> <source>     toy_load_cell_capacity -> "<rc> <capacity or ->"
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "contract.h"
#include "ctx.h"
#include "emit.h"

extern uint8_t ckb_ctx[TOY_CTX_SIZE];

void toy_putc(int c) { putchar(c); }

static int set_ctx(const char *hex) {
    if (strlen(hex) != 2 * TOY_CTX_SIZE) return 0;
    for (unsigned i = 0; i < TOY_CTX_SIZE; ++i) {
        unsigned b;
        if (sscanf(hex + 2 * i, "%2x", &b) != 1) return 0;
        ckb_ctx[i] = (uint8_t)b;
    }
    return 1;
}

static int api(void) {
    char op[4], hex[2 * TOY_CTX_SIZE + 2];
    uint64_t a, b, c;
    while (scanf("%3s", op) == 1) {
        if (op[0] == 'C') {
            if (scanf("%257s", hex) != 1 || !set_ctx(hex)) return 2;
        } else if (op[0] == 'L') {
            uint8_t buf[TOY_MAX_BYTES + 8];
            if (scanf("%" SCNu64 " %" SCNu64 " %" SCNu64, &a, &b, &c) != 3 || b > sizeof buf) return 2;
            memset(buf, 0xEE, sizeof buf);
            uint64_t len = b;
            const int rc = toy_partial_load(buf, &len, c, a);
            printf("%d %" PRIu64 " ", rc, len);
            const uint64_t got = rc == TOY_SUCCESS ? (b < len ? b : len) : 0;
            if (!got) putchar('-');
            for (uint64_t i = 0; i < got; ++i) printf("%02X", buf[i]);
            for (uint64_t i = got; i < sizeof buf; ++i)
                if (buf[i] != 0xEE) { printf(" OVERWRITE"); break; }
            putchar('\n');
        } else if (op[0] == 'K') {
            uint64_t cap = 0;
            if (scanf("%" SCNu64 " %" SCNu64, &a, &b) != 2) return 2;
            const int rc = toy_load_cell_capacity(&cap, a, b);
            if (rc == TOY_SUCCESS) printf("%d %" PRIu64 "\n", rc, cap);
            else printf("%d -\n", rc);
        } else {
            return 2;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "run")) {
        if (!set_ctx(argv[2])) return 2;
        contract_dispatch(argv[3]);
        return 0;
    }
    if (argc == 2 && !strcmp(argv[1], "api")) return api();
    fprintf(stderr, "usage: host_toy run <ctx hex> <payload> | host_toy api\n");
    return 2;
}
