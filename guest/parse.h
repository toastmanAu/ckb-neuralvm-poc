#ifndef TOY_PARSE_H
#define TOY_PARSE_H
#include <stdint.h>

/* The v0.1 wire payload: V1;I=<16 uppercase hex>;O=<16 uppercase hex>;L=<2 hex>;W=<2 hex> */
struct v1_tx {
    uint64_t input_capacity, output_capacity, lock_tag, witness_tag;
};

int parse_v1(const char *payload, struct v1_tx *tx);  /* 1 on success */
int str_eq(const char *a, const char *b);

#endif
