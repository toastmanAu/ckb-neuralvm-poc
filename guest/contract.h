#ifndef TOY_CONTRACT_H
#define TOY_CONTRACT_H

/* Runs the contract the payload selects and prints its one-line verdict:
 *   "V2"    toy lock over the host-injected context (contract_v2)
 *   "V2;T"  API probe transcript over the context (probe_v2)
 *   else    the v0.1 payload check (contract_v1); a malformed payload is FAIL;R=FMT */
void contract_dispatch(const char *payload);

#endif
