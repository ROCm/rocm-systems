# Scalar control qualification

MOVREL uses DWORD offsets with full-width M0; its split form uses M0[9:0]
for the source and M0[25:16] for the destination. Out-of-range sources select
SGPR0 and invalid destinations suppress the write. B64 offsets clear bit zero.
The range, non-wrapping full-width behavior and B64 alignment were measured on
gfx1100 and gfx1201. The SGPR/VCC region and odd-index behavior on older CDNA
are extrapolated from RDNA, where the manuals spell out those details.

The screen-partition table is published for CDNA1/2. CDNA3/4 retain the opcode
in machine-readable ISA; their table is extrapolated from CDNA2. Tests cover
all 256 selectors, scalar and SIMD execution, aliases, EXEC, DPP, and SDWA.

Several archive cases have invalid references rather than execution defects:

- Initial EXEC for a one-thread workgroup has only bit zero set. Scalar popcount
  tests must use that initial mask, not assume all 64 lanes are active.
- The signed multiply-add overflow indicator reports signed overflow, including
  INT64_MIN plus -1; it is not an unsigned carry-out bit.
- Screen partition returns a four-bit coverage mask, not a single engine ID.
- HW_ID fields depend on configured topology and dispatch placement. The archive
  assumes a fixed single-XCC schedule while its targets use multi-XCC profiles.
- S_RFE and S_RFE_RESTORE are privileged. An ordinary HIP kernel cannot write
  trap-handler state and return as though it entered a configured trap handler.
  The existing WaveDebug configured-handler tests cover the authorized path.
