# Missing formatted buffer store channels

Formatted stores with fewer source components than the memory format use zero
for missing components on CDNA1-4 and RDNA1-3.5. RDNA4 replicates X.

A native HIP probe on 2026-10-14 issued `tbuffer_store_d16_format_xyz` to a
16_16_16_16_UINT format from packed sources `beefabcd c0feccdd`. The result was
`beefabcd 0000ccdd` on gfx1100 (Radeon Pro W7900), and `beefabcd abcdccdd` on
gfx1201 (Radeon AI Pro R9700). The latter also matches the explicit missing
component rule in the RDNA4 ISA buffer-memory chapter. Earlier families use
the zero-fill behavior inferred from the older texture-unit store contract;
they were not independently measured in this run.

Regression tests cover all source widths, packed and ordinary sources, typed
and descriptor-selected formats, reversed selectors, and inactive lanes.
CDNA5 exposes no formatted-buffer instructions, so this policy makes no claim
about a CDNA5 formatted-buffer execution path.

An exhaustive native sweep of all 24 selector permutations, four store widths,
and packed/unpacked sources (192 cases per GPU) found that arbitrary DST_SEL
routing is not the inverse of load routing on either gfx1100 or gfx1201.
Repeated and constant selectors also diverge. Additional native probes covered
one-, two-, and three-channel UINT formats (8, 8_8, and 32_32_32), all source
widths and packed/unpacked sources. Four-channel straight/reversed orders and
narrow canonical R001/RG01/RGB1 patterns agree with inverse routing. Single-channel
identity, WZYX and A8 views also agree. In contrast, identity RGBA on an 8_8
format writes X/W, not X/Y, and narrow WZYX is not generally an inverse.
Untyped stores and the public packing helper share an exact qualification
predicate and reject unsupported selectors before writing. Typed stores use
their explicit identity mapping, separately confirmed on narrow native formats.
Other numeric formats and older families inherit the same channel-routing rule;
they were not independently measured. General store routing remains unqualified.

Cross-family formatted loads and stores require semantic expansion in DBT when
format encoding or the missing-channel policy differs. RDNA1 and RDNA2 have
distinct format tables. Until expansion exists, translation rejects these
accesses instead of silently re-encoding an inequivalent descriptor or opcode.
