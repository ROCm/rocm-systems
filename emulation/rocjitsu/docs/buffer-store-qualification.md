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
Repeated and constant selectors also diverge. Untyped formatted stores now
support the measured straight/reversed orders and report an unsupported operand
for other routing before issuing memory writes. Constants for components absent
from a narrow memory format do not affect the straight channel order. Typed
stores use their explicit identity mapping. General store routing remains
unqualified; reproducing inverse swizzling for arbitrary selectors is incorrect.

Cross-family formatted stores now require semantic expansion in DBT when format
encoding or the missing-channel policy differs. Until that expansion exists,
translation rejects them instead of silently re-encoding an inequivalent store.
