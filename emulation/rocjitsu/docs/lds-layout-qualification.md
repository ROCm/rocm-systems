# LDS layout qualification

CDNA4 B4 and B6 transpose loads use contiguous 16-lane matrices. CDNA5 uses
interleaved source groups. These must have separate transpose kinds.

The CDNA4 B4 mapping agrees with Triton's `TargetFeatures::queryLDSTransLoadParams`
and `lowerDsReadTr` at revision `faafaf91de0bf424cf38133a9bdfbc4d8315a130`:
register bits precede the remaining lane bits in the address-lane basis.
Both CDNA4 B4 and B6 mappings are independently specified by
[ROCm FlyDSL's CDNA4 CopyAtom](https://github.com/ROCm/FlyDSL/blob/1f3511677662afc58bcf141dec812ccf05334fd8/lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp):
source lane strides are 64/96 bits and destination lane strides are 4/6 bits,
with the element index stepping by the source lane stride. This is a packed
16 by 16 transpose within each 16-lane group. Regression tests move every bit,
including B6 fields crossing DWORD boundaries. Native CDNA4 hardware was not
available for a separate measurement.

ADDTID's pre-CDNA5 M0-plus-instruction-offset addition wraps to sixteen bits
before the lane offset is added, measured on gfx1100 and gfx1201 in wave32 and
wave64. CDNA5 extracts twenty M0 bits and then adds the instruction and lane
offsets without wrapping, per its ISA pseudocode. Earlier-family behavior is
qualified by the published formula and the available RDNA measurements.

MFMA partial-EXEC tests apply only to CDNA3/4, whose manuals explicitly force
all lanes active. CDNA1/2 accumulator-bank tests use full EXEC.
