# Coverage per architecture

Approximate coverage of library instructions used by Rocjitsu, counting each
mnemonic once with default modifiers and FP mode (not the entire GPU ISA):

## RDNA3

| Instructions | Loose | Bit-exact | FP status |
|---|---:|---:|---:|
| Matrix (WMMA) | 100% | 100% | N/A |
| FP arithmetic | 100% | 40% | 2% |
| Math (RCP, RSQ, SQRT, etc.) | 100% | 93% | 7% |
| Conversions | 100% | 30% | 0% |
| Dot products | 100% | 80% | 0% |
| Comparisons | 100% | 100% | 0% |
| Dual-issue arithmetic | 100% | 90% | 8% |
| Graphics/interpolation | 100% | 40% | 0% |
| Vector integer and register routing | 100% | 100% | N/A |
| Scalar instructions | 100% | 100% | N/A |

## RDNA4

| Instructions | Loose | Bit-exact | FP status |
|---|---:|---:|---:|
| Matrix (WMMA, SWMMAC) | 100% | 100% | N/A |
| FP arithmetic | 100% | 30% | 10% |
| Math (RCP, RSQ, SQRT, etc.) | 100% | 93% | 20% |
| Conversions | 100% | 40% | 0% |
| Dot products | 100% | 50% | 50% |
| Comparisons | 100% | 100% | 100% |
| Dual-issue arithmetic | 100% | 90% | 30% |
| Graphics/interpolation | 100% | 40% | 0% |
| Vector integer and register routing | 100% | 100% | N/A |
| Scalar instructions | 100% | 90% | 50% |

Percentages are rounded to the nearest 10%, or nearest 1% below 10% and above 90%.
Bit-exact coverage excludes loose fallbacks. FP status measures bit-exact
execution with reporting, counting only instructions that can produce FP status;
N/A means none in that family do. Support for nondefault modifiers and FP modes
varies by instruction.
