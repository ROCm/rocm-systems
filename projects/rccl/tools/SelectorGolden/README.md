# SelectorGolden

Records which backend RCCL picks for every point of a grid of collective calls,
so the selector redesign can prove it makes the same choices as `develop`.

This is phase 0 of the selector redesign: first freeze today's behavior, then
move the selection code behind one scheduler and diff against the frozen copy
after every step. Nothing here changes tuning.

## What it records

For every combination of

| Axis | Values |
| --- | --- |
| collective | AllReduce, AllGather, ReduceScatter, AlltoAll, Broadcast |
| dtype | float, half, bfloat16 by default (fp8e4m3, fp8e5m2, double, int32 on request) |
| op | sum by default (prod, max, min, avg on request; reductions only) |
| count | each power of two in the count argument's bytes and in the whole buffer's bytes, one element either side, and the midpoint |
| buffer registration | R0 (hipMalloc), R1 (ncclMemAlloc + ncclCommRegister), R2 (ncclMemAlloc + symmetric window), matching rccl-tests `-R` |
| in-place | out-of-place and in-place, with rccl-tests pointer offsets |
| graph capture | off and on (passed as the capture hint) |

it calls `rcclGetCollImplInfo` and writes one CSV row with the algorithm id and
name, protocol, channel count, status, and whether every rank agreed. The file
header records the RCCL version, arch, rank and host counts, CTA policy and every
`NCCL_*` / `RCCL_*` variable that was set, so two files from different setups are
not compared by mistake.

Every rank queries the whole grid and rank 0 checks they all agree, because a
backend chosen on some ranks and not others is a hang, and some inputs (in-place
offsets, alignment) depend on the rank.

## Build and run (needs an MI455 or other ROCm node)

```bash
cd projects/rccl/tools/SelectorGolden
make RCCL_DIR=../../build/release MPI_DIR=/opt/ompi

# one node, 8 ranks, default table and CTA policy
mpirun -np 8 ./SelectorGolden -f develop.csv

# the matrix the redesign has to hold: SPX and DPX tables x CTA policy 0 and 2
./run_matrix.sh out/develop -- mpirun -np 8
```

For multi-node, launch across hosts as usual (`-H a:8,b:8 ...`). Rows depend on
the rank count and host count, so record a baseline for each shape that matters
(for example 8, 2x8, 4x8 and a full rack).

## Compare

```bash
./golden_diff.py out/develop/golden_SPX_cta0.csv out/new/golden_SPX_cta0.csv
```

Exit code 0 means identical decisions, 1 means some changed, 2 means the files
came from different setups. Changes are folded into size ranges per slice, for
example

```
AllReduce float sum R2 inplace=0 capture=1  bytes 8M..64M  (14 pts)  <name>(<id>)/LL128/ch8 -> <name>(<id>)/LL/ch16
```

`--ignore-channels` compares backend and protocol only.

## Limits

- It checks the query path (`query=true`). Live dispatch is meant to agree with
  it and `test/CollImplInfoTests.cpp` checks that; this tool does not.
- Calls are made outside a group, so paths that need `ncclGroupDepth == 0`
  (GIN SDMA, CE 2-shot) are seen as eligible. Grouped calls are not covered.
- Backends that set up lazily (CE staging, DDA scratch) are warmed up with one
  small and one large call of each collective first. `--no-warmup` records the
  first-call behavior instead.
- Environment overrides other than the partition table and CTA policy are not
  swept. Set them on the launcher if a run needs them; they land in the header.
- Reduce, Gather, Scatter and AlltoAllv are not covered yet.
