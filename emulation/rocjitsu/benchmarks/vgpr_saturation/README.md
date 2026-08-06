# VGPR saturation benchmark

This standalone HIP program measures rocjitsu's behavior as a Wave64 kernel's
live VGPR set approaches the architectural limit. It is intentionally not a
CTest: the results are performance observations rather than pass/fail gates.

The kernel keeps 8 through 252 independent values live across a runtime-sized
update loop and consumes every value in an observable checksum. On gfx942, the
compiler reports 9 through 253 VGPRs respectively, with no VGPR spills. The
program prints the actual value returned by `hipFuncGetAttributes` so a changed
compiler cannot silently invalidate the experiment. After the timed launches,
the program copies and checks every output lane from the final launch.

## Reference measurements

These historical measurements record the motivation and observed cost of the
demand-paged representation.

- Date: 2026-08-06
- Host: AMD EPYC 9554, Linux 6.8.0-136-generic
- Compiler: AMD clang 23.0.0git, LLVM revision
  `2abe93d58c833c804914bed3ffcebb3a6a01e237`
- Baseline: `bee28e14a8ec31916c228e9a1cb29461db1a0387`
- Configuration: `configs/gfx942_cdna3_kmd.json`, Wave64, 320 simulated CUs

The shipped configuration creates 8 XCDs, each with 4 shader engines and 10
CUs, for 320 CUs total. The simulator models each CU with 32 wavefront slots,
and the former representation eagerly allocated and zero-initialized the entire
VGPR backing. It therefore allocated:

```text
320 CUs x 32 slots x 512 VGPRs x 64 lanes x 4 bytes = 1,280 MiB
```

The fixed-topology result is the median of six CPU-pinned process runs that
loaded the gfx942 configuration without launching a kernel:

| Measurement       |        Baseline | Lazy storage | Change     |
|-------------------|----------------:|-------------:|------------|
| Peak RSS          |   1,511,424 KiB |  198,656 KiB | -86.9%     |
| Process wall time |          0.86 s |       0.13 s | -84.9%     |

The representative single-queue saturation run used 320 Wave64 workgroups per
launch. rocjitsu assigns one HIP hardware queue to one XCD command processor,
so this places eight waves on each of that XCD's 40 CUs. Each launch was timed
through `hipDeviceSynchronize`; values are paired medians with five timed
launches and one warmup per point. One update pass traverses every live
VGPR-backed value once and applies the benchmark's XOR-multiply-add recurrence.

| 253-VGPR launch       | Baseline median | Lazy median | Change |
|-----------------------|----------------:|------------:|-------:|
| Initialization only   |      149.162 ms |  161.507 ms | +8.28% |
| Eight update passes   |    1,257.192 ms | 1,262.361 ms | +0.41% |

Across the complete single-queue sweep, peak RSS fell from 1,592,320 KiB to
299,676 KiB, an 81.2% reduction. The 320 simultaneously resident waves touch
about 19.8 MiB of VGPR backing, versus the 1,280 MiB eagerly allocated for the
whole simulated device. The measurements were collected with `taskset -c 8`;
the commands below reproduce the benchmark shape.

An intentionally hardware-unrealistic upper-bound run used eight HIP streams
and `GPU_MAX_HW_QUEUES=8` to assign one hardware queue to each XCD. Each stream
launched 1,280 waves: 40 CUs times all 32 simulator slots per CU. The resulting
10,240 simultaneous waves each used 253 VGPRs. Eight update passes kept their
initialized register pages live across rocjitsu's functional scheduling
quantum. Values are medians of three timed launches with no warmup, and every
output lane was checked.

| All-slot 253-VGPR run |        Baseline |    Lazy storage | Change     |
|-----------------------|----------------:|----------------:|------------|
| Median launch time    |        47.678 s |        46.955 s | -1.5%      |
| Peak RSS              | 1,620,992 KiB   |   947,352 KiB   | -41.6%     |

This is a simulator storage stress, not an achievable hardware occupancy for a
253-VGPR wave. rocjitsu currently admits waves according to its 32 fixed slots
without reducing occupancy as register use rises. The lazy build still saves
673,640 KiB because this pure-VGPR kernel touches 253 of the 512 combined
VGPR/AGPR registers backed for each slot; pages for the unused half remain
uncommitted.

Build with the same ROCm SDK used for rocjitsu:

```bash
ROCM_PATH=/path/to/rocm \
  /path/to/amdclang++ -O3 --offload-arch=gfx942 \
  --rocm-path=/path/to/rocm vgpr_saturation.hip -o vgpr_saturation
```

Before collecting timings, verify that the compiler still emits the intended
resource profile. This standalone check compiles GPU assembly and requires each
kernel to use `Count + 1` VGPRs, zero VGPR spills, zero private-segment bytes,
and zero AGPRs:

```bash
ROCM_PATH=/path/to/rocm ./verify_resources.py --arch gfx942
```

The supported count list is shared by the benchmark and verifier through
`vgpr_saturation_counts.def`, so adding or removing a kernel updates both.

Run through rocjitsu:

```bash
/usr/bin/time -v /path/to/rocjitsu \
  --config ../../configs/gfx942_cdna3_kmd.json -- \
  ./vgpr_saturation --csv
```

Useful focused sweeps:

```bash
# One-wave scaling, including an initialization-only point.
./vgpr_saturation --counts 8,64,128,192,252 \
  --iterations 0,1,8,64 --samples 9

# Representative single-queue concurrency: eight waves per CU on one XCD.
./vgpr_saturation --counts 8,64,128,192,252 \
  --iterations 0,8 --waves 320 --samples 5

# Deliberate simulator upper bound: all 32 slots on all 320 configured CUs.
GPU_MAX_HW_QUEUES=8 ./vgpr_saturation --counts 252 --iterations 8 \
  --waves 1280 --streams 8 --samples 3 --warmups 0
```

Configured runtime warmups occur before every point, but rocjitsu's demand-paged
VGPR storage discards a wave's physical pages when that wave retires.
Consequently, every timed launch includes the page-commit cost for its live
register set.
Increasing `--iterations` amortizes that launch-fixed cost over more accesses
to the already resident VGPRs. Comparing an unchanged baseline build with a
lazy-storage build across both axes answers two separate questions:

- whether the demand-paging cost is primarily fixed per dispatch;
- whether that fixed cost grows with the number of touched VGPR pages.

Use `--csv` for paired analysis. Wall-clock timing covers kernel launch through
`hipDeviceSynchronize`; output copies and checksum verification are excluded.
