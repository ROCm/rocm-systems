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

# One Wave64 workgroup per gfx942 CU, to scale the committed working set.
./vgpr_saturation --counts 8,64,128,192,252 \
  --iterations 0,8 --waves 320 --samples 5
```

Runtime warmups occur before every point, but rocjitsu's demand-paged VGPR
storage discards a wave's physical pages when that wave retires. Consequently,
every timed launch includes the page-commit cost for its live register set.
Increasing `--iterations` amortizes that launch-fixed cost over more accesses
to the already resident VGPRs. Comparing an unchanged baseline build with a
lazy-storage build across both axes answers two separate questions:

- whether the demand-paging cost is primarily fixed per dispatch;
- whether that fixed cost grows with the number of touched VGPR pages.

Use `--csv` for paired analysis. Wall-clock timing covers kernel launch through
`hipDeviceSynchronize`; output copies and checksum verification are excluded.
