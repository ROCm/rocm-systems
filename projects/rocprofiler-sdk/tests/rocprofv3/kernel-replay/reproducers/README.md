# Kernel replay reproducers

One small reproducer per bug or suspected bug, each stating what a correct implementation
does and what was actually observed. None is wired into CTest; they are run by hand while
investigating.

Build the ones that need a compiler with `./build.sh --sdk-root /opt/rocm`. `R8` needs
nothing, and the `R5` shim needs no ROCm.

| | Bug | Status | Reproducer |
|---|---|---|---|
| R1 | `buffered-api-tracing` aborts on the finalize output path with `std::length_error` from `basic_string::_M_create` | **Observed in CI**, 1 of 4 OS images | `r1_r2_ci_failures.sh --only r1` |
| R2 | `rocprofv3-test-hip-streams-per-thread` segfaults, dragging three validate tests to Not Run | **Observed in CI**, 3/12 on gfx94X vs 1/35 on mi325 | `r1_r2_ci_failures.sh --only r2` |
| R3 | `hipFree` during a replay window, so `restore()` writes into freed device memory | Suspected from source | `r3_free_during_replay.cpp` |
| R4 | An indefinite replay loop has no bound and no watchdog, and holds the agent writer lock | Suspected from source | `r4_r7_bounded_and_csv.sh` |
| R5 | A failed snapshot or restore copy is warned about and ignored, so passes run against unreverted inputs | Suspected from source | `r5_copy_failure_shim.c` |
| R6 | A module-scope device variable over 1 GiB is silently skipped by the snapshot | Suspected from source | `r6_large_module_variable.cpp` |
| R7 | `Replay_Pass` reaches JSON but not `counter_collection.csv`, so passes are indistinguishable | Known gap | `r4_r7_bounded_and_csv.sh` |
| R8 | The counter tolerance is asymmetric: 10% under-reporting is caught, 10.5% over-reporting is not | **Reproduces, self-verifying** | `r8_tolerance_asymmetry.py` |

"Suspected from source" means the defect was identified by reading the implementation and the
reproducer has not yet been run on hardware. Those four could turn out to be handled
somewhere not accounted for, and the reproducer is how to find out.

## Notes on individual reproducers

**R1** is the highest-value one. The abort happens after `Outputting collected data to
api_buffered_trace.log...`, so it is on the finalize path, and `basic_string::_M_create`
throwing `length_error` means a string was built from a garbage length — a use-after-free or
uninitialised size rather than a timing flake. It appeared on ubuntu-22.04 only, after the
`finalize()` teardown order was restored to the original. Run the loop first; if it does not
reproduce under plain `ctest`, go straight to an ASAN build, which the script explains.

**R2** is intermittent and environment sensitive, which is why the script reports a rate with
a Wilson interval rather than a verdict. The two-proportion test between job families gives
p = 0.018, so run it on the image where it was seen.

**R3** is a race, so a single clean run is not evidence. The window between snapshot and
restore is short; raise `KR_REPRO_PASSES`, shorten the sleep in the freeing thread, and
prefer an ASAN build. The reproducer checks the kernel's own output for corruption as well as
looking for a fault, because the failure may be silent.

**R5** exercises a path that is otherwise unreachable without hardware faults. Small values
of `KR_FAIL_COPY_AFTER` hit the snapshot, larger ones hit a restore between passes; both are
worth trying, since a failed snapshot and a failed restore have different consequences.

**R6** needs the control to pass before the subject means anything. If the 300 MiB variable
also accumulates, module variables are not being restored at all and the run says nothing
about a size cap — the script checks for exactly that and exits 3.

**R8** is the only one that runs anywhere. It bisects the comparison to find each threshold
and asserts the asymmetric prediction, so it doubles as a regression test if the tolerance is
ever made symmetric.
