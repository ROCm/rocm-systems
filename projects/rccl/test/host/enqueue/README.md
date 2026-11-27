# Enqueue host tests

`rccl-UnitTestsMicroEnqueue` tests `src/enqueue/enqueue.cc` on the host, with no GPU.
The tests call `static` functions in `enqueue.cc`, so `enqueue-test.cc` includes
`enqueue.cc` and every test compiles into that one translation unit. The tests are
split across several files so that two pull requests that add tests to different
areas edit different files.

## Files

| File | Contents |
|---|---|
| `enqueue-test.cc` | The harness: the device-header shim, the `#include` of `enqueue.cc`, the `EnqueueMicrotest` fixture, then the two includes below. |
| `enqueue-test-scenes.inc` | Scenes and helpers that more than one test file uses. |
| `enqueue-test.d/*.inc` | The tests, one file per functional area of `enqueue.cc`. |
| `host_tests.cmake` | The enqueue binaries. It globs `enqueue-test.d/*.inc` into a generated include list. |

`enqueue-test.d/`:

| File | Unit under test |
|---|---|
| `algo-cost-table.inc` | `ncclGetCollNetSupport`, `initCollCostTable`, `updateCollCostTable` |
| `algo-info.inc` | `topoGetAlgoInfo`, `getAlgoInfo`, and the RCCL tuning hooks they call |
| `coll-chunking.inc` | `calcCollChunking`, including tuner chunk sizes and net registration |
| `comm-check.inc` | `ncclEnqueueCommCheck` |
| `finish-plan.inc` | `finishPlan`, `waitWorkFifoAvailable` |
| `kernel-config.inc` | `ncclFuncTrafficPerByte`, `ncclTestBudget`, shared-memory sizing, `rcclKernelPackedChannels`, `ncclPlanSetDefaultKernel` |
| `launch.inc` | `rcclAddonLaunchBegin`/`End`, addon launches under graph capture, `ncclLaunchFinish` under capture, `getImplicitOrder` |
| `p2p-add-to-plan.inc` | `addP2pToPlan`: protocol selection, CE routing, LL thresholds |
| `p2p-policy.inc` | `rcclP2pTaskParts`, `rcclP2pPolicyChannels`, `calcP2pChannelCount`, `rcclEffectiveP2pBatchEnable`, `rcclP2pBatchEligible` |
| `p2p-schedule.inc` | `scheduleP2pTasksToPlan`: self pairs and per-round argument budgets |
| `prepare-tasks.inc` | `ncclPrepareTasks`, including `rcclSimulateDirectAllGather` for `ncclGroupSimulateEnd` |
| `proxy-ops.inc` | `geteActivationMask`, `gettaskEventHandle`, `addProxyOpIfNeeded` |
| `redop.inc` | `hostToDevRedOp`, `ncclRedOpCreatePreMulSum_impl`, `ncclRedOpDestroy_impl` |
| `task-append.inc` | `collTaskAppend`, `ceCollTaskAppend`, `rawTaskAppend`, `ncclCeInitJob`, `rmaTaskAppend`, `ncclPlannerSetCapturingGraph` |
| `tasks-reg-enqueue.inc` | `ncclTasksRegAndEnqueue` |
| `work-batch.inc` | `addWorkBatchToPlan` |

## Adding a test

Put the test in the file for the unit it tests. Put a helper that only that file uses in
the same file, inside an anonymous namespace.

A test file must not use a helper defined in another test file. CMake includes the files
in name order, so such a test would compile or not depending on the file names. If a
second file needs a helper, move the helper to `enqueue-test-scenes.inc`.

Every test file shares one anonymous namespace, so two helpers with the same name are a
compile error. Name a helper after what it builds (for example `Gfx1250AddP2pScene`)
rather than with a generic name such as `Scene`.

Add a file to `enqueue-test.d/` when a new area of `enqueue.cc` gets tests, or when a file
covers two areas that each change often. Name it after the area, not the ticket. Start it
with the copyright header and a one-line comment that names the units it covers. The next
build includes the file: `CONFIGURE_DEPENDS` re-runs the glob, so neither
`host_tests.cmake` nor `enqueue-test.cc` changes.

New tests use the `EnqueueMicrotest` fixture, so
`test/test_categories_micro_enqueue.yaml` already selects them.

## Building and running

The host tests compile against the hipified sources, so stage those first. From
`projects/rccl/test/host`:

```sh
bash run_host_tests.sh rccl-configure
bash run_host_tests.sh hipify
bash run_host_tests.sh configure
cmake --build build --target rccl-UnitTestsMicroEnqueue rccl-UnitTestsMicroEnqueue-warpspeed \
  rccl-UnitTestsMicroEnqueue-devlinker
./build/rccl-UnitTestsMicroEnqueue --gtest_shuffle
```

The three binaries compile the same tests. `-warpspeed` adds `ENABLE_WARP_SPEED`, and
`-devlinker` adds `RCCL_DEVICE_LINKER`, which is the configuration RCCL ships. The
standalone `test/host` build above makes all three. A build inside RCCL with
`BUILD_TESTS=ON` makes only the first two. Run all three when a test depends on
either option.
