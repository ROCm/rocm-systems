# IB-MSEG: multi-segment registration in one forked plugin

This note describes the multi-segment PR stack. Instead of editing classic `net_ib` and `net_ib_cast` in place, multi-segment support goes into one separate plugin, IB-MSEG (`src/transport/net_ib_multiseg/`), a renamed copy of `net_ib` that `NCCL_MULTI_SEGMENT_REGISTER=1` selects. It covers why the fork exists, how it is selected, the six fork branches and their status, the copy and its rename map, the compile-twice host tests, the phases, how IB-CAST joins in phase 3, validation, and how the code can later move back into `net_ib` and `net_ib_cast`. Terms follow the [overview](multi_segment_overview.md#terms).

> [!NOTE]
> **Status (2026-11-23).** Forks 1 to 4 are committed locally on `users/atulkulk/netib-mseg/01` to `04`, on develop `c43b4a9e7307`, and are not pushed. They will be opened as a new PR stack that supersedes the earlier multi-segment PRs. Forks 5 and 6 are planned and have no branch. The source of truth for the plan is `~/ginocc/sw/oneplugin/plan_fork_net_ib_multiseg.md`; where this note and the fork branches differ from it, the branches win and the difference is listed under [open items](#open-items-and-differences-from-the-plan).

## Why a fork

The reviewers asked to keep multi-segment code out of production `net_ib` and `net_ib_cast`. With the fork, both stay at develop, so with the switch off nothing changes and there is no performance question for them. Inside the fork there is no compatibility machinery (capability negotiation, old-peer fallbacks) and no segmented copies of develop functions beside the originals, because the fork edits its own `ncclIbMultiSend`, flush, `iput` and `iget` in place.

What stays is what the hardware limit forces: a ROCm DMA-BUF export covers one physical segment, so a multi-segment buffer still needs one MR and one rkey per segment, the receiver still tells the sender its per-segment layout, and a transfer still posts more than one WR. Those pieces go away only when ROCm can export one DMA-BUF over a whole VA range; then develop `net_ib` handles such buffers unchanged and the fork can be retired.

The cost is one more directory to sync by hand: every upstream change to `net_ib` must be considered for the fork. The recorded rename map keeps that merge mechanical.

## Selection

```mermaid
%% caption: Which plugin a rank uses
flowchart TD
 req{"NCCL_NET or autodetect"} -->|"IB-CAST, ROCM-IB or AINIC"| cast["IB-CAST net plugin, develop code"]
 req -->|"IB or unset"| sw{"NCCL_MULTI_SEGMENT_REGISTER"}
 sw -->|0| ib["net_ib, develop code"]
 sw -->|1| mseg["IB-MSEG net plugin in the IB slot"]
 cast --> rma{"RMA backend"}
 mseg --> rma
 rma -->|"raw NCCL_NET is IB-CAST"| castrma["IB-CAST RMA backend"]
 rma -->|"switch on, otherwise"| msegrma["IB-MSEG RMA backend"]
 classDef netib fill:#e0f5f8,stroke:#008299,color:#1d1d1f
 classDef ibcast fill:#fff1e0,stroke:#ff9500,color:#1d1d1f
 classDef rma fill:#f5ebfb,stroke:#af52de,color:#1d1d1f
 class ib,mseg netib
 class cast,castrma ibcast
 class msegrma rma
```

- `plugin/net.cc` keeps IB-CAST first for `NCCL_NET=IB-CAST`, `ROCM-IB` and AINIC autodetect. Otherwise the switch registers `ncclNetIbMseg`, named `IB-MSEG`, in the IB slot, and a request for `IB` (from the environment or the comm's `netName`) matches it.
- `plugin/rma.cc` follows the same order. Under the switch it picks `IbMsegRmaIbProxy` unless the raw `NCCL_NET` is `IB-CAST`, so `ROCM-IB` and AINIC autodetect pair the IB-CAST net plugin with the IB-MSEG RMA backend until phase 3. `plugin/gin.cc` is unchanged, so IB-CAST keeps its GDAKI backend; multi-segment GIN runs over the proxy backend (GIN type 2).
- `NCCL_NET=IB-MSEG` also selects the fork without turning on multi-segment registration. It is a temporary developer selector, documented as such, and is removed without an alias at fold-back.
- Jobs and runner configs select only by the switch, so folding the code back later changes nothing for them.
- GDAKI is not copied. A multi-segment GIN window on a non-proxy backend keeps failing with `ncclInvalidUsage`; GDAKI support would be new kernel work (per-segment rkey lookup and slicing in the WQE builder), not a port.

## No edits to net_ib or net_ib_cast

No fork branch edits `src/transport/net_ib/`, `src/transport/net_ib_cast/`, or the sibling headers `net_ib_gid_inspect.h` and `net_ib_wqe_lat_inspect.h`. The fork gets their behaviour by copying and renaming, never by calling into them. Shared code outside those directories (core files, `misc/ibvwrap`, the GID helpers) may change. Each fork branch is checked with `git diff --stat <develop base> HEAD -- <those paths>`, which must be empty; it is empty at the fork 4 tip.

## Forks

```mermaid
%% caption: Fork branches, stacked on develop
flowchart TD
 dev["develop"] --> f1["fork 1: core gate"]
 f1 --> f2["fork 2: net_ib copy"]
 f2 --> f3["fork 3: registration and P2P"]
 f3 --> f4["fork 4: RMA and GIN"]
 f4 --> f5["fork 5: IB-CAST base, planned"]
 f5 --> f6["fork 6: IB-CAST multi-segment, planned"]
 dev --> pd["PD-reference fix, standalone"]
 classDef planned fill:#fff9e0,stroke:#ffcc00,color:#1d1d1f
 class f5,f6 planned
```

| Fork | Branch (`users/atulkulk/netib-mseg/...`) | Commits | Change over the parent (whole tree) |
|---|---|---|---|
| 1 | `01-core-backend-multiseg-gate` | `a0578eda57e6` code, `8d8b991c2e7c` tests, `3f5973a3fe00` owner-fd fix | 34 files, +1,612 / −40 over develop `c43b4a9e7307` |
| 2 | `02-net-ib-multiseg-copy` | `04b94a1636b6` copy, `9f125db92b47` tests | 61 files, +12,484 / −113 over fork 1 |
| 3 | `03-registration-p2p-multiseg` | `4b5968c3b120` code, `68042a159272` tests | 62 files, +13,061 / −131 over fork 2 |
| 4 | `04-rma-gin-multiseg` | `e15bec575504` code, `933c24ae63d5` tests, `568d71f9a2ea` review fixes, `9c67177e1f99` compile-twice host tests | 63 files, +9,457 / −5,301 over fork 3 (the compile-twice commit removes the hand-copied host tests) |
| 5 | none yet | planned, phase 3 | the develop `net_ib` to `net_ib_cast` delta, a few thousand lines |
| 6 | none yet | planned, phase 3 | CAST-specific multi-segment hooks |
| PD fix | `users/atulkulk/fix/ibcast-failed-pd-ref` | standalone branch from develop | the AICOMRCCL-3085 PD-reference fix |

All four committed forks are local and stacked on develop `c43b4a9e7307`; none is pushed.

### Fork 1: core gate

- Two comm-level flags, `comm->rcclMultiSegNet` and `comm->rcclMultiSegRma`, say whether the comm's net or RMA plugin registers one MR per physical segment. `rcclMultiSegOn(bool)` combines that with `NCCL_MULTI_SEGMENT_REGISTER` and `NCCL_DMABUF_ENABLE`. The flags are separate because AINIC with the switch on uses IB-CAST for net and IB-MSEG for RMA. `proxyState->rcclMultiSegNet` copies the net flag. The plugin ABI and plugin lists are unchanged, and nothing sets the flags in fork 1.
- The core names no plugin: `rcclRmaProxyMaxInflightRequests` reads the RMA flag instead of comparing the backend with a plugin. The RMA and GIN proxies cap requests in flight at 32 when it is set; the GIN proxy takes the flag through `rcclGinProxySetMultiSeg`, because `ncclGinProxyInit` runs inside `ncclRmaInit` before the flag exists.
- NET registration: a multi-segment buffer goes through `regMrDmaBuf` with fd -1 (one helper, `rcclNetRegMultiSeg`) only when the net flag is set, and stages otherwise. No new plugin pointer-type bit is added.
- Symmetric windows: RMA registers a multi-segment window only with the RMA flag, equal segment counts on every rank, and a segment table that fits; GIN only on the proxy backend. An RMA put on a window without an RMA registration returns `ncclInvalidUsage`, and hierarchical CE falls back when either window has none.
- Host owner fd for hybrid windows under `NCCL_SYM_REUSE_SYSMEM_HANDLES=1`: a rank that exported its own host handle uses its fd; otherwise it borrows the fd of the one rank that exported the segment, and with several exporters it borrows none. The decision counts exporters, not host-typed handles, because an imported handle reads as host or device depending on the ROCm version. Every rank agrees on the fetch result, also with GIN off; see [host owner DMA-BUF](multi_segment_gin_rma.md#host-owner-dma-buf).
- `TODO(ROCM-29810)` is removed: the runtime fix ships in 7.0.2.x and 10.3. The host-owner type propagation it described stays.
- Headers: `transport/rccl_mseg.h` (the per-registration cap `RCCL_IB_REG_SEG_DEFAULT`) and `rccl_mseg_rma.h` (segment-table fit, in-flight cap `RCCL_RMA_MAX_INFLIGHT_REQUESTS`). They are the only `rccl_mseg*` headers in `src/transport/` at the fork 4 tip.
- Changelog fragment `changelog.d/mseg-core-dmabuf-multiseg.changed.md`; its wording was amended on 2026-11-23 to "the single exporter's fd".

### Fork 2: net_ib copy

- `src/transport/net_ib_multiseg/` is develop `net_ib` without `gdaki/`, plus the two test-hook headers. The copy is the byte-identical, renamed output of a rename tool that is kept outside the stack on the side branch `users/atulkulk/netib-mseg/net-ib-multiseg-tools`.
- `tools/scripts/net_ib_multiseg/rename_map.txt` records the 395 renames: `ncclNetIb` becomes `ncclNetIbMseg`, other IB, RMA and GIN symbols get an `IbMseg` prefix, and `rcclNetP2pPolicy` becomes `IbMsegNetP2pPolicy`. Headers take a `_mseg` suffix, as IB-CAST uses `_cast`, because hipify renames repeated basenames. Parameter definitions become `extern` declarations, so both copies read the same `NCCL_IB_*` values. Like IB-CAST, the copy is plain code that later forks edit directly.
- `transport/net.cc` applies `IbMsegNetP2pPolicy` when the fork is the active net, as it does for `net_ib` and IB-CAST. RAS lists `IB-MSEG` as a verbs plugin.
- `RCCL_IB_GDR_FLUSH_FLAG` works on IB-MSEG: `init.cc` and `net.cc` reach the flag through `rcclIbGdrFlushOpsFor(net)` instead of comparing with `&ncclNetIb`. While the fork is selected, the RAS live link speed stays tied to `net_ib` and the active bandwidth diagnostic does not run, as for IB-CAST.
- Tests: plugin selection with the switch and with `NCCL_NET=IB-MSEG`, name matching, RMA backend choice, the RAS list and the P2P policy; new `*_mseg` runner jobs run the MPI whitebox suites on IB-MSEG on the Mellanox IB and Thor2 configs and in `net_ib_transport`.

### Fork 3: registration and P2P

- `init.cc` sets the net flag when the comm selected IB-MSEG and `rcclMultiSegOn()` holds; split comms resolve the same net there.
- Registration: one MR per segment through `seg_reg_mseg.h`, the handle sized from the walked count, up to `RCCL_IB_REG_SEG_DEFAULT` segments, and only while one more MR fits the device's `max_mr`.
- No capability bit. Both ends compute `IbMsegUseMultiSeg(switch and DMA-BUF, GIN, port failover)`; the GIN role rides in the listen handle. GIN and resilient comms keep the legacy CTS and decline a multi-segment receive registration, so those buffers stage.
- Mismatch check: each side puts its decision in the connect metadata (`useMultiSeg`). The receiver fails the accept and the sender fails the connect with `ncclInvalidUsage` when the two differ, instead of reading the CTS in the wrong layout.
- CTS: the receiver sends the overlapped segments in the CTS write and marks it with the layout bit (`NCCL_IB_CTS_NREQS_LAYOUT`), inline for up to four, otherwise as a reference the sender reads once into a cache. Single-segment receives keep the short CTS write. Each layout line carries the receive's index (the V1 stamp from the fork 3 review).
- Send: one WR fill builds every send, single-segment being the one-entry case; each write stays inside one local and one remote segment; a segmented send posts one slice per progress call; a partial post keeps or frees its request correctly; `isend` returns the post error.
- Flush: one read through the MR of the segment that holds the last receive's first byte; segmented sends write the GDR flush flag.
- Memory: with multi-segment on, the CTS slots add about 384 KiB on the send side and 512 KiB on the receive side.
- Code: `seg_*.h`, `seg_p2p.cc` and `p2p_multiseg.cc` inside the fork, with hooks in the copied files.
- Tests: four IB-MSEG host binaries (connect, GIN, P2P, registration), fixture tests for the segmented send, flush, hooks, split math and CTS layout, MPI suites on IB-MSEG, and runner jobs on the Mellanox IB and Thor2 configs and in `net_ib_transport`. AINIC configs get the jobs once IB-MSEG carries AINIC support.

### Fork 4: RMA and GIN

- `init.cc` sets `comm->rcclMultiSegRma` when IB-MSEG's RMA proxy is the comm's RMA backend and `rcclMultiSegOn()` holds; a child comm takes its parent's flag. `ginDevCommSetupWithBackend` hands each comm's flag to the GIN proxy just before it creates that comm's contexts, so a later comm cannot change an earlier comm's in-flight cap.
- Registration: one DMA-BUF MR per segment, up to `RCCL_IB_REG_SEG_DEFAULT` segments and within `max_mr` (`rcclIbMrAdmit`); every rank needs the same count, or registration fails on every rank with `ncclInvalidUsage` before the tables are gathered; ranks times segments is capped at 1,048,576 entries. Connect allocates the per-rank registration records before the first registration collective; close frees them.
- Data path: `iput`, `iget` and `iputSignal` split at segment boundaries and post in fixed-size rounds; the in-flight cap bounds the send queue; RMA QPs get `RCCL_RMA_SEND_QP_WRS` send WRs, and only GIN and RMA connects pass `nQpsPerDev`, which marks the comm as RMA. `iflush` reads one device segment, or nothing when every segment is host memory.
- Hybrid windows register host segments with the fd the core publishes by VA.
- `alloc.h`'s host-segment query is `rcclCuMemIsSysmemSegment` and releases the retained handle when the properties query fails.
- Fork-local helpers: `seg_export_mseg.h`, `seg_sym_mseg.h`, `seg_rma_mseg.h`, `seg_rma_round_mseg.h`.
- Runner jobs: RMA multi-segment (plus hybrid, 1x2 and 2x4 scale), RMA segment math and rounds, UBR multi-segment comprehensive and LSA plus GIN on `mi300x_mellanox_ib`, `mi300_mellanox_ib_func`, `mi355x_thor2_roce` and `mi355x_thor2_roce_func`; five RMA jobs in `net_ib_transport`. The AINIC configs have no IB-MSEG jobs yet, and the two hierarchical CE multi-segment cases (`CeMPI_AllGather` and `CeMPI_AlltoAll` `MultiNodeHierarchicalMultiSegment`, job `ce_hierarchical_multisegment_8node`) are not ported yet.
- The last commit replaces the hand-copied IB-MSEG host tests with the [compile-twice host tests](#compile-twice-host-tests).

### Fork 5: IB-CAST base (planned)

Phase 3 replays the develop diff from `net_ib` to `net_ib_cast` onto the fork: multiplane, QP sharing, CTS offload, the `ionicdv` helpers, AINIC recovery, BY_INDEX ordering, and the per-WR cap `IbCastMaxPostSendBytes`. It has no multi-segment logic and can be split in two if reviewers ask. It also adds the AINIC runner entries. The replay produces `rename_map_cast.txt` (CAST symbol to IB-MSEG symbol) and keeps CAST's internal names and struct shapes where possible, so CAST's develop host tests compile against the fork through the same compile-twice module, as a second base (`BASE net_ib_cast`). Where CAST and `net_ib` disagree on a name, the fork follows CAST and the `net_ib` base lists that test as `replaced`; CAST tests the fork cannot run yet start as `lagging`.

### Fork 6: IB-CAST multi-segment (planned)

Fork 6 adds only what is specific to IB-CAST, because the shared send, receive, flush and RMA paths are already in the fork: the BY_INDEX immediate flag, the scheduler interaction (one segmented group at a time), CAST's own CQ poll, `IbCastMaxPostSendBytes` over segmented P2P chains and multi-segment RMA rounds, and the IB-CAST RMA registration record with a magic and a version. After it, the switch selects the fork on AINIC too, with AINIC features. Three IB-CAST requirements are carried here: the multi-segment scope must exclude AINIC inline CTS (`IbCastAinicCtsInlineData`); the receiver must size its CTS slots from the peer's raw capability, not from `peerCaps` masked by its own scope; and the send-resume path must bound the peer-written `nreqs` to 1 to 8, as `IbCastIsend` does. See [review feedback](multi_segment_review_feedback.md#summary) and [IB-CAST P2P open items](multi_segment_netib_cast.md#open-items).

## The copy and the rename map

The fork 2 copy is develop `net_ib` with its global symbols, header names and include guards renamed, produced by a rename tool and committed unchanged; the tool is not part of the stack and nothing in the build or CI depends on it.

- `tools/scripts/net_ib_multiseg/rename_map.txt` records each `net_ib` global the copy renames and its fork name (395 entries). Reversing it turns a fork change into a patch against `net_ib`, which is what fold-back needs.
- The [compile-twice host tests](#compile-twice-host-tests) read the same file at configure time to build the alias header that points `net_ib` test code at the fork's symbols.
- Re-syncing the fork with a later develop `net_ib` is done with the rename tool from the side branch `users/atulkulk/netib-mseg/net-ib-multiseg-tools` until the decision whether to keep that tool in the tree or as a standalone tool is made.

## Compile-twice host tests

The IB-MSEG host binaries compile the `net_ib` host test files and fakes a second time, against the fork's units, so every `net_ib` host test keeps checking the fork without a hand-kept copy. The fork-only tests stay in the IB-MSEG test files, which include the `net_ib` ones. This replaced about 4,450 copied lines in fork 4.

- A generic module, `test/host/transport/derived_transport_tests.cmake`, handles any transport derived from one or more base transports. Per base, it generates at configure time a force-included alias header from the base's rename map, minus the names the hand-written seam units (`shim/connect.cc`, `gin.cc`, `p2p.cc`) alias back, and header shims that resolve the base's includes to the derived headers.
- Configure fails when a seam unit's `#define` names a rename-map entry on either side but does not reverse that entry. This check replaced a pytest.
- Per binary, `rccl_add_derived_transport_binary` takes an exclusion list whose entries are tagged `replaced` (the derived test file has its own test of that name) or `lagging` (a base change the fork has not taken yet), and a list of `DISABLED_` base tests it re-enables.
- Around the base test include, `TEST_F` registers each base test through `testing::RegisterTest` unless it is excluded, so the shared `main()` and `test/host/CMakeLists.txt` stay unchanged.
- A self-check suite in each binary fails on a stale or wrong exclusion; it was confirmed to fail on an injected stale entry and on a `replaced` entry with no fork test.
- The IB-MSEG configuration is `test/host/transport/net_ib_multiseg/host_tests.cmake`, included from `transport/host_tests.cmake`, with four binaries (connect, GIN, P2P, registration) on `BASE net_ib`. Today the P2P binary excludes eight tests as `replaced` (the fork's `ctsFifo` is a pointer) and one as `lagging` (`DISABLED_Iflush_ReadPostFailurePropagatesError`: the fork chains the flush write and read in one post), and the registration binary re-enables `DISABLED_RegMr_MultiDeviceFailureUnwindsEarlierRegistrations`.
- Developer rule: when a `net_ib` change breaks an IB-MSEG host test, add that test to its binary's `EXCLUDE_BASE_TESTS` as `lagging` with a reason; the fork owner ports the change and removes the entry. There is no Python step for developers.
- Fork 5 adds `BASE net_ib_cast` as one more base block in the same file, with its own seam list and exclusions.

```mermaid
%% caption: How one net_ib host test runs against the fork
flowchart TD
 base["net_ib test file and fakes"] --> alias["alias header from rename_map.txt"]
 alias --> shim["header shims to _mseg headers"]
 shim --> bin["IB-MSEG host binary"]
 excl{"excluded?"} -->|no| reg["RegisterTest"]
 excl -->|"replaced or lagging"| skip["not registered"]
 bin --> excl
 reg --> check["self-check suite"]
 skip --> check
```

## What the fork supersedes

| Note | In the fork |
|---|---|
| [Classic NET/IB P2P](multi_segment_netib_wire.md) | fork 3: no capability bit or old-peer fallback; the layout, slicing and one-read flush are in the fork |
| [GIN/RMA](multi_segment_gin_rma.md) | core part in fork 1 (flags instead of identity checks, owner-fd agreement), classic RMA in fork 4 |
| [IB-CAST P2P](multi_segment_netib_cast.md), [IB-CAST GIN/RMA](multi_segment_netib_cast_gin.md) | forks 5 and 6, planned |
| [Current contract](multi_segment_current.md), [code map](multi_segment_code_map.md) | the fork's footprint against `net_ib` is zero, and removal becomes deleting `net_ib_multiseg/` |
| `design_one_multiseg_plugin.md` (outside this book) | superseded by the plan for the near term |

## Phases

```mermaid
%% caption: Plan phases
flowchart TD
 p0["P0: replies, split the PD fix"] --> p1["P1: forks 1 to 4"]
 p1 --> p2["P2: validation and performance"]
 p2 --> p3["P3: IB-CAST into the fork, forks 5 and 6"]
 p3 --> p4["P4: refactor, fold back or retire"]
 classDef done fill:#e8f8ec,stroke:#34c759,color:#1d1d1f
 classDef planned fill:#fff9e0,stroke:#ffcc00,color:#1d1d1f
 class p1 done
 class p3,p4 planned
```

- **P0.** Post the drafted replies to Aleksandr's four review comments (not posted yet). Split the AICOMRCCL-3085 PD-reference fix into a standalone branch (done: `users/atulkulk/fix/ibcast-failed-pd-ref`).
- **P1.** Forks 1 to 4, as above: copy, rename, select, add per-segment registration, P2P, RMA and GIN to the fork in place, keep in the core only the segment walk and gate, the DMA-BUF path in `net.cc`, the owner fd table and `rccl_mseg.h`/`rccl_mseg_rma.h`, give the fork its own host tests, and make the jobs set the switch. Committed locally.
- **P2.** Validation on ROCm 7.0.2.1 build 55 and 10.3 on Mellanox IB (MI300X), Thor2 (MI355X) and AINIC, single and multi node, plus hostval with the gate-off build. Performance: switch off identical to develop; switch on with single-segment buffers at parity with develop `net_ib`; switch on with multi-segment buffers against develop staging at 2, 17, 64, 256 and 1024 segments; registration time and MR count. AINIC runs as plain RoCE until P3, and AINIC hardware is not reachable from the current cluster.
- **P3.** IB-CAST into the fork (forks 5 and 6, see the [consolidation plan](#consolidation-plan)).
- **P4.** Reduce the remaining protocol surface, merge upstream changes, and decide between keeping the fork, folding it back, or retiring it once ROCm DMA-BUF range export exists. The ROCm request should be filed in parallel with P1, because it is the only change that removes the per-segment design.

## Consolidation plan

The fork consolidates multi-segment support into one plugin in two steps. P1 touches classic only and leaves IB-CAST at develop. P3 brings IB-CAST's develop features into the fork as fork 5, then CAST's own multi-segment hooks as fork 6, instead of adding multi-segment to `net_ib_cast`. After P3 one plugin serves classic IB, RoCE and AINIC with multi-segment support, and one fork test suite covers both.

```mermaid
%% caption: From develop plus IB-MSEG to one plugin
flowchart TD
 subgraph p1s["after P1"]
 ib1["net_ib, develop"]
 cast1["net_ib_cast, develop"]
 m1["IB-MSEG, classic features"]
 end
 subgraph p3s["after P3"]
 ib3["net_ib, develop"]
 cast3["net_ib_cast, develop"]
 m3["IB-MSEG, classic and CAST features"]
 end
 p1s --> p3s
```

## Validation

| Run | Tree | Result | Status file |
|---|---|---|---|
| Fork 3 hardware | fork 3 before commit | 207 of 207 units on ROCm 7.0.2.2 and 10.3, two MI355X Thor2 nodes | `~/ginocc/sw/fork3-hw4/status` |
| Fork 4 hardware | fork 4 tests commit | 214 of 214 units on 7.0.2.1 build 55 and 10.3 | `~/ginocc/sw/fork4-hw3-b55/status` |
| Fork 4 hardware with review fixes | uncommitted review fixes | 212 of 214 on each version; both failures were the alloc test pinning the old handle leak, flipped since | `~/ginocc/sw/fork4-hw4-argus/status` |
| Fork 4 hostval | final fork 4 tip | pytest 167 passed; 7.0.2.1 build 55 and 10.3 0 failed binaries; gate-off build 197 binaries, 13,271 passed | `~/ginocc/sw/oneplugin/fork4/hv-final/status` |
| Fork 4 hardware, final tip | fork 4 `7f84f495265d`, before the fork 3 amend | 214 of 214 units on 7.0.2.1 build 55 and 10.3, 0 gtest failures | `~/ginocc/sw/fork4-hw5-final/status` |
| Fork 3 hostval after the amend | fork 3 `a36ed4043329` | pytest 166 passed; 7.0.2.1 build 55 and 10.3 0 failed binaries; gate-off build 197 binaries, 13,197 passed | `~/ginocc/sw/oneplugin/fork3/fix-hv/status` |
| Fork 4 hostval after the restack | fork 4 `21d89ca9d755` | pytest 167 passed; 7.0.2.1 build 55 and 10.3 0 failed binaries; gate-off build 197 binaries, 13,272 passed | `~/ginocc/sw/oneplugin/fork4/hv-vercheck/status` |
| Tool-split tips, hostval | fork 2 `9f125db92b47`, fork 3 `68042a159272`, fork 4 `9c67177e1f99` | rerun in progress | |

Forks 1 and 2 were validated with hostval (pytest, both ROCm versions, gate-off), and fork 2's `*_mseg` jobs ran in the stack-top hardware runs. Hardware has not run on the current fork 3 and fork 4 tips. The Mellanox IB configs have not run any fork yet.

## Open items and differences from the plan

- IB-CAST requirements carried to fork 6 (not fixed earlier either): exclude AINIC inline CTS (`IbCastAinicCtsInlineData`) from the scope; size the receiver's CTS slots from the peer's raw capability, not from `peerCaps` masked by its own scope; bound the peer-written `nreqs` on the send-resume path to 1 to 8, as `IbCastIsend` does.
- Replies to Aleksandr's four review comments are drafted but not posted. The three merged layer 01 threads (nusislam, alexander-sannikov, garzaran) have no reply.
- There is no switch-off latency and bandwidth comparison for single-segment sends yet (performance work, P2).
- The `reg.cc` unwind is covered only by develop's disabled AICOMRCCL-3112 test, by decision.
- The 2080-entry RMA send queue fits Thor2's `max_qp_wr` (4351); Mellanox has not been checked, and only a `static_assert` guards the size.
- Hardware has not run on the current fork 3 and fork 4 tips; the last full run was 214 of 214 units on fork 4 `7f84f495265d`, before the fork 3 mismatch-check amend. The Mellanox IB configs have not run any fork.
- AINIC runner entries come with fork 5. The hierarchical CE multi-segment tests (`MultiNodeHierarchicalMultiSegment`) are not ported yet, nor are two shared-math tests, `OffsetRangeOkRejectsPastEnd` and `RmaRoundTest.DropUnlinksTheMiddleTailAndHead`.
- The test_runner include (a shared job fragment for the near-identical configs) is a separate pending PR.
- Runtime tickets: ROCM-29812 (CPU access to imported host VMM; hybrid tests skip collectively without it), AIRUNTIME-2638 (`HostNuma` rejected), AIRUNTIME-2351 (the original single-segment export limitation). ROCM-29810 is fixed in the runtime and its TODO is removed in fork 1.

## Migration and fold-back

This part is separate from the plan and does not change it. IB-MSEG ships first behind the switch and is tested in the field. Pieces then move into production `net_ib` and `net_ib_cast` one at a time, each after field results support it, rather than in one large merge.

```mermaid
%% caption: Shared files and hooks, now and after fold-back
flowchart LR
 eng["shared multi-segment files"] --> hk["small hooks in the fork's copied files"]
 hk --> fork["net_ib_multiseg today"]
 eng -.->|same files| ib["net_ib later"]
 eng -.->|same files| cast["net_ib_cast later"]
 hk -.->|re-apply hooks| ib
 hk -.->|re-apply hooks| cast
```

Rules the fork follows, from the plan appendix:

1. Most logic lives in shared, transport-neutral files that take plain structs and callbacks, not in the copied files. (While the fork exists this logic is in `net_ib_multiseg/seg_*`, written without fork-specific symbols, and it moves to `src/transport/` at fold-back.)
2. The copied files change only at hook points: registration, CTS post, send, completion, flush, `iput`, `iget` and QP creation. The hook diff is tracked against develop's `net_ib` at the copy point, rendered through `rename_map.txt`; the target is a few hundred lines.
3. Single-segment buffers take develop's code unchanged: every hook first asks whether the buffer has more than one segment.
4. The rename is mechanical and recorded, so the reverse map turns the fork's hook diff back into a patch against `net_ib`. Fork 2 stays a pure rename.
5. The copy point is develop `c43b4a9e7307`'s `net_ib` (its last `net_ib` change is `138896d1c4ad`). Re-syncing with a later develop is done with the copy tool from the side branch `users/atulkulk/netib-mseg/net-ib-multiseg-tools`.
6. Decisions that depend on the switch go through one helper each (send-queue depth, in-flight cap); the core already checks the comm-level flags rather than a plugin name.
7. The connect check is designed to outlive the fork, so no capability negotiation or old-peer fallback has to come back. Fork 3 compares the two ends' `useMultiSeg` in the connect metadata and fails on a mismatch.
8. CAST features enter the fork as develop code, kept apart from the hooks; at merge time that replayed code is dropped and only the CAST hooks are applied to `net_ib_cast`.
9. Tests target the shared files and the hooks separately: unit tests call the shared files directly, and fixture tests reach the hooks through one CMake variable that names the transport's sources.

Order, each step gated on field results from the previous one:

```mermaid
%% caption: Fold-back order
stateDiagram-v2
 [*] --> Registration
 Registration --> ClassicP2P
 ClassicP2P --> RmaGin
 RmaGin --> IbCast
 IbCast --> Retire
 Retire --> [*]
```

1. Registration: the per-segment walk and `max_mr` admission move into `net_ib` behind the switch.
2. Classic P2P: the layout exchange, segmented send and flush hooks move into `net_ib`; with the switch on, `net.cc` selects production `net_ib` again for NET, and the fork remains only for RMA and GIN.
3. RMA and GIN: the RMA hooks move into the `net_ib` RMA backend, the `init.cc` flag check points at it, and `plugin/rma.cc` stops selecting the fork.
4. IB-CAST: the same hooks plus CAST's own move into `net_ib_cast` and `IbCastRmaIbProxy`.
5. Delete `net_ib_multiseg/` once no switch setting selects it, and drop `NCCL_NET=IB-MSEG` without an alias.

Each step is three actions per transport: keep the shared files where they are, apply that step's part of the hook diff (reverse-renamed), and move the selection in `net.cc` or `plugin/rma.cc`. Between steps the configuration that already works stays selectable, so a regression can be backed out by switching the selection back.
