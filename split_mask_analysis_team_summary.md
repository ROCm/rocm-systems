**Subject: RCCL SPLIT_MASK + PCIe Topology Analysis — MI300X 8-GPU / 4-NIC (BD Customer)**

Hi team,

Sharing the analysis spec for the BD customer performance issue on SMC300x MI300X nodes running data-parallel workloads with `NCCL_TESTS_SPLIT_MASK=0x7` and 4 NICs (masked from 8). Full spec is attached.

**Root Cause**

The 4-NIC mask removes one NIC per physical Broadcom PEX89104 switch, forcing half the GPUs to reach a NIC on the other partition of their physical switch. RCCL lacks visibility into the inter-partition fabric, so it cannot distinguish cross-partition paths (same physical switch, internal fabric) from cross-switch paths (different physical switch, through CPU) — both get classified as PATH_PHB, disabling GDR for 4 of 8 GPUs.

**What Was Analyzed**

- **SPLIT_MASK mechanics**: How `mask=0x7` creates 8 independent 2-rank communicators from 16 ranks
- **RCCL path computation**: BFS algorithm, PATH_PXB vs PATH_PHB classification rules, and why cross-partition paths fall to PATH_PHB without inter-switch visibility
- **BCM switch flattening**: Confirmed `ncclTopoFlattenBcmSwitches()` does NOT apply on SMC300x due to `subsystem_vendor` mismatch (Supermicro `0x15d9` vs expected Broadcom `0x1000`)
- **Virtual switch link mechanism**: How the `switch_discovery` kernel module + RCCL's pcilink support (PR #3121, already in latest RCCL) creates PCI-to-PCI edges that bypass the CPU in topology graph
- **NIC selection logic**: Full walkthrough of `ncclTopoGetLocalNet()` and `mirrorBits()` for 3 scenarios — 8-NIC, 4-NIC without switch links, 4-NIC with switch links
- **Physical PCIe topology**: DSN-based switch membership mapping on the actual SMC300x hardware
- **Privilege constraints**: Verified that PCIe extended config space (DSN at offset 0x100) requires root/`CAP_SYS_RAWIO` — no unprivileged workaround exists

**The Fix (with switch_discovery)**

When inter-partition links are visible, cross-partition paths become PATH_PXB instead of PATH_PHB. Every GPU picks a NIC on its own physical switch and GDR is enabled for all 8 GPUs.

**Approaches for Further Exploration**

The `switch_discovery` kernel module is the intended solution but may not always be deployable. We analyzed 4 alternative approaches to populate the same sysfs interface (`/sys/kernel/pci_switch_link/virtual_switch_links/`) that RCCL already reads — requiring zero RCCL code changes:

```
┌─────┬────────────────────────────────────┬─────────────────────┬─────────────────────┬──────────────────────┐
│  #  │ Approach                           │ Kernel Module?      │ RCCL Changes?       │ Best For             │
├─────┼────────────────────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ 1   │ Privileged systemd service +       │ Thin module         │ None                │ Short-term           │
│     │ minimal sysfs-writer module        │ (no VSEC)           │                     │ deployment           │
├─────┼────────────────────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ 2   │ Capability-restricted helper       │ No                  │ Minor               │ Prototyping /        │
│     │ binary at NIC bring-up             │                     │ (fallback path)     │ validation           │
├─────┼────────────────────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ 3   │ udev rule triggered on PCIe        │ No                  │ Minor               │ Automated            │
│     │ enumeration                        │                     │                     │ deployments          │
├─────┼────────────────────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ 4   │ Simplified DSN-only kernel module  │ Yes                 │ None                │ Medium-term,         │
│     │                                    │ (vendor-agnostic)   │                     │ cross-vendor         │
└─────┴────────────────────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘
```

Approach 2 has a direct precedent — AINIC already runs `disable_acs.sh` (privileged `setpci` operations) at NIC bring-up, so a DSN mapper helper fits naturally into that same workflow.

**Recommendation**: Start with Approach 2 to validate DSN-based switch grouping on SMC300x, then move to Approach 4 (vendor-agnostic DSN-only kernel module) for production.

---

**UPDATE: Experimental Validation Complete (2026-10-10, corrected 2026-10-10)**

We implemented and validated Approach 2 end-to-end on the SMC300x cluster. The fix consists of two components:

> **Correction note**: Earlier results were generated from experiments with a typo in
> `run-rccl.sh` (`${BD_EXPERIMENT_FLAG}` missing trailing `S`) that caused
> `NCCL_MAX_NCHANNELS=16` to not be applied. The data below is from corrected
> experiments with `NCCL_MAX_NCHANNELS=16` properly set. With the channel cap
> applied, before/after produce identical saturated throughput — the previously
> reported +13% before-fix advantage was entirely a typo artifact.

**1. DSN Mapper Script (`rccl_dsn_mapper.sh`)**
- Reads PCIe Device Serial Number (DSN) from Broadcom PEX89104 upstream ports via `setpci`
- Groups ports sharing the same DSN (= same physical switch, different partitions)
- Creates bidirectional peer link entries at `/var/run/rccl_bcm_links/<busid>/<peer_busid>`
- Runs as `sudo` at NIC bring-up (same workflow as `disable_acs.sh`)
- Discovered 4 multi-partition switch groups + 1 solo group per node (8 bidirectional links total)

**2. RCCL Patch (`src/os/linux.cc:ncclOsGetBcmLinks`)**
- Adds conditional `RCCL_BCM_LINKS_PATH` env var fallback
- Only activates when the default sysfs path `/sys/kernel/pci_switch_link/virtual_switch_links` doesn't exist
- Does NOT override `switch_discovery` kernel module if present
- Uses `stat()` + `S_ISDIR()` check — no new dependencies

**Results: PATH Selection — Before vs After**

```
┌──────┬──────────┬─────────────────────────┬───────────────────────────┐
│ Rank │ GPU BDF  │ Before (no inter-link)  │ After (DSN mapper)        │
├──────┼──────────┼─────────────────────────┼───────────────────────────┤
│ 0    │ 05:00.0  │ PXB, GDR ON,  1 NIC    │ PXB, GDR ON,  1 NIC      │
│ 1    │ 29:00.0  │ PHB, GDR OFF, 2 NICs * │ PXB, GDR ON,  1 NIC  ✓   │
│ 2    │ 49:00.0  │ PHB, GDR OFF, 2 NICs * │ PXB, GDR ON,  1 NIC  ✓   │
│ 3    │ 65:00.0  │ PXB, GDR ON,  1 NIC    │ PXB, GDR ON,  1 NIC      │
│ 4    │ 85:00.0  │ PXB, GDR ON,  1 NIC    │ PXB, GDR ON,  1 NIC      │
│ 5    │ a9:00.0  │ PHB, GDR OFF, 2 NICs * │ PXB, GDR ON,  1 NIC  ✓   │
│ 6    │ c9:00.0  │ PHB, GDR OFF, 2 NICs * │ PXB, GDR ON,  1 NIC  ✓   │
│ 7    │ e5:00.0  │ PXB, GDR ON,  1 NIC    │ PXB, GDR ON,  1 NIC      │
└──────┴──────────┴─────────────────────────┴───────────────────────────┘
* = dual-NIC, no GDRDMA, 16 channels (8/NIC), 33 proxy connections
✓ = fixed — single NIC, GDRDMA, 16 channels, 33 proxy connections
```

**GDR Coverage**: 50% (4/8) → **100% (8/8)**

**Results: Transport Comparison** (with NCCL_MAX_NCHANNELS=16)

```
┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Metric          │ Before (no inter-link)       │ After (DSN mapper)           │
├─────────────────┼──────────────────────────────┼──────────────────────────────┤
│ GDR ON          │ 4 of 8 GPUs (50%)            │ 8 of 8 GPUs (100%)          │
│ NICs per rank   │ 1 (GDR-ON), 2 (GDR-OFF)     │ 1 (uniform)                 │
│ Channels        │ 16 (all ranks)               │ 16 (all ranks)              │
│ Proxy conns     │ 33 (all ranks)               │ 33 (all ranks)              │
│ Transport       │ GDRDMA / plain NET mix       │ GDRDMA (all channels)       │
│ NIC ch distrib  │ 16/NIC (GDR-ON), 8/NIC (OFF)│ 16/NIC (uniform)            │
└─────────────────┴──────────────────────────────┴──────────────────────────────┘
```

**Results: RCCL busBw Comparison (out-of-place, GB/s, NCCL_MAX_NCHANNELS=16)**

```
┌───────────┬──────────────────────────┬──────────────────────────┬───────────┐
│ Msg Size  │ Before: AR / A2A         │ After: AR / A2A          │ Delta     │
├───────────┼──────────────────────────┼──────────────────────────┼───────────┤
│ 1M        │  9.37 /  8.41           │ 13.49 /  8.87            │ +44/+5%   │
│ 4M        │  9.99 / 16.23           │ 16.25 / 14.07            │ +63/−13%  │
│ 8M        │ 11.05 / 19.36           │ 18.44 / 17.56            │ +67/−9%   │
│ 64M       │ 19.84 / 23.49           │ 16.64 / 22.47            │ −16/−4%   │
│ 128M      │ 24.54 / 23.52           │ 23.90 / 22.75            │  −3/−3%   │
│ 256M      │ 24.47 / 23.82           │ 24.02 / 22.87            │  −2/−4%   │
│ 512M      │ 24.47 / 23.93           │ 24.11 / 23.11            │  −1/−3%   │
│ 1G        │ 24.46 / 23.81           │ 24.30 / 23.14            │  −1/−3%   │
│ 2G        │ 24.46 / 23.58           │ 24.33 / 23.21            │   0/−2%   │
│ 4G        │ 24.46 / 23.47           │ 24.35 / 23.27            │   0/−1%   │
│ 8G        │ 24.46 / 23.62           │ 24.40 / 23.30            │   0/−1%   │
│ 16G       │ 24.47 / 23.62           │ 24.39 / 23.30            │   0/−1%   │
└───────────┴──────────────────────────┴──────────────────────────┴───────────┘
```

**Performance Analysis**:
- **all_reduce mid-range (1M–8M)**: +44% to +67% improvement — uniform GDRDMA eliminates CPU bounce latency at these sizes
- **all_reduce saturated (1G+)**: ~24.4 GB/s both before and after — identical throughput when channel cap is properly applied
- **all_reduce 16M–64M anomaly**: Before-fix ~20 GB/s vs after-fix ~16 GB/s in the LL128→Simple protocol transition region; mixed GDR-ON/GDR-OFF transport handles this transition differently
- **alltoall**: Small before-fix advantage (~3% at saturated sizes, 23.62 vs 23.30 GB/s) — host-bounced paths may reduce PCIe switch contention in point-to-point patterns
- **Primary win is architectural correctness**: uniform GDR, uniform NIC assignment, uniform transport, simplified tuning

**Results: GDR Verification — All 10 Collectives (NCCL_DEBUG=info, 1G msg)**

Ran all 10 collectives with INFO-level logging to verify GDR ON/OFF per GPU:

```
┌─────────────────┬────────────────────────────────┬────────────────────────────────┬────────────┐
│ Collective      │ Before-Fix GDR ON              │ After-Fix GDR ON               │ Fix Effect │
├─────────────────┼────────────────────────────────┼────────────────────────────────┼────────────┤
│ all_reduce      │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ alltoall        │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ alltoallv       │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ broadcast       │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ reduce_scatter  │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ all_gather      │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ reduce          │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ scatter         │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ gather          │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
│ sendrecv        │ 4/8 ( 50%) GPU[0,3,4,7]        │ 8/8 (100%) GPU[0,1,2,3,4,5,6,7] │    +4 GPUs │
└─────────────────┴────────────────────────────────┴────────────────────────────────┴────────────┘
```

GDR pattern is **identical across all 10 collectives** — before-fix GPUs 1,2,5,6 (cross-partition) always get PATH_PHB / GDR OFF / 2 NICs / plain NET; after-fix all 8 GPUs get PATH_PXB / GDR ON / 1 NIC / GDRDMA.

NIC assignment change for the 4 affected GPUs:

```
┌─────┬──────────┬────────────────────────────┬────────────────────────────┐
│ GPU │ GPU BDF  │ Before NICs                │ After NICs                 │
├─────┼──────────┼────────────────────────────┼────────────────────────────┤
│  1  │ 29:00.0  │ 2 NIC: ionic_0, ionic_2    │ 1 NIC: ionic_0             │
│  2  │ 49:00.0  │ 2 NIC: ionic_0, ionic_2    │ 1 NIC: ionic_2             │
│  5  │ a9:00.0  │ 2 NIC: ionic_4, ionic_6    │ 1 NIC: ionic_4             │
│  6  │ c9:00.0  │ 2 NIC: ionic_4, ionic_6    │ 1 NIC: ionic_6             │
└─────┴──────────┴────────────────────────────┴────────────────────────────┘
```

**Deployment Path**

1. **Immediate** (validated): Run `rccl_dsn_mapper.sh --populate` at NIC bring-up + set `RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links` in RCCL env
2. **Short-term**: Upstream the `RCCL_BCM_LINKS_PATH` conditional fallback to RCCL
3. **Medium-term**: Deploy `switch_discovery` kernel module (eliminates env var dependency)

**Artifacts**

- Analysis spec: `split_mask_pcie_topo_analysis_spec.md` (section 9 has full validation details)
- DSN mapper: `rccl_dsn_mapper.sh`
- RCCL patch: `projects/rccl/src/os/linux.cc` (`ncclOsGetBcmLinks()`)
- NCCL_MAX_NCHANNELS=16 report: `test-experiments/nccl_max_nchannels_16_report.md`
- All-collectives busBw comparison: `test-experiments/all_collectives_comparison_report.md`
- GDR verification (all collectives): `test-experiments/gdr_verification_report.md`
- Before-fix evidence: `test-experiments/before-inter-link-visibility/`
- After-fix evidence: `test-experiments/after-inter-link-visibility/`

Please review the attached spec for full details including topology tables, code references, and scenario walkthroughs.

Thanks,
Karthik
