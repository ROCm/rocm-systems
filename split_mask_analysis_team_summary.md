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

**UPDATE: Experimental Validation Complete (2026-10-10)**

We implemented and validated Approach 2 end-to-end on the SMC300x cluster. The fix consists of two components:

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
* = alternating dual-NIC, no GDRDMA, 32 channels, 65 proxy connections
✓ = fixed — now single NIC, GDRDMA, 16 channels, 33 proxy connections
```

**GDR Coverage**: 50% (4/8) → **100% (8/8)**

**Results: Transport Comparison**

```
┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Metric          │ Before (no inter-link)       │ After (DSN mapper)           │
├─────────────────┼──────────────────────────────┼──────────────────────────────┤
│ GDR ON          │ 4 of 8 GPUs (50%)            │ 8 of 8 GPUs (100%)          │
│ NICs per rank   │ 1 (GDR-ON), 2 (GDR-OFF)     │ 1 (uniform)                 │
│ Channels        │ 16 (GDR-ON), 32 (GDR-OFF)   │ 16 (uniform)                │
│ Proxy conns     │ 33 (GDR-ON), 65 (GDR-OFF)   │ 33 (uniform)                │
│ Transport       │ GDRDMA / plain NET mix       │ GDRDMA (all channels)       │
└─────────────────┴──────────────────────────────┴──────────────────────────────┘
```

**Results: RCCL busBw Comparison (out-of-place, GB/s)**

```
┌───────────┬──────────────────────────┬──────────────────────────┬──────────┐
│ Msg Size  │ Before: AR / A2A         │ After: AR / A2A          │ Delta    │
├───────────┼──────────────────────────┼──────────────────────────┼──────────┤
│ 64M       │ 14.47 / 7.62            │ 18.26 / 11.00            │ +26/+44% │
│ 128M      │ 21.12 / 17.67           │ 23.36 / 16.26            │ +11/−8%  │
│ 256M      │ 22.00 / 22.55           │ 23.71 / 22.57            │  +8/0%   │
│ 512M      │ 22.57 / 23.14           │ 24.05 / 23.22            │  +7/0%   │
│ 1G        │ 23.10 / 23.35           │ 24.19 / 23.43            │  +5/0%   │
│ 2G        │ 26.68 / 23.51           │ 24.23 / 23.54            │  −9/0%   │
│ 4G        │ 24.75 / 23.57           │ 24.32 / 23.62            │  −2/0%   │
│ 8G        │ 24.55 / 23.60           │ 24.35 / 23.66            │  −1/0%   │
│ 16G       │ 24.74 / 23.59           │ 24.39 / 23.66            │  −1/0%   │
└───────────┴──────────────────────────┴──────────────────────────┴──────────┘
```

**Performance Analysis**:
- **all_reduce mid-range (64M–256M)**: +8% to +26% improvement — uniform GDRDMA eliminates CPU bounce overhead at these sizes
- **all_reduce saturated (1G+)**: ~24.2 GB/s uniform profile. The before-fix 2G "peak" of 26.68 GB/s was an artifact of GDR-OFF ranks using dual-rail (2 NICs, 32 channels) which inflated throughput at that specific size despite CPU bounce overhead
- **alltoall**: Comparable at saturated sizes (~23.5 GB/s); +44% improvement at 64M mid-range
- **Primary win is architectural correctness**: uniform GDR, uniform NIC assignment, halved proxy connections, elimination of mixed-transport asymmetry

**Deployment Path**

1. **Immediate** (validated): Run `rccl_dsn_mapper.sh --populate` at NIC bring-up + set `RCCL_BCM_LINKS_PATH=/var/run/rccl_bcm_links` in RCCL env
2. **Short-term**: Upstream the `RCCL_BCM_LINKS_PATH` conditional fallback to RCCL
3. **Medium-term**: Deploy `switch_discovery` kernel module (eliminates env var dependency)

**Artifacts**

- Analysis spec: `split_mask_pcie_topo_analysis_spec.md` (section 9 has full validation details)
- DSN mapper: `rccl_dsn_mapper.sh`
- RCCL patch: `projects/rccl/src/os/linux.cc` (`ncclOsGetBcmLinks()`)
- Before-fix evidence: `test-experiments/before-inter-link-visibility/`
- After-fix evidence: `test-experiments/after-inter-link-visibility/`

Please review the attached spec for full details including topology tables, code references, and scenario walkthroughs.

Thanks,
Karthik
