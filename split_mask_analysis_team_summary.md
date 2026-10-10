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

Please review the attached spec for full details including topology tables, code references, and scenario walkthroughs.

Thanks,
Karthik
