#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Host metadata collector.

Best-effort snapshot of the test node's hardware/software state, captured at run
time and stored with the run so the dashboard can show exactly what a result ran
on. Every probe is wrapped and failure-tolerant -- collection never fails the run.

Scale-up fabric telemetry (UALink) is gated on capability presence: it is
collected only when the UALink sysfs (/sys/class/drm/card*/device/ualink/...)
exists, rather than guessing from product/arch strings. Every probe reads
standard sysfs or ROCm CLIs; nothing here is hardcoded to a specific host,
cluster, or product.
"""

import glob
import os
import re
import socket
import subprocess


def _run(cmd, timeout=20):
    """Run a command, returning stripped stdout or None. Best-effort."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _sudo_read(path_glob):
    """Read one-or-more files (glob) that may require sudo. Returns {path: text}."""
    result = {}
    paths = sorted(glob.glob(path_glob))
    for p in paths:
        txt = _read(p)
        if txt is None:
            # Retry via sudo -n (non-interactive); silently skip if unavailable.
            txt = _run(["sudo", "-n", "cat", p], timeout=10)
        if txt is not None:
            result[p] = txt.strip()
    return result


def _os_pretty_name():
    data = _read("/etc/os-release") or ""
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', data, re.MULTILINE)
    return m.group(1) if m else None


def _open_file_limit():
    try:
        import resource
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        return None


def _gpu_arch():
    out = _run(["rocm_agent_enumerator"], timeout=15)
    if out:
        for tok in out.split():
            if tok.startswith("gfx") and tok != "gfx000":
                return tok
    info = _run(["rocminfo"], timeout=25) or ""
    m = re.search(r"\bgfx[0-9a-f]+\b", info)
    return m.group(0) if m else None


def _compute_units():
    info = _run(["rocminfo"], timeout=25) or ""
    m = re.search(r"Compute Unit:\s*(\d+)", info)
    return int(m.group(1)) if m else None


def _gpu_bdfs():
    """PCI BDFs of the amdgpu devices."""
    bdfs = []
    for path in sorted(glob.glob("/sys/bus/pci/drivers/amdgpu/0000:*")):
        bdfs.append(os.path.basename(path))
    return bdfs or None


def _firmware_versions():
    """Parse `rocm-smi --showfwinfo` into {component: version} (first GPU block)."""
    out = _run(["rocm-smi", "--showfwinfo"], timeout=30)
    if not out:
        return None
    fw = {}
    for line in out.splitlines():
        # e.g. "GPU[0]  : MEC FW version: 0x000000a5"
        m = re.search(r"([A-Za-z0-9_ ]+?)\s*FW version:\s*(\S+)", line)
        if m:
            key = m.group(1).strip().split()[-1].lower()
            fw.setdefault(key, m.group(2))
    return fw or None


def _cmdline():
    return _read("/proc/cmdline") or ""


def _check_iommu_pt():
    """RCCL/RDMA want the kernel booted with iommu=pt (passthrough)."""
    cl = _cmdline()
    pt = "iommu=pt" in cl
    return {
        "status": "OK" if pt else "WARN",
        "value": "iommu=pt" if pt else "iommu=pt not on kernel cmdline",
        "amd_iommu_on": ("amd_iommu=on" in cl or "iommu=on" in cl),
    }


def _check_acs():
    """ACS should be disabled for GPU P2P / RDMA. SrcValid+ => still enabled."""
    out = _run(["sudo", "-n", "bash", "-c", "lspci -vvv 2>/dev/null | grep ACSCtl"], timeout=40)
    if out is None:
        out = _run(["bash", "-c", "lspci -vvv 2>/dev/null | grep ACSCtl"], timeout=40)
    if not out:
        return {"status": "SKIP", "value": "needs root / lspci"}
    enabled = "SrcValid+" in out
    return {"status": "WARN" if enabled else "OK",
            "value": "ACS not disabled" if enabled else "ACS disabled"}


def _check_limits():
    checks = {}
    try:
        import resource
        nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        checks["nofile"] = {"status": "OK" if nofile >= 1048576 else "WARN", "value": nofile}
        memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        unlim = memlock == resource.RLIM_INFINITY
        checks["memlock"] = {"status": "OK" if unlim else "WARN",
                             "value": "unlimited" if unlim else memlock}
    except Exception:
        pass
    return checks


def _rdma_metadata():
    """RDMA NIC (RoCE/IB) port state -- the inter-node scale-out fabric RCCL uses.
    Passive sysfs read (no tool/root needed). Returns None if no ibverbs devices."""
    base = "/sys/class/infiniband"
    if not os.path.isdir(base):
        return None
    devices = []
    active = total = 0
    try:
        devs = sorted(os.listdir(base))
    except OSError:
        return None
    for dev in devs:
        ports_dir = os.path.join(base, dev, "ports")
        if not os.path.isdir(ports_dir):
            continue
        for port in sorted(os.listdir(ports_dir)):
            p = os.path.join(ports_dir, port)
            state = _read(os.path.join(p, "state"))
            total += 1
            is_active = bool(state and "ACTIVE" in state)
            active += 1 if is_active else 0
            devices.append({
                "device": dev, "port": port,
                "state": state,
                "phys_state": _read(os.path.join(p, "phys_state")),
                "rate": _read(os.path.join(p, "rate")),
                "link_layer": _read(os.path.join(p, "link_layer")),
                "active": is_active,
            })
    if not devices:
        return None
    return {"present": True, "active_ports": active, "total_ports": total, "devices": devices}


def _xgmi_metadata():
    """XGMI / Infinity Fabric topology -- an intra-node scale-up fabric.
    Best-effort; records the topology matrix when XGMI links are present."""
    topo = _run(["amd-smi", "topology"], timeout=40)
    src = "amd-smi topology"
    if not topo or "XGMI" not in topo.upper():
        topo = _run(["rocm-smi", "--showtopo"], timeout=40)
        src = "rocm-smi --showtopo"
    if not topo or "XGMI" not in topo.upper():
        return None
    return {
        "present": True,
        "source": src,
        "xgmi_link_mentions": topo.upper().count("XGMI"),
        "raw": topo[:8000],
    }


def _ualink_metadata():
    """UALink scale-up fabric telemetry -- gated on the UALink sysfs. Returns
    None when absent (i.e. not a UALink-capable machine)."""
    ualink_dirs = glob.glob("/sys/class/drm/card*/device/ualink")
    if not ualink_dirs:
        return None

    md = {"present": True}

    # Station lane-enable bitmaps: 18-char hex per GPU, 'f'=active, '0'=masked.
    masks = _sudo_read("/sys/class/drm/card*/device/ualink/stations/lane_en_bitmap")
    # Key by card name for readability.
    station_mask = {}
    for path, val in masks.items():
        m = re.search(r"(card\d+)", path)
        station_mask[m.group(1) if m else path] = val.strip()
    if station_mask:
        md["station_mask"] = station_mask

    # Fabric manager version/device, best-effort (may need sudo).
    afmctl_ver = _run(["afmctl", "--version"], timeout=15) or _run(["sudo", "-n", "afmctl", "--version"], timeout=15)
    if afmctl_ver:
        md["afmctl_version"] = afmctl_ver.splitlines()[0] if afmctl_ver else None
    dev = _run(["sudo", "-n", "afmctl", "show", "device"], timeout=30)
    if dev:
        md["afmctl_device"] = dev
    return md


def collect(rocm_version=None):
    """Collect the host metadata snapshot. `rocm_version` may be passed in from the
    caller (test_runner already resolves it); everything else is probed here."""
    md = {
        "hostname": socket.gethostname(),
        "kernel": _run(["uname", "-r"]),
        "os": _os_pretty_name(),
        "cmdline": _read("/proc/cmdline"),
        "numa_balancing": _read("/proc/sys/kernel/numa_balancing"),
        "open_file_limit": _open_file_limit(),
        "gpu_recovery": _read("/sys/module/amdgpu/parameters/gpu_recovery"),
    }

    lsmod = _run(["lsmod"]) or ""
    md["modules"] = {
        "amdgpu": bool(re.search(r"^amdgpu\b", lsmod, re.MULTILINE)),
    }

    md["gpu"] = {
        "arch": _gpu_arch(),
        "compute_units": _compute_units(),
        "bdfs": _gpu_bdfs(),
    }

    hip = _run(["hipconfig", "--version"], timeout=15)
    ucx = _run(["ucx_info", "-v"], timeout=15)
    mpi = _run(["mpirun", "--version"], timeout=15)
    md["versions"] = {
        "rocm": rocm_version,
        "amdgpu": _read("/sys/module/amdgpu/version"),
        "sbios": _read("/sys/class/dmi/id/bios_version"),
        "kernel": md["kernel"],
        "hip": hip.splitlines()[0] if hip else None,
        "ucx": (re.search(r"Library version:\s*(\S+)", ucx).group(1) if ucx and re.search(r"Library version:\s*(\S+)", ucx) else None),
        "mpi": mpi.splitlines()[0] if mpi else None,
    }
    fw = _firmware_versions()
    if fw:
        md["firmware"] = fw

    # Config-correctness checks (record + flag; never gates the run).
    md["checks"] = {
        "iommu_pt": _check_iommu_pt(),
        "acs": _check_acs(),
        "numa_balancing": {
            "status": "OK" if md.get("numa_balancing") == "0" else "WARN",
            "value": md.get("numa_balancing"),
        },
        **_check_limits(),
    }

    # Scale-out fabric: RDMA NICs (RoCE/IB) -- gated on ibverbs presence.
    rdma = _rdma_metadata()
    md["rdma"] = rdma if rdma else {"present": False}

    # Intra-node scale-up: XGMI / Infinity Fabric -- gated on XGMI topology presence.
    xgmi = _xgmi_metadata()
    md["xgmi"] = xgmi if xgmi else {"present": False}

    # Intra-node scale-up: UALink -- gated on ualink sysfs presence.
    ualink = _ualink_metadata()
    md["ualink"] = ualink if ualink else {"present": False}

    return md


if __name__ == "__main__":
    import json
    print(json.dumps(collect(), indent=2, default=str))
