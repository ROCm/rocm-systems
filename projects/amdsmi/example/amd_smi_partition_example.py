#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# NOTE: Partition changes alter device topology -- AMD SMI must re-initialize
#
# Changing either memory partition (NPS mode) or accelerator (compute)
# partition changes the number of logical devices visible to the OS and to
# AMD SMI. AMD SMI builds its internal device table at amdsmi_init(); it does
# not update live. After any partition change, callers must call
# amdsmi_shut_down() followed by amdsmi_init() to pull in the updated
# topology; any processor handle obtained before re-initialization is
# invalid and must not be used.
#
#   Memory partition (NPS mode):
#     Requires amdsmi_gpu_driver_reload() to take effect. The driver reload
#     tears down and rebuilds all kernel device objects, which also destroys
#     any existing handles. amdsmi_shut_down() + amdsmi_init() is therefore
#     mandatory before querying or changing anything further.
#
#   Accelerator (compute) partition:
#     Takes effect immediately without a driver reload, but still changes the
#     logical device count (e.g. SPX: 1 handle per physical GPU vs. DPX: 2).
#     amdsmi_shut_down() + amdsmi_init() is required to get the correct
#     post-change handle list before issuing any further queries or sets.

import amdsmi
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Partition name converters
# ---------------------------------------------------------------------------


# Returns a comma-separated list of supported NPS modes (e.g. "NPS1,NPS2,NPS4").
def nps_caps_str(caps: List[str]) -> str:
    filtered = [c for c in caps if c != "N/A"]
    return ",".join(filtered) if filtered else "none"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_FILL = "-" * 70
_NPS_ORDER = {"NPS1": 1, "NPS2": 2, "NPS4": 4, "NPS8": 8}


def print_separator(title: str = "") -> None:
    if not title:
        print(_FILL)
        return
    prefix_len = 5 + len(title)
    remaining = _FILL[prefix_len:] if prefix_len < len(_FILL) else ""
    print(f"\n--- {title} {remaining}")


# ---------------------------------------------------------------------------
# AMD SMI lifecycle
# ---------------------------------------------------------------------------


# Scoped amdsmi session. Calls amdsmi_init on __enter__ and amdsmi_shut_down
# on __exit__. Re-enumerate (new instance) after any partition change that
# alters device topology:
#   - Memory partition change requires a driver reload, which destroys all
#     kernel device objects and invalidates every processor handle.
#   - Accelerator partition change changes the number of visible logical
#     devices (e.g. SPX: 1 per GPU -> DPX: 2 per GPU), so the handle list
#     built at amdsmi_init is stale even without a driver reload.
# Treat every processor handle as valid only within the session that
# obtained it.
class AmdsmiSession:
    def __init__(self) -> None:
        self._ok = False

    def is_ok(self) -> bool:
        return self._ok

    def __enter__(self) -> "AmdsmiSession":
        try:
            amdsmi.amdsmi_init()
            self._ok = True
            print("AMDSMI session started (amdsmi_init called)")
        except amdsmi.AmdSmiException as e:
            print(f"amdsmi_init failed: {e}")
        return self

    def __exit__(self, *_: Any) -> None:
        if not self._ok:
            return
        amdsmi.amdsmi_shut_down()
        print("AMDSMI session ended (amdsmi_shut_down called)")


# ---------------------------------------------------------------------------
# GPU enumeration
# ---------------------------------------------------------------------------


# Enumerate all AMD GPU handles into a flat list.
def get_all_gpu_handles() -> List:
    try:
        return amdsmi.amdsmi_get_processor_handles()
    except amdsmi.AmdSmiException as e:
        print(f"amdsmi_get_processor_handles failed: {e}")
        return []


# Returns only the "primary" handles from a full handle list -- i.e., those
# that successfully respond to amdsmi_get_gpu_accelerator_partition_profile_config.
# In a partitioned layout (e.g. CPX/NPS4) amdsmi enumerates one handle per
# *logical* partition. Only the root handle of each physical GPU owns the
# partition config; sub-partition handles return NOT_SUPPORTED. Calling set on
# a sub-partition handle is a no-op (NOT_SUPPORTED), so filter them out first.
# Each entry is (amd-smi GPU index, handle).
def get_primary_gpu_handles(all_gpus: List) -> List[Tuple[int, Any]]:
    primary = []
    for i, gpu in enumerate(all_gpus):
        try:
            amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
            primary.append((i, gpu))
        except amdsmi.AmdSmiException:
            pass  # sub-partition handles are expected to raise here
    return primary


# ---------------------------------------------------------------------------
# Per-step display helpers
# ---------------------------------------------------------------------------


def print_current_partition(idx: int, gpu: Any) -> None:
    print(f"  GPU {idx}:")
    try:
        cur_profile = amdsmi.amdsmi_get_gpu_accelerator_partition_profile(gpu)
        pp = cur_profile.get("partition_profile", {})
        print(f"    Accelerator profile type : {pp.get('profile_type', 'N/A')}")
        print(f"    Profile index            : {pp.get('profile_index', 'N/A')}")
        print(f"    Num partitions           : {pp.get('num_partitions', 'N/A')}")
        print(f"    Partition ID             : {cur_profile.get('partition_id', ['N/A'])[0]}")
        print(f"    Compatible NPS           : {nps_caps_str(pp.get('memory_caps', []))}")
    except amdsmi.AmdSmiException as e:
        print(f"    amdsmi_get_gpu_accelerator_partition_profile: {e}")
    try:
        mem_config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
        print(f"    Memory partition : {mem_config.get('mp_mode', 'N/A')}")
    except amdsmi.AmdSmiException as e:
        print(f"    amdsmi_get_gpu_memory_partition_config: {e}")


def print_available_modes(idx: int, gpu: Any) -> None:
    print(f"  GPU {idx}:")
    try:
        mem_config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
        caps = mem_config.get("partition_caps", [])
        print(f"    Supported NPS modes     : {nps_caps_str(caps)}")
    except amdsmi.AmdSmiException as e:
        print(f"    amdsmi_get_gpu_memory_partition_config: {e}")
    try:
        acc_config = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
        profiles = acc_config.get("profiles", [])
        print(f"    Available accelerator profiles ({len(profiles)}):")
        for profile in profiles:
            print(
                f"      Index {profile.get('profile_index')} "
                f"({profile.get('profile_type')}, "
                f"{profile.get('num_partitions')} partition(s), "
                f"compatible NPS: {nps_caps_str(profile.get('memory_caps', []))})"
            )
            for r, res in enumerate(profile.get("resources", [])):
                print(
                    f"        Resource {r}: {res.get('resource_type')}"
                    f", per_partition={res.get('partition_resource')}"
                    f", shared_by={res.get('num_partitions_share_resource')}"
                )
    except amdsmi.AmdSmiException as e:
        print(f"    amdsmi_get_gpu_accelerator_partition_profile_config: {e}")


# ---------------------------------------------------------------------------
# NPS -> accelerator-profile map
# ---------------------------------------------------------------------------


# Maps each NPS mode to the accelerator profiles that are compatible with it.
# Built by inverting the memory_caps list on every profile returned by
# amdsmi_get_gpu_accelerator_partition_profile_config.
def build_nps_to_profiles_map(acc_config: Dict) -> Dict[str, List[Dict]]:
    nps_map = {}
    for prof in acc_config.get("profiles", []):
        for nps in prof.get("memory_caps", []):
            if nps != "N/A":
                nps_map.setdefault(nps, []).append(prof)
    return nps_map


# ---------------------------------------------------------------------------
# Workflow helpers
# ---------------------------------------------------------------------------


def print_current_partition_info(gpus: List) -> None:
    print_separator("Current partition settings")
    for idx, gpu in enumerate(gpus):
        print_current_partition(idx, gpu)


def print_available_partition_modes(gpus: List) -> None:
    print_separator("Available partition modes")
    for idx, gpu in enumerate(gpus):
        print_available_modes(idx, gpu)


# Returns True if the memory partition was set successfully.
def set_memory_partition(gpu0: Any, target: amdsmi.AmdSmiMemoryPartitionType) -> bool:
    print_separator("Set memory partition")
    # Memory partition is hive-wide -- setting it on one device affects all.
    # Change the target mode to another supported mode as needed.
    target_name = target.name
    print(f"  Attempting: {target_name}")
    try:
        amdsmi.amdsmi_set_gpu_memory_partition_mode(gpu0, target)
        print(f"  amdsmi_set_gpu_memory_partition_mode(GPU 0, {target_name}): success")
        return True
    except amdsmi.AmdSmiException as e:
        print(f"  amdsmi_set_gpu_memory_partition_mode(GPU 0, {target_name}): {e}")
        return False


# Returns True if the driver was reloaded successfully.
def reload_driver() -> bool:
    print_separator("Reload driver")
    # Mandatory to apply memory partition change. All GPU activity must be
    # stopped first. The reload may reset the accelerator partition to default.
    # Root privileges (sudo) are required for driver reload.
    print("  Reloading driver, this may take some time...")
    try:
        amdsmi.amdsmi_gpu_driver_reload()
        print("  amdsmi_gpu_driver_reload: success")
        return True
    except amdsmi.AmdSmiException as e:
        print(f"  amdsmi_gpu_driver_reload: {e}")
        return False


# For each GPU: build the NPS->profiles map, print every entry, then demonstrate
# iterating over only the profiles compatible with the currently active NPS mode.
def print_profiles_by_nps(gpus: List) -> None:
    print_separator("Accelerator profiles grouped by NPS mode")
    for gpu_idx, gpu in enumerate(gpus):
        print(f"  GPU {gpu_idx}:")

        # --- query available accelerator profiles ---
        try:
            acc_config = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpu)
        except amdsmi.AmdSmiException as e:
            print(f"    amdsmi_get_gpu_accelerator_partition_profile_config: {e}")
            continue

        # --- query current NPS mode ---
        current_nps = None
        try:
            mem_config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpu)
            current_nps = mem_config.get("mp_mode")
        except amdsmi.AmdSmiException:
            pass

        # --- build map ---
        nps_map = build_nps_to_profiles_map(acc_config)

        # Print every NPS mode and its compatible profiles
        print("    All profiles by NPS mode:")
        for nps, profiles in sorted(nps_map.items(), key=lambda x: _NPS_ORDER.get(x[0], 99)):
            is_current = " [current]" if nps == current_nps else ""
            print(f"      {nps}{is_current} ({len(profiles)} profile(s)):")
            for prof in profiles:
                print(
                    f"        index={prof.get('profile_index')}"
                    f"  type={prof.get('profile_type')}"
                    f"  partitions={prof.get('num_partitions')}"
                )

        # Iterate over profiles for the *current* NPS mode
        if current_nps:
            print(f"    Profiles available for current NPS ({current_nps}):")
            current_profiles = nps_map.get(current_nps, [])
            if current_profiles:
                for prof in current_profiles:
                    print(
                        f"      -> index={prof.get('profile_index')}"
                        f"  type={prof.get('profile_type')}"
                        f"  partitions={prof.get('num_partitions')}"
                    )
            else:
                print("      (none)")


# Set the accelerator partition on every *primary* handle.
#
# Why primary-only:
#   In a partitioned layout (e.g. CPX+NPS4) amdsmi enumerates one handle per
#   logical partition (e.g. 64 handles for 8 physical GPUs in CPX). Only the
#   root handle of each physical GPU owns the partition config; the rest return
#   NOT_SUPPORTED. Calling set on them is noise, so we filter first.
def set_accelerator_partition_all_devices(
    all_gpus: List, profile_index: int, profile_type_name: str
) -> None:
    print_separator(f"Set accelerator partition -> index={profile_index} ({profile_type_name})")
    primary = get_primary_gpu_handles(all_gpus)
    print(f"  Primary handles: {len(primary)} of {len(all_gpus)} total")
    for phy_idx, (gpu_num, gpu) in enumerate(primary):
        try:
            amdsmi.amdsmi_set_gpu_accelerator_partition_profile(gpu, profile_index)
            result = "success"
        except amdsmi.AmdSmiException as e:
            result = str(e)
        print(
            f"  Physical GPU {phy_idx} (amd-smi GPU {gpu_num})"
            f": amdsmi_set_gpu_accelerator_partition_profile -> {result}"
        )


def main() -> None:
    mem_changed = False

    # -----------------------------------------------------------------------
    # Phase 1: Query state, set memory partition, reload driver if changed.
    #
    # If memory partition is successfully changed, a driver reload follows and
    # the session is closed (amdsmi_shut_down()) before Phase 2 re-initializes
    # with fresh handles. If memory partition cannot be changed, the driver
    # reload is skipped and accelerator partition changes are attempted
    # immediately within this same session.
    # -----------------------------------------------------------------------
    with AmdsmiSession() as session:
        if not session.is_ok():
            return
        gpus = get_all_gpu_handles()
        if not gpus:
            print("No GPUs found.")
            return
        print(f"Found {len(gpus)} GPU(s).")

        print_current_partition_info(gpus)
        print_available_partition_modes(gpus)

        # In real-world usage, the target mode may be dictated by workload requirements.
        # For this demo, we pick the highest supported NPS mode to show the most
        # significant partition change.

        # Pick the highest supported NPS mode, preferring NPS4 > NPS2 > NPS1.
        # Query the caps from GPU 0 (memory partition is hive-wide, so any GPU works).
        target_memory_partition = amdsmi.AmdSmiMemoryPartitionType.NPS1
        current_nps = None
        try:
            mem_config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpus[0])
            caps = mem_config.get("partition_caps", [])
            current_nps = mem_config.get("mp_mode")
            if "NPS4" in caps:
                target_memory_partition = amdsmi.AmdSmiMemoryPartitionType.NPS4
            elif "NPS2" in caps:
                target_memory_partition = amdsmi.AmdSmiMemoryPartitionType.NPS2
        except amdsmi.AmdSmiException as e:
            print(f"  [warn] Could not query NPS capabilities: {e}; defaulting to NPS1")
        print(f"  Selected NPS target: {target_memory_partition.name}")
        if current_nps == target_memory_partition.name:
            print(
                f"\n[info] Current NPS mode is already {current_nps}; "
                f"skipping set and driver reload."
            )
        else:
            mem_changed = set_memory_partition(gpus[0], target_memory_partition)
            if mem_changed:
                if not reload_driver():
                    print(
                        "[warn] Driver reload failed; memory partition change "
                        "may not have taken effect."
                    )
            else:
                print("\n[info] Memory partition unchanged; skipping driver reload.")

    # -----------------------------------------------------------------------
    # Phase 2: Re-enumerate to get fresh handles and accurate device count after
    # the driver reload (triggered by the memory partition change). Display the
    # current state and capture the accelerator profiles compatible with the
    # active NPS mode.
    #
    # Accelerator-partition profile dicts are plain data with no embedded
    # handles, so they remain valid across amdsmi_shut_down / amdsmi_init
    # boundaries and can be saved here for use in Phase 3.
    # -----------------------------------------------------------------------
    profiles_to_test = []
    with AmdsmiSession() as session:
        if not session.is_ok():
            return
        gpus = get_all_gpu_handles()
        if not gpus:
            print("No GPUs found after driver reload.")
            return
        print(f"Found {len(gpus)} GPU(s) after driver reload.")

        print_current_partition_info(gpus)
        print_available_partition_modes(gpus)
        print_profiles_by_nps(gpus)

        # Build NPS->profiles map and save the profiles for the active NPS mode.
        # These are plain dicts and remain valid after the session is torn down.
        try:
            mem_config = amdsmi.amdsmi_get_gpu_memory_partition_config(gpus[0])
            acc_config = amdsmi.amdsmi_get_gpu_accelerator_partition_profile_config(gpus[0])
            nps_map = build_nps_to_profiles_map(acc_config)
            current_nps = mem_config.get("mp_mode")
            if current_nps in nps_map:
                profiles_to_test = nps_map[current_nps]
                print(
                    f"\n[info] {len(profiles_to_test)} accelerator profile(s) to iterate "
                    f"for current NPS mode ({current_nps})."
                )
        except amdsmi.AmdSmiException as e:
            print(f"[warn] Could not build accelerator profile map: {e}")

    # -----------------------------------------------------------------------
    # Phase 3: For each accelerator profile compatible with the current NPS mode:
    #   a) Open a session, set the profile on every device, close the session.
    #   b) Open a NEW session to re-enumerate before reading anything.
    #      Accelerator partition changes alter the number of logical devices
    #      visible to the OS (e.g. SPX: N handles -> DPX: 2*N handles), so the
    #      handle list from step (a) is immediately stale. A new amdsmi_init
    #      is required to get the correct post-change device count and handles
    #      before displaying topology or issuing further queries.
    # -----------------------------------------------------------------------
    if not profiles_to_test:
        print("[info] No accelerator profiles available for current NPS mode; skipping Phase 3.")
        return

    for target in profiles_to_test:
        profile_index = target.get("profile_index")
        profile_type = target.get("profile_type")

        # a) Set accelerator partition on all devices.
        with AmdsmiSession() as session:
            if not session.is_ok():
                continue
            gpus = get_all_gpu_handles()
            if not gpus:
                print(
                    f"\n[warn] No GPUs found; skipping profile "
                    f"index={profile_index} ({profile_type})."
                )
                continue
            print(
                f"Found {len(gpus)} GPU(s) BEFORE re-initialization for accelerator profile "
                f"change to index={profile_index} ({profile_type})."
            )
            set_accelerator_partition_all_devices(gpus, profile_index, profile_type)

        # b) Re-enumerate and show resulting device topology.
        with AmdsmiSession() as session:
            if not session.is_ok():
                continue
            gpus = get_all_gpu_handles()
            if not gpus:
                print(
                    f"\n[warn] No GPUs found; skipping profile "
                    f"index={profile_index} ({profile_type})."
                )
                continue
            print(
                f"Found {len(gpus)} GPU(s) AFTER re-initialization for accelerator profile "
                f"change to index={profile_index} ({profile_type})."
            )
            print_current_partition_info(gpus)
            print_available_partition_modes(gpus)
            print_profiles_by_nps(gpus)


if __name__ == "__main__":
    main()
