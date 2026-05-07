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
"""
GPU Partition Workflow Example

Demonstrates the recommended workflow for changing GPU partition settings:

  1. View current partition settings
  2. View available modes (run BEFORE any partition change)
  3. Set memory partition mode (only to a supported mode from step 2)
  4. Reload driver (required; may reset accelerator partition to default)
  5. Re-initialize and verify
  6. Set accelerator partition (only valid profiles for the active memory
     partition, as seen in step 2)
  7. Verify

Requires root/sudo privileges. Run with:
    sudo python3 amd_smi_partition_example.py
"""

from amdsmi import (
    amdsmi_init,
    amdsmi_shut_down,
    amdsmi_get_processor_handles,
    amdsmi_get_gpu_accelerator_partition_profile,
    amdsmi_get_gpu_accelerator_partition_profile_config,
    amdsmi_set_gpu_accelerator_partition_profile,
    amdsmi_get_gpu_memory_partition,
    amdsmi_get_gpu_memory_partition_config,
    amdsmi_set_gpu_memory_partition,
    amdsmi_gpu_driver_reload,
    AmdSmiMemoryPartitionType,
    AmdSmiException,
)


def print_separator(title=""):
    width = 60
    if title:
        print(f"\n--- {title} {'-' * (width - len(title) - 5)}")
    else:
        print("-" * width)


def main():
    amdsmi_init()

    try:
        gpus = amdsmi_get_processor_handles()
        if not gpus:
            print("No GPUs found.")
            return

        gpu0 = gpus[0]
        print(f"Found {len(gpus)} GPU(s). Running partition workflow on GPU 0.\n")

        # ------------------------------------------------------------------
        # Step 1: View current partition settings
        # ------------------------------------------------------------------
        print_separator("Step 1: Current partition settings")

        try:
            cur_profile = amdsmi_get_gpu_accelerator_partition_profile(gpu0)
            pp = cur_profile.get("partition_profile", {})
            print(f"  Accelerator profile type : {pp.get('profile_type', 'N/A')}")
            print(f"  Profile index            : {pp.get('profile_index', 'N/A')}")
            print(f"  Num partitions           : {pp.get('num_partitions', 'N/A')}")
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_accelerator_partition_profile: {e}")

        try:
            cur_mem = amdsmi_get_gpu_memory_partition(gpu0)
            print(f"  Memory partition         : {cur_mem}")
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_memory_partition: {e}")

        # ------------------------------------------------------------------
        # Step 2: View available modes -- run BEFORE any partition change.
        #         Only set modes that appear as supported here.
        # ------------------------------------------------------------------
        print_separator("Step 2: Available partition modes")

        available_acc_profiles = []
        try:
            mem_config = amdsmi_get_gpu_memory_partition_config(gpu0)
            print(f"  Supported memory modes: {mem_config.get('partition_caps', 'N/A')}")
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_memory_partition_config: {e}")

        try:
            acc_config = amdsmi_get_gpu_accelerator_partition_profile_config(gpu0)
            profiles = acc_config.get("profiles", [])
            print(f"  Available accelerator profiles ({len(profiles)}):")
            for p in profiles:
                print(
                    f"    Index {p.get('profile_index')}: "
                    f"type={p.get('profile_type')}, "
                    f"num_partitions={p.get('num_partitions')}, "
                    f"memory_caps={p.get('memory_caps')}"
                )
            available_acc_profiles = profiles
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_accelerator_partition_profile_config: {e}")

        # ------------------------------------------------------------------
        # Step 3: Set memory partition mode (must be supported per step 2)
        # ------------------------------------------------------------------
        print_separator("Step 3: Set memory partition")

        # Use NPS1 as target; change to another supported mode as needed.
        target_mem = AmdSmiMemoryPartitionType.NPS1
        mem_set_ok = False
        try:
            amdsmi_set_gpu_memory_partition(gpu0, target_mem)
            print(f"  amdsmi_set_gpu_memory_partition(NPS1): success")
            mem_set_ok = True
        except AmdSmiException as e:
            print(f"  amdsmi_set_gpu_memory_partition(NPS1): {e}")

    finally:
        amdsmi_shut_down()

    if not mem_set_ok:
        print("\nMemory partition set failed; skipping driver reload and remaining steps.")
        return

    # ------------------------------------------------------------------
    # Step 4: Reload driver (mandatory; may reset accelerator partition)
    # ------------------------------------------------------------------
    print_separator("Step 4: Reload driver")
    print("  Reloading driver, this may take some time...")

    amdsmi_init()
    try:
        amdsmi_gpu_driver_reload()
        print("  amdsmi_gpu_driver_reload: success")
    except AmdSmiException as e:
        print(f"  amdsmi_gpu_driver_reload: {e}")
    finally:
        amdsmi_shut_down()

    # ------------------------------------------------------------------
    # Step 5: Re-initialize and verify
    # ------------------------------------------------------------------
    print_separator("Step 5: Re-initialize and verify")

    amdsmi_init()
    try:
        gpus = amdsmi_get_processor_handles()
        gpu0 = gpus[0]
        print(f"  GPU count after memory partition change: {len(gpus)}")

        try:
            new_mem = amdsmi_get_gpu_memory_partition(gpu0)
            print(f"  New memory partition: {new_mem}")
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_memory_partition: {e}")

        # ------------------------------------------------------------------
        # Step 6: Set accelerator partition (only valid profiles for the
        #         active memory partition, as seen in step 2)
        # ------------------------------------------------------------------
        print_separator("Step 6: Set accelerator partition")

        # Use profile index 0 (typically SPX); change to desired index as needed.
        target_acc_index = 0
        try:
            amdsmi_set_gpu_accelerator_partition_profile(gpu0, target_acc_index)
            print(
                f"  amdsmi_set_gpu_accelerator_partition_profile(index={target_acc_index}): success"
            )
        except AmdSmiException as e:
            print(f"  amdsmi_set_gpu_accelerator_partition_profile(index={target_acc_index}): {e}")

        # ------------------------------------------------------------------
        # Step 7: Verify
        # ------------------------------------------------------------------
        print_separator("Step 7: Verify accelerator partition")

        gpus = amdsmi_get_processor_handles()
        gpu0 = gpus[0]
        print(f"  GPU count after accelerator partition change: {len(gpus)}")

        try:
            new_profile = amdsmi_get_gpu_accelerator_partition_profile(gpu0)
            pp = new_profile.get("partition_profile", {})
            print(f"  New accelerator profile type : {pp.get('profile_type', 'N/A')}")
            print(f"  New profile index            : {pp.get('profile_index', 'N/A')}")
            print(f"  Num partitions               : {pp.get('num_partitions', 'N/A')}")
        except AmdSmiException as e:
            print(f"  amdsmi_get_gpu_accelerator_partition_profile: {e}")

    finally:
        amdsmi_shut_down()

    print_separator()
    print("Partition workflow complete.")


if __name__ == "__main__":
    main()
