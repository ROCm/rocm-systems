#!/usr/bin/env python3
"""Replace duplicate HIP_SKIP_TEST string literals with HipTest::SkipReason::* constants."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXT = {".cc", ".hh", ".cpp", ".hpp"}

# Full-call replacements (exact substring). Longer / more specific first.
REPLACEMENTS: list[tuple[str, str]] = [
    # Multi-line: no GPU (perf)
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfBufferCopyRectSpeed as there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfVMMAlloc as"
        "there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfDevMemWriteSpeed as"
        "there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfDevMemReadSpeed as"
        "there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfMemMallocCpyFree as"
        "there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfHostNumaAlloc as "
        "there is no device to test.\\n");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfSharedMemReadSpeed as"
        "there is no device to test.\\n");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Skipped testcase hipPerfBufferCopySpeed as"
        "there is no device to test.");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);",
    ),
    # Multi-line: memory pools (performance stream)
    (
        """    HipTest::HIP_SKIP_TEST(
        "GPU 0 doesn't support hipDeviceAttributeMemoryPoolsSupported "
        "attribute. Hence skipping the testing with Pass result.\\n");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported);",
    ),
    (
        """    HipTest::HIP_SKIP_TEST(
        "Test need at least one device with managed memory support");""",
        "    HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryDeviceRequired);",
    ),
]

# HipTest::HIP_SKIP_TEST("...") -> HipTest::HIP_SKIP_TEST(HipTest::SkipReason::...)
LITERAL_REPLACEMENTS: list[tuple[str, str]] = [
    ('HipTest::HIP_SKIP_TEST("Skipped the test as there is no peer access")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("skipping the testcases as numDevices < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("GPU doesn\'t support managed memory so skipping test.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Memory allocation failed. Skip test. Is SVM atomic supported?")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kCoherentHostAllocFailed)"),
    ('HipTest::HIP_SKIP_TEST("Does not support HostPinned Memory")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("No GPU for Testing")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Device does not support mapping host memory (canMapHostMemory)")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Coherence memory allocation failed. Is SVM atomic supported?")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kCoherentHostAllocFailed)"),
    ('HipTest::HIP_SKIP_TEST("Assembly file does not exist")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kAssemblyFileMissing)"),
    ('HipTest::HIP_SKIP_TEST("Doesn\'t support HostPinned Memory")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Mipmapped arrays not supported on this device")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMipmappedArraysUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Memory Pool not supported. Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Multiple GPUs not available. Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Runtime doesn\'t support Memory Pool. Skip the test case.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device Does not have P2P capability")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("Machine Does not have P2P capability")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("Number of devices are < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Machine does not seem to have P2P")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("skipped the testcase as no of devices is less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("deviceCount less then 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("P2P capability is not present")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("No of devices are less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping test because more than one device was not found.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Gpu doesnt support HMM! Hence skipping the test with PASS result")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipped testcase hipPerfMemcpy as there is no device to test.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Skipped the testcase as there is no device to test.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Skipped test as there is no device to test.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("skipped the testcase as number of devices is less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Not enough GPUs to run the masked GPU tests")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNotEnoughGpusMaskedTests)"),
    ('HipTest::HIP_SKIP_TEST("skipped the testcase as no of devices is less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Test requires at least 2 devices")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("At least 2 devices are required")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because devices < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because devices < 1")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Two or more device are required")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Test requires more than one device")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Machine doesn\'t have multiple GPUs; skipping test")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Machine doesn\'t have P2P support enabled; skipping test")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("No device found, skipping the test.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Skipping hipDynamicLogging test - no devices available")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("This test requires 2 GPUs. Skipping.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Test only runs on devices with memory pool support")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipping test since Memory Pool is not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device doest have memory pool support")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kMemoryPoolUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("managed memory access not supported on device")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    # Second pass: remaining duplicates
    ('HipTest::HIP_SKIP_TEST("Managed memory not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipping test since managed memory not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Managed memory is not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because no P2P support")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable)"),
    ('HipTest::HIP_SKIP_TEST("Skipping the testcases as numDevices < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("NumDevices <2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("NumDevices are less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("No of Devices < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("skipped testcase as Device count is < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping the test-cases as number of Devices found less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping the test as number of Devices found less than 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Number of GPUs insufficient for test")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Need 2 GPUs to run test. Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Machine is Single GPU. Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("GPU doesn\'t support managed memory.Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipping since cooperative launch not supported")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device doesn\'t support cooperative launch!")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device doesn\'t support pcie atomic, Skipped")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kPcieAtomicUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("hipStreamWaitValue not supported on this device.")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported)"),
    # Third pass
    ('HipTest::HIP_SKIP_TEST("Device number is < 2")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 1")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Skipping because devices <= 1")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice)"),
    ('HipTest::HIP_SKIP_TEST("Device doesn\'t support Warp Shuffle!")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kWarpShuffleUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device doesn\'t support Warp Vote!")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kWarpVoteUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device doesn\'t support Warp Ballot!")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kWarpBallotUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device does not support mapping host memory")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Device Property canMapHostMemory is not set")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Skipping, as this test requires more than 2 GPUs")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus)"),
    ('HipTest::HIP_SKIP_TEST("Machine does not support VMM. Skipping Test..")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported)"),
    ('HipTest::HIP_SKIP_TEST("Test need at least one device with managed memory support")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryDeviceRequired)"),
    ('HipTest::HIP_SKIP_TEST("Test needs at least 1 device that supports managed memory")',
     "HipTest::HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryDeviceRequired)"),
]

# Indented variants (6 spaces) — same literals, used inside nested blocks
INDENT = "      "
INDENT_LITERAL_REPLACEMENTS: list[tuple[str, str]] = []
for old, new in LITERAL_REPLACEMENTS:
    if old.startswith("HipTest::HIP_SKIP_TEST("):
        INDENT_LITERAL_REPLACEMENTS.append(
            (INDENT + old, INDENT + new)
        )


def main() -> int:
    changed = 0
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in EXT:
            continue
        text = path.read_text(encoding="utf-8")
        orig = text
        for a, b in REPLACEMENTS + LITERAL_REPLACEMENTS + INDENT_LITERAL_REPLACEMENTS:
            text = text.replace(a, b)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            changed += 1
            print(path.relative_to(ROOT))
    print(f"Modified {changed} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
