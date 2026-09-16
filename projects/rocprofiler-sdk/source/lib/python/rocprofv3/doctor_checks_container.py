# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Container and virtualization checks."""

from __future__ import absolute_import

from rocprofv3.doctor_checks_driver import KFD_DEVICE
from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_pass,
    make_warn,
    SEV_INFO,
    SEV_WARNING,
)

DOCKER_RUN_HINT = (
    "Pass the GPU devices into the container:\n"
    "  docker run --device=/dev/kfd --device=/dev/dri \\\n"
    "    --group-add video --security-opt seccomp=unconfined \\\n"
    "    --cap-add=SYS_PTRACE <image>"
)


def detect_container(accessor):
    """Return ``(runtime_or_None, evidence_list)``."""
    evidence = []
    runtime = None

    if accessor.path_exists("/.dockerenv"):
        evidence.append("/.dockerenv exists")
        runtime = "docker"

    if accessor.path_exists("/run/.containerenv"):
        evidence.append("/run/.containerenv exists")
        runtime = runtime or "podman"

    cgroup = accessor.read_file("/proc/1/cgroup")
    if cgroup:
        for marker, name in (
            ("docker", "docker"),
            ("containerd", "containerd"),
            ("lxc", "lxc"),
            ("kubepods", "kubernetes"),
        ):
            if marker in cgroup:
                evidence.append("/proc/1/cgroup mentions {!r}".format(marker))
                runtime = runtime or name

    if accessor.getenv("SINGULARITY_CONTAINER") or accessor.getenv("APPTAINER_CONTAINER"):
        evidence.append("Singularity/Apptainer environment variable set")
        runtime = runtime or "apptainer"

    return (runtime, evidence)


@register(
    id="container.detected",
    group="container",
    title="Container environment detection",
    severity=SEV_INFO,
    order=60,
)
def check_container_detected(accessor):
    runtime, evidence = detect_container(accessor)
    data = {"container_runtime": runtime, "evidence": evidence}
    if runtime is None:
        return make_pass("not running inside a container", "", data)
    return make_pass(
        "running inside a {} container ({})".format(runtime, "; ".join(evidence)),
        "",
        data,
    )


@register(
    id="container.device-passthrough",
    group="container",
    title="GPU devices passed through to the container",
    severity=SEV_WARNING,
    order=61,
)
def check_device_passthrough(accessor):
    runtime, _ = detect_container(accessor)
    data = {"container_runtime": runtime}

    if runtime is None:
        return make_pass(
            "not containerized; device passthrough is not applicable", "", data
        )

    missing = []
    if not accessor.path_exists(KFD_DEVICE):
        missing.append(KFD_DEVICE)
    elif not (accessor.path_readable(KFD_DEVICE) and accessor.path_writable(KFD_DEVICE)):
        missing.append("{} (present but not accessible)".format(KFD_DEVICE))

    if not accessor.glob("/dev/dri/renderD*"):
        missing.append("/dev/dri/renderD*")
    data["missing"] = missing

    if missing:
        return make_warn(
            "GPU device node(s) not usable inside this {} container: {}".format(
                runtime, ", ".join(missing)
            ),
            DOCKER_RUN_HINT,
            data,
        )
    return make_pass("GPU device nodes are present and accessible", "", data)


@register(
    id="container.virt-gpu",
    group="container",
    title="Virtualized / SR-IOV GPU detection",
    severity=SEV_INFO,
    order=62,
)
def check_virt_gpu(accessor):
    # A virtualized GPU (SR-IOV guest) usually cannot collect hardware counters:
    # the host retains the performance-monitoring resources.
    data = {}
    hypervisors = []

    cpuinfo = accessor.read_file("/proc/cpuinfo") or ""
    if "hypervisor" in cpuinfo:
        hypervisors.append("CPU reports the hypervisor flag")

    product = accessor.read_file("/sys/class/dmi/id/product_name")
    if product:
        data["dmi_product_name"] = product.strip()
        for marker in ("KVM", "VMware", "VirtualBox", "Xen", "Hyper-V", "Amazon EC2"):
            if marker.lower() in product.lower():
                hypervisors.append("DMI product name is {!r}".format(product.strip()))
                break

    # NOTE: deliberately no SR-IOV bit test on the KFD `capability` field.
    # In the HSA_CAPABILITY layout (see HSA_CAPABILITY in
    # source/include/rocprofiler-sdk/cxx/serialization/save.hpp) the bits in
    # that region belong to the 4-bit ASICRevision field and SRAM_EDCSupport,
    # not to any virtual-function flag -- testing them would warn about
    # "SR-IOV" on any ordinary GPU with a high enough ASIC revision. There is
    # no read-only sysfs indicator of VF status that we can cite, and a
    # speculative bit test in a diagnostic is worse than no test at all, so
    # this check reports only the hypervisor evidence it can actually stand
    # behind.
    data["hypervisor_evidence"] = hypervisors

    if hypervisors:
        return make_pass(
            "running under a hypervisor ({}); GPU appears to be passed through "
            "rather than virtualized".format("; ".join(hypervisors)),
            "",
            data,
        )
    return make_pass("no GPU virtualization detected", "", data)
