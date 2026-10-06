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

"""Driver and device checks: /dev/kfd, amdgpu, KFD topology, group membership."""

from __future__ import absolute_import

from rocprofv3.doctor_registry import register
from rocprofv3.doctor_result import (
    make_fail,
    make_pass,
    make_warn,
    SEV_ERROR,
    SEV_WARNING,
)

KFD_DEVICE = "/dev/kfd"
KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"
AMDGPU_MODULE_DIR = "/sys/module/amdgpu"
AMDGPU_VERSION = AMDGPU_MODULE_DIR + "/version"

DRIVER_INSTALL_HINT = (
    "Install the amdgpu kernel driver:\n"
    "  sudo amdgpu-install --usecase=rocm\n"
    "then reboot."
)


def parse_topology_node(accessor, node_dir):
    """Parse a KFD topology node's ``properties`` into a dict of ints.

    Returns None when the file is absent or empty (cgroup-masked nodes have an
    empty properties file).
    """
    content = accessor.read_file(accessor.join(node_dir, "properties"))
    if not content:
        return None
    props = {}
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            props[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return props or None


def gpu_nodes(accessor):
    """List of ``(node_dir, properties)`` for topology nodes that are GPUs.

    A node is a GPU iff ``simd_count > 0`` -- node 0 is the CPU and reports
    ``simd_count 0``.
    """
    found = []
    for node_dir in accessor.glob(accessor.join(KFD_TOPOLOGY_NODES, "*")):
        props = parse_topology_node(accessor, node_dir)
        if props and props.get("simd_count", 0) > 0:
            found.append((node_dir, props))
    return found


def _group_check(accessor, group_name):
    """Shared implementation for the render/video group membership checks."""
    gid = accessor.get_group_gid(group_name)
    groups = accessor.getgroups()
    data = {"group": group_name, "gid": gid, "effective_gids": groups}

    if accessor.getuid() == 0:
        # Root bypasses device-node permissions, so group membership grants it
        # nothing. Whether root can really open /dev/kfd (not a given in a
        # user namespace) is what driver.kfd-readable checks.
        return make_pass(
            "running as root: membership in {!r} is not needed".format(group_name),
            "",
            data,
        )

    if gid is None:
        return make_warn(
            "group {!r} does not exist on this system".format(group_name),
            "Some distributions do not define this group; if GPU access works, "
            "this is expected.",
            data,
        )

    if gid in groups:
        return make_pass("user is in the {!r} group".format(group_name), "", data)

    username = accessor.get_username()
    members = accessor.get_group_members(group_name)
    data["username"] = username
    data["group_members"] = members

    if members is not None and username and username in members:
        # already a member on paper -- telling them to run usermod would be
        # wrong and confusing; the session just predates the membership
        return make_warn(
            "{!r} is a member of {!r} but this session does not have the group "
            "in its effective group list".format(username, group_name),
            "Log out and log back in, or start a new session with:\n"
            "  newgrp {}".format(group_name),
            data,
        )

    return make_warn(
        "user is not in the {!r} group".format(group_name),
        "sudo usermod -a -G {} $USER\n"
        "then log out and log back in.".format(group_name),
        data,
    )


@register(
    id="driver.kfd-device",
    group="driver",
    title="/dev/kfd device node exists",
    severity=SEV_ERROR,
    depends=["install.rocm-root"],
    order=20,
)
def check_kfd_device(accessor):
    data = {"path": KFD_DEVICE}
    if accessor.path_exists(KFD_DEVICE):
        return make_pass("{} exists".format(KFD_DEVICE), "", data)
    return make_fail(
        "{} does not exist -- the amdgpu/KFD driver is not loaded or no AMD GPU "
        "is present".format(KFD_DEVICE),
        DRIVER_INSTALL_HINT,
        data,
    )


@register(
    id="driver.kfd-readable",
    group="driver",
    title="/dev/kfd readable and writable by the current user",
    severity=SEV_ERROR,
    depends=["driver.kfd-device"],
    order=21,
)
def check_kfd_readable(accessor):
    readable = accessor.path_readable(KFD_DEVICE)
    writable = accessor.path_writable(KFD_DEVICE)
    data = {"path": KFD_DEVICE, "readable": readable, "writable": writable}
    if readable and writable:
        return make_pass("{} is readable and writable".format(KFD_DEVICE), "", data)
    if accessor.getuid() == 0:
        # root without access means a user namespace (rootless container) that
        # does not map the device's owner -- group membership cannot fix that
        return make_fail(
            "{} is not {} even though this process runs as root".format(
                KFD_DEVICE, "readable" if not readable else "writable"
            ),
            "This happens in rootless containers and user namespaces. Pass the\n"
            "device and keep its group, for example:\n"
            "  podman run --device /dev/kfd --device /dev/dri "
            "--group-add keep-groups ...",
            data,
        )
    return make_fail(
        "{} is not {} by the current user".format(
            KFD_DEVICE,
            "readable" if not readable else "writable",
        ),
        "Add your account to the render and video groups, then re-login:\n"
        "  sudo usermod -a -G render,video $USER",
        data,
    )


@register(
    id="driver.dri-devices",
    group="driver",
    title="DRI render nodes present",
    severity=SEV_WARNING,
    depends=["driver.kfd-device"],
    order=22,
)
def check_dri_devices(accessor):
    nodes = accessor.glob("/dev/dri/renderD*")
    data = {"render_nodes": nodes}
    if not nodes:
        return make_warn(
            "no /dev/dri/renderD* nodes found",
            "In a container, pass the devices through:\n"
            "  docker run --device=/dev/kfd --device=/dev/dri ...",
            data,
        )

    unreadable = []
    for node in nodes:
        if not (accessor.path_readable(node) and accessor.path_writable(node)):
            unreadable.append(node)
    data["inaccessible"] = unreadable
    if unreadable:
        return make_warn(
            "render node(s) not accessible: {}".format(", ".join(unreadable)),
            "sudo usermod -a -G render,video $USER\nthen log out and log back in.",
            data,
        )
    return make_pass("{} render node(s) accessible".format(len(nodes)), "", data)


@register(
    id="driver.render-group",
    group="driver",
    title="User is in the render group",
    severity=SEV_WARNING,
    depends=["driver.kfd-device"],
    order=23,
)
def check_render_group(accessor):
    return _group_check(accessor, "render")


@register(
    id="driver.video-group",
    group="driver",
    title="User is in the video group",
    severity=SEV_WARNING,
    depends=["driver.kfd-device"],
    order=24,
)
def check_video_group(accessor):
    return _group_check(accessor, "video")


@register(
    id="driver.amdgpu-module",
    group="driver",
    title="amdgpu kernel driver loaded",
    severity=SEV_ERROR,
    depends=["driver.kfd-device"],
    order=25,
)
def check_amdgpu_module(accessor):
    # /sys/module/amdgpu exists whenever the driver is present, loadable or
    # built in. Its "version" file does not: only the out-of-tree DKMS driver
    # sets a module version, so the in-tree driver of a distribution kernel
    # has none. Treat the version as optional metadata, never as evidence
    # that the driver is missing.
    data = {"path": AMDGPU_MODULE_DIR}
    if not accessor.path_is_dir(AMDGPU_MODULE_DIR):
        return make_fail(
            "{} does not exist -- the amdgpu driver is not loaded".format(
                AMDGPU_MODULE_DIR
            ),
            "sudo modprobe amdgpu\n" "# if that fails:\n" + DRIVER_INSTALL_HINT,
            data,
        )

    version = accessor.read_file(AMDGPU_VERSION)
    if version is None or not version.strip():
        data["amdgpu_version"] = None
        return make_pass(
            "amdgpu driver loaded (no module version: the in-tree driver of the "
            "distribution kernel, or version metadata not visible here)",
            "",
            data,
        )
    data["amdgpu_version"] = version.strip()
    return make_pass("amdgpu driver loaded, version {}".format(version.strip()), "", data)


@register(
    id="driver.gpu-topology",
    group="driver",
    title="KFD topology enumerates at least one GPU",
    severity=SEV_ERROR,
    depends=["driver.kfd-device"],
    order=26,
)
def check_gpu_topology(accessor):
    node_dirs = accessor.glob(accessor.join(KFD_TOPOLOGY_NODES, "*"))
    data = {"topology_nodes": node_dirs}

    if not node_dirs:
        return make_fail(
            "no KFD topology nodes under {}".format(KFD_TOPOLOGY_NODES),
            DRIVER_INSTALL_HINT,
            data,
        )

    gpus = gpu_nodes(accessor)
    names = []
    for node_dir, props in gpus:
        names.append(
            "{} (gfx_target_version={})".format(
                accessor.basename(node_dir), props.get("gfx_target_version", 0)
            )
        )
    data["gpu_nodes"] = names

    if not gpus:
        return make_fail(
            "{} topology node(s) present but none is a GPU (all report "
            "simd_count 0)".format(len(node_dirs)),
            "Confirm an AMD GPU is installed and visible:\n"
            "  rocm_agent_enumerator\n"
            "If running in a container or cgroup, confirm the GPU is not masked.",
            data,
        )
    return make_pass("{} GPU agent(s): {}".format(len(gpus), ", ".join(names)), "", data)
