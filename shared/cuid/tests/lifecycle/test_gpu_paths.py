# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real symlink/device-number resolution in a private mount namespace.

The only character devices are private instances of /dev/null. The executable
shim supplies DRM device numbers and rejects execution against the host /sys.
"""

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import unittest

BUILD = Path(sys.argv.pop(1)).resolve()
ROOT = BUILD / "lifecycle-state"
spec = importlib.util.spec_from_file_location(
    "vectors", Path(__file__).resolve().parents[1] / "vectors/cuid_vectors.py"
)
vectors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vectors)
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_ulong,
    ctypes.c_char_p,
]
libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]


def mount(source, target, kind=None, flags=0, data=None):
    encoded = [
        s.encode() if s is not None else None for s in (source, target, kind, data)
    ]
    if libc.mount(encoded[0], encoded[1], encoded[2], flags, encoded[3]) != 0:
        raise OSError(ctypes.get_errno(), f"mount {target}")


def namespace():
    if libc.unshare(0x20000) != 0:  # CLONE_NEWNS
        raise OSError(ctypes.get_errno(), "unshare mount namespace")
    mount(None, "/", flags=(1 << 18) | 16384)  # recursive private propagation
    for path, mode in (("/dev", "755"), ("/run", "755"), ("/tmp", "1777")):
        mount("tmpfs", path, "tmpfs", data=f"mode={mode},size=64m")
    for name, minor in (("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)):
        os.mknod(f"/dev/{name}", stat.S_IFCHR | 0o666, os.makedev(1, minor))
        os.chmod(f"/dev/{name}", 0o666)
    Path("/dev/dri").mkdir()


class GpuPaths(unittest.TestCase):
    def setUp(self):
        # An existing directory belongs to another fixture run; never erase it.
        ROOT.mkdir(mode=0o700)
        self.addCleanup(shutil.rmtree, ROOT)
        self.sys = ROOT / "sys"
        self.dri = ROOT / "dri"
        for path in (self.sys, self.dri):
            path.mkdir(mode=0o755)
            path.chmod(0o755)
        self.write(self.sys / ".cuid-path-fixture", "private\n")
        self.devices = {}
        # A captured tree brings its own GPUs.
        if not self._testMethodName.startswith("test_captured"):
            self.gpu("0000:03:00.0", 0, 128, 0x11110001)
            self.gpu("0000:63:00.0", 1, 129, 0x22220002)
        mount(str(self.sys), "/sys", flags=4096)  # bind only private fixture data
        self.addCleanup(self.unmount, "/sys")
        mount(str(self.dri), "/dev/dri", flags=4096)
        self.addCleanup(self.unmount, "/dev/dri")

    @staticmethod
    def unmount(path):
        if libc.umount2(path.encode(), 2) != 0:  # detach after every child finished
            raise OSError(ctypes.get_errno(), f"unmount {path}")

    @staticmethod
    def write(path, text, mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        # umask 077 must not make synthetic public sysfs directories private.
        current = path.parent
        while ROOT in current.parents:
            current.chmod(0o755)
            current = current.parent
        path.write_text(text)
        path.chmod(mode)

    @staticmethod
    def link(path, target):
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        current = path.parent
        while ROOT in current.parents:
            current.chmod(0o755)
            current = current.parent
        path.symlink_to(target)

    def drm(self, device, card, render):
        for name, minor in ((f"card{card}", card), (f"renderD{render}", render)):
            node = device / "drm" / name
            node.mkdir(parents=True, mode=0o755)
            node.chmod(0o755)
            node.parent.chmod(0o755)
            self.link(node / "device", "../..")
            logical = "/sys/" + str(node.relative_to(self.sys))
            self.link(self.sys / "class/drm" / name, logical)
            self.link(self.sys / "dev/char" / f"226:{minor}", logical)
            self.write(node / "dev", f"226:{minor}\n")
            os.mknod(self.dri / name, stat.S_IFCHR | 0o666, os.makedev(1, 3))
            (self.dri / name).chmod(0o666)

    def gpu(self, bdf, card, render, serial):
        device = (
            self.sys / "devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0" / bdf
        )
        self.write(device / "vendor", "0x1002\n")
        self.write(device / "device", "0x73a3\n")
        self.write(device / "revision", "0x01\n")
        self.write(device / "class", "0x030000\n")
        self.write(device / "unique_id", f"{serial:016x}\n", 0o400)
        self.write(device / "config", "0" * 256, 0o600)
        self.link(
            self.sys / "bus/pci/devices" / bdf,
            "/sys/" + str(device.relative_to(self.sys)),
        )
        self.link(device / "subsystem", "/sys/bus/pci")
        self.drm(device, card, render)
        self.devices[bdf] = device

    def publish(self, device, serial, unit=0):
        primary = vectors.pack_primary(serial, unit, 1, 0x73A3, 0x1002, vectors.GPU)
        self.write(
            device / "cuid_primary",
            vectors.uuid_str(vectors.to_uuidv8(primary)) + "\n",
            0o400,
        )
        self.write(device / "cuid_unit_id", f"{unit}\n", 0o444)

    def driver(self):
        for bdf, serial in (("0000:03:00.0", 0x11110001), ("0000:63:00.0", 0x22220002)):
            self.publish(self.devices[bdf], serial)

    def components(self):
        """SMBIOS and a two-port NIC whose functions share one PCIe Device
        Serial Number, so CPU, NIC and Platform are discovered and have a
        serial."""
        self.write(self.sys / ".cuid-path-components", "\n")
        dmi = self.sys / "devices/virtual/dmi/id"
        self.write(
            dmi / "product_uuid", "12345678-9abc-4def-8123-456789abcdef\n", 0o400
        )
        self.write(dmi / "board_vendor", "0x1022\n")
        self.link(self.sys / "class/dmi/id", "/sys/devices/virtual/dmi/id")
        config = bytearray(4096)
        config[0:4] = bytes([0xE4, 0x14, 0x50, 0x17])
        config[0x100:0x104] = bytes([0x03, 0x00, 0x01, 0x00])
        config[0x104:0x10C] = (0x5A5A00C0FFEE0001).to_bytes(8, "little")
        for function in (0, 1):
            bdf = f"0000:07:00.{function}"
            device = self.sys / "devices/pci0000:00" / bdf
            self.write(device / "vendor", "0x14e4\n")
            self.write(device / "device", "0x1750\n")
            self.write(device / "revision", "0x01\n")
            self.write(device / "class", "0x020000\n")
            (device / "config").write_bytes(bytes(config))
            (device / "config").chmod(0o600)
            self.link(
                self.sys / "bus/pci/devices" / bdf,
                "/sys/" + str(device.relative_to(self.sys)),
            )
            net = device / f"net/eth{function}"
            self.link(net / "device", f"../../../{bdf}")
            self.link(
                self.sys / f"class/net/eth{function}",
                "/sys/" + str(net.relative_to(self.sys)),
            )

    def partitions(self, count=8, published=8):
        """Partition 0 on the PCI device and the rest on amdgpu_xcp_<n>
        platform devices, as amdgpu lays them out; MAX_XCP is 8."""
        self.assertLessEqual(count, 8)
        self.parts = []
        for n in range(count):
            if n == 0:
                device = self.devices["0000:03:00.0"]
            else:
                device = self.sys / f"devices/platform/amdgpu_xcp_{n}"
                device.mkdir(parents=True, mode=0o755)
                device.chmod(0o755)
                self.drm(device, 10 + n, 139 + n)
            attr = device / "xcp"
            attr.mkdir(mode=0o755)
            attr.chmod(0o755)
            self.parts.append(attr)
            if n < published:
                self.publish(attr, 0x11110001, vectors.partition_unit_id(count, n))

    def kfd(self, count=8):
        """KFD topology for 0000:03:00.0 and its partitions: one node per
        partition, all with the GPU's unique_id, the partition index ORed into
        location_id's device/function bits as kfd_topology.c does."""
        nodes = self.sys / "class/kfd/kfd/topology/nodes"
        for n in range(count):
            render = 128 if n == 0 else 139 + n
            self.write(
                nodes / str(n + 1) / "properties",
                f"drm_render_minor {render}\ndomain 0\n"
                f"location_id {0x0300 | n}\nunique_id {0x11110001}\n",
            )

    STOCK_PARTITIONS = {"spx": 1, "dpx": 2, "qpx": 4, "cpx": 8}
    STOCK_CPX_GPU_ID = [40001, 20001, 10001, 40002, 40003, 20002, 3001, 40004]
    STOCK_CPX_UNIQUE_ID = [
        3617611476677427204, 10535140504318509070, 10535140504318509072,
        5923454485891121159, 447077339008598017, 10535140504318509076,
        10535140504318509073, 10535140504318509078,
    ]

    def kfd_node(self, index, properties):
        self.write(
            self.sys / "class/kfd/kfd/topology/nodes" / str(index) / "properties",
            "".join(f"{key} {value}\n" for key, value in properties.items()),
        )

    def stock(self, mode):
        """A synthetic MI350X topology: no cuid_* anywhere and no
        key. 0000:03:00.0 is in `mode` and 0000:63:00.0 in SPX. Partition 0 is
        the PCI device's own; every GPU has seven amdgpu_xcp_<n> platform
        devices in every mode, with no link to their GPU; only KFD's topology
        ties an active one back, by render minor, location_id and domain."""
        count = self.STOCK_PARTITIONS[mode]
        topology = self.sys / "class/kfd/kfd/topology"
        self.write(topology / "generation_id", "26\n")
        for cpu in (0, 1):
            self.kfd_node(
                cpu,
                dict(cpu_cores_count=128, simd_count=0, location_id=0, domain=0,
                     drm_render_minor=0),
            )
        node = 2
        for g, (bdf, partitions) in enumerate((("0000:03:00.0", count),
                                                ("0000:63:00.0", 1))):
            device = self.devices[bdf]
            self.write(device / "current_compute_partition",
                       {1: "SPX", 2: "DPX", 4: "QPX", 8: "CPX"}[partitions] + "\n")
            self.write(device / "available_compute_partition", "SPX, DPX, QPX, CPX\n", 0o444)
            self.write(device / "xcp/xcp_metrics", "", 0o444)
            renders = [128 + g]
            for i in range(1, 8):
                n = 7 * g + i - 1
                xcp = self.sys / f"devices/platform/amdgpu_xcp_{n}"
                self.write(xcp / "uevent", f"MODALIAS=platform:amdgpu_xcp_{n}\n")
                self.write(xcp / "modalias", f"platform:amdgpu_xcp_{n}\n")
                self.write(xcp / "xcp/xcp_metrics", "", 0o444)
                self.drm(xcp, 2 + n, 130 + n)
                renders.append(130 + n)
            bus = int(bdf[5:7], 16)
            for i in range(partitions):
                cpx = partitions == 8 and g == 0
                self.kfd_node(
                    node,
                    dict(
                        simd_count=1024 // partitions,
                        location_id=(bus << 8) | (i if partitions > 1 else 0),
                        domain=0,
                        drm_render_minor=renders[i],
                        unique_id=self.STOCK_CPX_UNIQUE_ID[i] if cpx else 0x11110001 + g,
                        num_xcc=8 // partitions,
                        vendor_id=4098,
                        device_id=0x73A3,
                    ),
                )
                self.write(topology / f"nodes/{node}/gpu_id",
                           f"{self.STOCK_CPX_GPU_ID[i] if cpx else 4468 + node}\n")
                node += 1

    @staticmethod
    def machine_id():
        return bytes.fromhex(Path("/etc/machine-id").read_text().strip())

    def temporary(self, bdf, unit, revision=0x01, device=0x73A3, vendor=0x1002):
        """tv_ref's temporary CUID of a PCIe GPU component on this host."""
        segment, bus, rest = bdf.split(":")
        dev, fn = rest.split(".")
        segment = int(segment, 16)
        routing = vectors.routing_id(segment, int(bus, 16), int(dev, 16), int(fn, 16))
        st = vectors.pack_temp_input(vectors.AUX_FORMAT_PCIE, self.machine_id(), routing,
                                     revision, device, vendor, vectors.GPU, unit, segment >> 16)
        return vectors.uuid_str(vectors.temporary(st)[1])

    def listing(self, *lookups, nonroot=False):
        """The probe's GPU listing, {device path: (unit, kind, source, cuid)},
        and {lookup: cuid or status}."""
        rows, found = {}, {}
        for line in self.run_probe("stock", *lookups, nonroot=nonroot).splitlines():
            fields = line.split("\t")
            if fields[0] == "lookup":
                found[fields[1]] = fields[2]
            else:
                rows[fields[0]] = tuple(fields[1:])
        return rows, found

    def stock_scenario(self, mode):
        self.stock(mode)
        count = self.STOCK_PARTITIONS[mode]
        pci = "/sys/bus/pci/devices/0000:03:00.0"
        physical = "/sys/" + str(self.devices["0000:03:00.0"].relative_to(self.sys))
        paths = {i: (physical + "/xcp" if i == 0 else f"/sys/devices/platform/amdgpu_xcp_{i - 1}/xcp")
                 for i in range(count)}
        renders = {i: 128 if i == 0 else 129 + i for i in range(count)}
        expected = {}
        if count > 1:
            expected[pci] = ("0x000", self.temporary("0000:03:00.0", 0))
            for i in range(count):
                unit = vectors.partition_unit_id(count, i)
                expected[paths[i]] = (f"0x{unit:03x}", self.temporary("0000:03:00.0", unit))
        else:
            expected["/sys/class/drm/renderD128"] = ("0x000", self.temporary("0000:03:00.0", 0))
        expected["/sys/class/drm/renderD129"] = ("0x000", self.temporary("0000:63:00.0", 0))
        lookups = ["bdf=0000:03:00.0", pci, "/dev/dri/renderD128", "fd=/dev/dri/renderD128"]
        lookups += [f"/dev/dri/renderD{renders[i]}" for i in range(1, count)]
        lookups += [f"/dev/dri/renderD{130 + n}" for n in range(count - 1, 7)]
        for role in (False, True):
            with self.subTest(mode=mode, nonroot=role):
                rows, found = self.listing(*lookups, nonroot=role)
                got = {path: (row[0], row[3]) for path, row in rows.items()}
                self.assertEqual(got, expected)
                self.assertTrue(all(row[1] == "temporary" and row[2] == "3"
                                    for row in rows.values()), rows)
                self.assertEqual(len({row[3] for row in rows.values()}), len(rows))
                whole = expected[pci if count > 1 else "/sys/class/drm/renderD128"][1]
                self.assertEqual(found["bdf=0000:03:00.0"], whole)
                self.assertEqual(found[pci], whole)
                # A partitioned GPU's render node is its partition 0's.
                zero = expected[paths[0]][1] if count > 1 else whole
                self.assertEqual(found["/dev/dri/renderD128"], zero)
                self.assertEqual(found["fd=/dev/dri/renderD128"], zero)
                for i in range(1, count):
                    self.assertEqual(found[f"/dev/dri/renderD{renders[i]}"],
                                     expected[paths[i]][1])
                # Inactive amdgpu_xcp devices are no component at all.
                for n in range(count - 1, 7):
                    self.assertEqual(found[f"/dev/dri/renderD{130 + n}"], "UNSUPPORTED")
        return expected

    def start(self, *args, nonroot=False, baseline=False, errors=False):
        binary = BUILD / (
            "cuid_gpu_path_baseline" if baseline else "cuid_gpu_path_probe"
        )
        executable = os.memfd_create("cuid-gpu-path", os.MFD_CLOEXEC)
        with binary.open("rb") as source, os.fdopen(os.dup(executable), "wb") as target:
            shutil.copyfileobj(source, target)
        os.fchmod(executable, 0o555)
        env = dict(os.environ)
        options = dict(user=65534, group=65534, extra_groups=[]) if nonroot else {}
        if baseline:
            corpus = Path("/tmp/cuid-path-vectors.txt")
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "vectors/cuid_vectors.txt", corpus
            )
            corpus.chmod(0o644)
            env["AMDCUID_VECTORS_PATH"] = str(corpus)
        try:
            return subprocess.Popen(
                [f"/proc/self/fd/{executable}", *args],
                env=env,
                pass_fds=(executable,),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if baseline or errors else subprocess.DEVNULL,
                text=True,
                **options,
            )
        finally:
            os.close(executable)

    def run_probe(self, *args, nonroot=False):
        child = self.start(*args, nonroot=nonroot, errors=True)
        try:
            out, err = child.communicate(timeout=30)
            self.assertEqual(
                child.returncode,
                0,
                f"{args}: exit {child.returncode}, status {out.strip()} {err.strip()}",
            )
            return out
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_split_gpu_without_kfd_never_names_partition_zero_as_whole(self):
        self.stock("qpx")
        shutil.rmtree(self.sys / "class/kfd")
        for nonroot in (False, True):
            self.run_probe("reject", "/dev/dri/renderD128", "fd=/dev/dri/renderD128",
                           nonroot=nonroot)

    def test_partition_metadata_rejects_malformed_and_overflowing_ids(self):
        self.partitions(count=2, published=2)
        self.kfd(count=2)
        attr = self.parts[1]
        for name, value in (("cuid_unit_id", "8129"), ("cuid_unit_id", "8191"),
                            ("cuid_unit_id", "18446744073709551618"),
                            ("vendor", "0x1002junk"), ("device", "0x173a3"),
                            ("revision", "0x101")):
            path = attr / name if name == "cuid_unit_id" else self.devices["0000:03:00.0"] / name
            original = path.read_text()
            try:
                self.write(path, value + "\n")
                for nonroot in (False, True):
                    with self.subTest(name=name, value=value, nonroot=nonroot):
                        self.run_probe("reject", "/dev/dri/renderD140", nonroot=nonroot)
            finally:
                self.write(path, original)

    def test_whole_gpu_requires_readable_pci_identity(self):
        device = self.devices["0000:03:00.0"]
        (device / "config").unlink()
        for name, value in (("device", "not-hex"), ("device", "0x173a3"),
                            ("revision", ""), ("revision", "0x101")):
            path = device / name
            original = path.read_text()
            try:
                self.write(path, value + "\n")
                for nonroot in (False, True):
                    with self.subTest(name=name, value=value, nonroot=nonroot):
                        self.run_probe("reject", "bdf=0000:03:00.0", "/dev/dri/renderD128",
                                       nonroot=nonroot)
            finally:
                self.write(path, original)

    def test_whole_gpu_spelling_matrix(self):
        physical = "/sys/" + str(self.devices["0000:03:00.0"].relative_to(self.sys))
        forms = [
            "bdf=0000:03:00.0",
            "/dev/dri/renderD128",
            "/sys/class/drm/renderD128",
            "/sys/class/drm/card0",
            "/sys/bus/pci/devices/0000:03:00.0",
            physical,
            physical + "/drm/renderD128",
            "fd=/dev/dri/renderD128",
            "fd=/dev/dri/card0",
        ]
        for driver in (False, True):
            if driver:
                self.driver()
            for nonroot in (False, True):
                with self.subTest(driver=driver, nonroot=nonroot):
                    for first in range(5):
                        order = [
                            forms[first],
                            *[f for f in reversed(forms) if f != forms[first]],
                        ]
                        self.run_probe("sequence", *order, nonroot=nonroot)

    def test_regular_fd_cannot_resolve_as_device_zero(self):
        self.driver()
        self.link(self.sys / "dev/char/0:0", "/sys/class/drm/renderD128")
        for role in (False, True):
            self.run_probe("reject-fd", nonroot=role)

    def test_card_only_and_pci_only_objects(self):
        self.driver()
        device = self.devices["0000:03:00.0"]
        for name, minor in (("renderD128", 128), ("card0", 0)):
            (self.sys / "class/drm" / name).unlink()
            (self.sys / "dev/char" / f"226:{minor}").unlink()
            shutil.rmtree(device / "drm" / name)
            for role in (False, True):
                forms = ["bdf=0000:03:00.0", "/sys/bus/pci/devices/0000:03:00.0"]
                if name == "renderD128":
                    forms += ["/sys/class/drm/card0", "fd=/dev/dri/card0"]
                self.run_probe("sequence", *forms, nonroot=role)

    def test_unrelated_pci_object_is_not_a_gpu(self):
        self.driver()
        self.gpu("0000:05:00.0", 2, 130, 0x33330003)
        device = self.devices["0000:05:00.0"]
        self.write(device / "class", "0x020000\n")
        for name, minor in (("card2", 2), ("renderD130", 130)):
            (self.sys / "class/drm" / name).unlink()
            (self.sys / "dev/char" / f"226:{minor}").unlink()
        shutil.rmtree(device / "drm")
        for role in (False, True):
            self.run_probe("not-gpu", nonroot=role)

    def test_drm_device_link_must_match_its_owning_object(self):
        self.driver()
        children = [
            self.start("wait-reject", "/dev/dri/renderD128", nonroot=role)
            for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            link = self.devices["0000:03:00.0"] / "drm/renderD128/device"
            link.unlink()
            link.symlink_to("/sys/bus/pci/devices/0000:63:00.0")
            for child in children:
                child.communicate("continue\n", timeout=30)
                self.assertEqual(child.returncode, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    def test_partition_paths_and_unpublish(self):
        self.driver()
        self.partitions()
        self.kfd()
        self.run_probe("partitions", "metadata", nonroot=True)
        self.run_probe("partitions", "metadata")
        children = [
            self.start("wait-unpublish", nonroot=role) for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            for attr in self.parts:
                for name in ("cuid_primary", "cuid_unit_id"):
                    (attr / name).unlink(missing_ok=True)
            nodes = self.sys / "class/kfd/kfd/topology/nodes"
            for n in range(2, len(self.parts) + 1):
                shutil.rmtree(nodes / str(n))
            for child in children:
                child.communicate("continue\n", timeout=30)
                self.assertEqual(child.returncode, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    def test_unnamed_vf_cannot_adopt_attributes(self):
        self.driver()
        self.partitions(count=2, published=2)
        forms = [
            "/sys/class/drm/renderD128",
            "/sys/bus/pci/devices/0000:03:00.0",
            "/sys/class/drm/renderD128/device/xcp",
            "/sys/class/drm/renderD140",
        ]
        for state in ("published", "unpublished"):
            if state == "unpublished":
                for device in [*self.devices.values(), *self.parts]:
                    for name in ("cuid_primary", "cuid_unit_id"):
                        (device / name).unlink(missing_ok=True)
            for role in (False, True):
                with self.subTest(state=state, nonroot=role):
                    self.run_probe("unnamed-vf", *forms, nonroot=role)

    def test_components_are_temporary(self):
        self.components()
        for role in (False, True):
            with self.subTest(nonroot=role):
                self.run_probe("components", nonroot=role)
                self.run_probe("nic-functions", nonroot=role)

    def test_other_vendor_display_is_not_a_gpu(self):
        self.driver()
        self.gpu("0000:05:00.0", 2, 130, 0x33330003)
        self.write(self.devices["0000:05:00.0"] / "vendor", "0x1a03\n")
        forms = [
            "bdf=0000:05:00.0",
            "/sys/bus/pci/devices/0000:05:00.0",
            "/dev/dri/renderD130",
            "/sys/class/drm/card2",
            "/sys/class/drm/renderD130",
            "fd=/dev/dri/renderD130",
        ]
        for role in (False, True):
            with self.subTest(nonroot=role):
                self.run_probe("foreign", *forms, nonroot=role)

    def test_whole_gpu_is_temporary(self):
        forms = ["bdf=0000:03:00.0", "/dev/dri/renderD128", "/sys/class/drm/card0"]
        for published in (False, True):
            if published:
                self.driver()
            for role in (False, True):
                with self.subTest(published=published, nonroot=role):
                    self.run_probe("temporary-gpu", *forms, nonroot=role)

    def test_gim_device_without_pci_sysfs_node(self):
        """GIM names two functions sysfs does not show. One is identified from
        GIM's ASIC information; the other, which GIM cannot describe, is not a
        GPU."""
        self.write(
            self.sys / ".cuid-path-gim",
            "0000:03:00.0 -\n"
            "0000:a1:00.0 0x1002 0x74a1 0x05 0x1234abcd\n"
            "0000:b1:00.0 -\n",
        )
        for role in (False, True):
            with self.subTest(nonroot=role):
                self.run_probe("gim", nonroot=role)

    def test_partitions_have_distinct_temporary_cuids(self):
        self.driver()
        self.partitions(count=8)
        self.kfd()
        for role in (False, True):
            with self.subTest(nonroot=role):
                self.run_probe("unpublished-partitions", nonroot=role)

    def test_stock_cpx_partitions_are_temporary_components(self):
        """A stock amdgpu in CPX: the whole GPU and eight partitions, UnitIDs
        29..36 from KFD, nine distinct temporary CUIDs, the same over two
        enumerations."""
        expected = self.stock_scenario("cpx")
        self.assertEqual(len(expected), 10)
        Path(BUILD / "stock-cpx.txt").write_text(
            "".join(f"{path}\t{unit}\t{cuid}\n" for path, (unit, cuid) in sorted(expected.items()))
        )

    def test_stock_dpx_partitions_are_temporary_components(self):
        self.assertEqual(len(self.stock_scenario("dpx")), 4)

    def test_stock_qpx_partitions_are_temporary_components(self):
        self.assertEqual(len(self.stock_scenario("qpx")), 6)

    def test_stock_spx_gpu_is_one_component(self):
        self.assertEqual(len(self.stock_scenario("spx")), 2)

    @staticmethod
    def drm_numbers(names):
        card = next(int(n[4:]) for n in names if n.startswith("card"))
        render = next(int(n[7:]) for n in names if n.startswith("renderD"))
        return card, render

    def captured(self, data):
        """Rebuild a synthetic MI350X tree from partition-topology/<mode>.json:
        PCI functions, amdgpu_xcp_<n> platform devices and KFD topology."""
        for bdf, pci in data["pci"].items():
            device = self.sys / pci["path"][len("/sys/"):]
            for name in ("vendor", "device", "revision", "class", "current_compute_partition"):
                self.write(device / name, pci[name] + "\n")
            self.write(device / "xcp/xcp_metrics", "", 0o444)
            self.link(self.sys / "bus/pci/devices" / bdf, pci["path"])
            self.link(device / "subsystem", "/sys/bus/pci")
            self.drm(device, *self.drm_numbers(pci["drm"]))
            self.devices[bdf] = device
        for name, drm in data["platform"].items():
            device = self.sys / "devices/platform" / name
            self.write(device / "modalias", f"platform:{name}\n")
            self.write(device / "xcp/xcp_metrics", "", 0o444)
            self.drm(device, *self.drm_numbers(drm))
        topology = self.sys / "class/kfd/kfd/topology"
        self.write(topology / "generation_id", data["kfd"]["generation_id"] + "\n")
        for index, node in data["kfd"]["nodes"].items():
            self.write(topology / f"nodes/{index}/properties", node["properties"])
            self.write(topology / f"nodes/{index}/gpu_id", f"{node['gpu_id']}\n")

    # The UnitIDs of each partitioned GPU in the captures.
    CAPTURED = {
        "SPX": {}, "SPX-reload": {}, "SPX-restored": {},
        "DPX-f5": {"0000:f5:00.0": [2, 3]},
        "QPX-f5": {"0000:f5:00.0": [7, 8, 9, 10]},
        "CPX-f5": {"0000:f5:00.0": list(range(29, 37))},
        "CPX-f5-membanks": {"0000:f5:00.0": list(range(29, 37))},
        "CPX-f5-e5": {"0000:f5:00.0": list(range(29, 37)),
                      "0000:e5:00.0": list(range(29, 37))},
    }

    def test_captured_mi350x_trees(self):
        """Discovery against synthetic MI350X topology fixtures."""
        directory = Path(__file__).resolve().parent / "partition-topology"
        for mode, partitioned in sorted(self.CAPTURED.items()):
            self.reset()
            data = json.loads((directory / f"{mode}.json").read_text())
            self.captured(data)
            for role in (False, True):
                with self.subTest(capture=mode, nonroot=role):
                    rows, found = self.listing(
                        *[f"bdf={bdf}" for bdf in data["pci"]], nonroot=role
                    )
                    units = sorted(row[0] for row in rows.values())
                    want = ["0x000"] * len(data["pci"]) + [
                        f"0x{u:03x}" for us in partitioned.values() for u in us
                    ]
                    self.assertEqual(units, sorted(want), rows)
                    self.assertTrue(all(row[1] == "temporary" for row in rows.values()), rows)
                    self.assertEqual(len({row[3] for row in rows.values()}), len(rows))
                    for bdf, pci in data["pci"].items():
                        whole = self.temporary(bdf, 0, 0x00, 0x75A0)
                        self.assertEqual(found[f"bdf={bdf}"], whole)
                        render = [n for n in pci["drm"] if n.startswith("renderD")][0]
                        path = (f"/sys/bus/pci/devices/{bdf}" if bdf in partitioned
                                else f"/sys/class/drm/{render}")
                        self.assertEqual(rows[path][3], whole)
                    for bdf, us in partitioned.items():
                        cuids = {self.temporary(bdf, u, 0x00, 0x75A0) for u in us}
                        self.assertLessEqual(cuids, {row[3] for row in rows.values()})

    def reset(self):
        """Empty the fixture's /sys and /dev/dri, keeping their mounts."""
        for top in (self.sys, self.dri):
            for child in top.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        self.write(self.sys / ".cuid-path-fixture", "private\n")
        self.devices = {}

    def test_existing_cuid_suites_in_private_namespace(self):
        self.driver()
        self.baselines()

    def test_existing_cuid_suites_with_partitions(self):
        self.driver()
        self.partitions(count=8)
        self.baselines()

    def baselines(self):
        for nonroot in (False, True):
            with self.subTest(nonroot=nonroot):
                # The existing tests have fixed /tmp names; isolate each role.
                mount("tmpfs", "/tmp", "tmpfs", data="mode=1777,size=64m")
                child = self.start(baseline=True, nonroot=nonroot)
                try:
                    out, _ = child.communicate(timeout=90)
                    summary = "\n".join(
                        s
                        for s in out.splitlines()
                        if re.match(r"\[  (PASSED|FAILED|SKIPPED)", s)
                    )
                    print(f"\nnamespace baseline nonroot={nonroot}\n{summary}")
                    self.assertEqual(child.returncode, 0)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()


if __name__ == "__main__":
    if os.geteuid() != 0:
        sys.exit(77)
    namespace()
    unittest.main()
