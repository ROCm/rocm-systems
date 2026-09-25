# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real symlink/device-number resolution in a private mount namespace.

The only character devices are private instances of /dev/null. The executable
shim supplies DRM device numbers and rejects execution against the host /sys.
"""

import ctypes
import hashlib
import importlib.util
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
A = b"A" * 32
B = bytes(0x40 + 3 * n for n in range(32))
EFIVAR = "firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d"
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

    def publish(self, device, serial, unit=0, key=A):
        primary = vectors.pack_primary(serial, unit, 1, 0x73A3, 0x1002, vectors.GPU)
        _, derived = vectors.derive(key, primary)
        self.write(
            device / "cuid_primary",
            vectors.uuid_str(vectors.to_uuidv8(primary)) + "\n",
            0o400,
        )
        self.write(
            device / "cuid_derived",
            vectors.uuid_str(vectors.to_uuidv8(derived)) + "\n",
            0o444,
        )

    def driver(self, key=A, state=None):
        for bdf, serial in (("0000:03:00.0", 0x11110001), ("0000:63:00.0", 0x22220002)):
            device = self.devices[bdf]
            self.publish(device, serial, key=key)
            (device / "cuid_seed").write_bytes(key)
            (device / "cuid_seed").chmod(0o600)
            if state:
                self.write(device / "cuid_seed_state", state + "\n")

    def efivar(self, key, flags=1, version=1, mode=0o600):
        """AmdCuidKey in the fixture's /sys, never the host's efivarfs."""
        path = self.sys / EFIVAR
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        path.write_bytes(bytes([7, 0, 0, 0, version, flags, 0, 0]) + key)
        path.chmod(mode)
        return path

    def components(self):
        """SMBIOS and one NIC with a PCIe Device Serial Number, so CPU, NIC and
        Platform are discovered and have a serial."""
        self.write(self.sys / ".cuid-path-components", "\n")
        dmi = self.sys / "devices/virtual/dmi/id"
        self.write(
            dmi / "product_uuid", "4c4c4544-0042-3510-8052-b2c04f4e3332\n", 0o400
        )
        self.write(dmi / "board_vendor", "0x1022\n")
        self.link(self.sys / "class/dmi/id", "/sys/devices/virtual/dmi/id")
        bdf = "0000:07:00.0"
        device = self.sys / "devices/pci0000:00" / bdf
        self.write(device / "vendor", "0x14e4\n")
        self.write(device / "device", "0x1750\n")
        self.write(device / "revision", "0x01\n")
        self.write(device / "class", "0x020000\n")
        config = bytearray(4096)
        config[0:4] = bytes([0xE4, 0x14, 0x50, 0x17])
        config[0x100:0x104] = bytes([0x03, 0x00, 0x01, 0x00])
        config[0x104:0x10C] = (0x5A5A00C0FFEE0001).to_bytes(8, "little")
        (device / "config").write_bytes(bytes(config))
        (device / "config").chmod(0o600)
        self.link(
            self.sys / "bus/pci/devices" / bdf,
            "/sys/" + str(device.relative_to(self.sys)),
        )
        net = device / "net/eth0"
        self.link(net / "device", f"../../../{bdf}")
        self.link(self.sys / "class/net/eth0", "/sys/" + str(net.relative_to(self.sys)))

    def partitions(self, count=64, published=8, collide=False):
        self.parts = []
        for n in range(count):
            if n == 0:
                device = self.devices["0000:03:00.0"]
            else:
                device = self.sys / f"devices/platform/amdgpu_xcp.{n}"
                device.mkdir(parents=True, mode=0o755)
                device.chmod(0o755)
                self.drm(device, 10 + n, 139 + n)
            attr = device / "xcp"
            attr.mkdir(mode=0o755)
            attr.chmod(0o755)
            self.parts.append(attr)
            if n < published:
                self.publish(attr, 0x11110001, 0 if collide else 0x40 | n)

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

    def test_driver_rekey_then_fresh_paths(self):
        for nonroot in (False, True):
            with self.subTest(nonroot=nonroot):
                self.driver()
                child = self.start("wait-rotate", nonroot=nonroot)
                try:
                    self.assertEqual(child.stdout.readline().strip(), "ready")
                    self.run_probe("rotate")
                    child.communicate("continue\n", timeout=30)
                    self.assertEqual(child.returncode, 0)
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait()

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

    def test_failed_collision_does_not_poison_prior_devices(self):
        self.driver()
        children = [
            self.start("wait-reject", "/dev/dri/renderD130", nonroot=role)
            for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            self.gpu("0000:05:00.0", 2, 130, 0x11110001)
            self.publish(self.devices["0000:05:00.0"], 0x11110001)
            for child in children:
                child.communicate("continue\n", timeout=30)
                self.assertEqual(child.returncode, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()

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

    def test_whole_and_partition_zero_collision_is_not_an_alias(self):
        self.driver()
        children = [
            self.start(
                "wait-reject", "/sys/bus/pci/devices/0000:03:00.0/xcp", nonroot=role
            )
            for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            self.partitions(count=1, published=1, collide=True)
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
        self.run_probe("partitions", "unknown-metadata", nonroot=True)
        self.run_probe("partitions", "metadata")
        children = [
            self.start("wait-unpublish", nonroot=role) for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            for attr in self.parts:
                for name in ("cuid_primary", "cuid_derived"):
                    (attr / name).unlink(missing_ok=True)
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
                    for name in ("cuid_primary", "cuid_derived"):
                        (device / name).unlink(missing_ok=True)
            for role in (False, True):
                with self.subTest(state=state, nonroot=role):
                    self.run_probe("unnamed-vf", *forms, nonroot=role)

    def test_distinct_xcp_objects_with_the_same_parent_are_not_aliases(self):
        self.driver()
        self.partitions(count=2, published=1)
        target = "/sys/devices/platform/amdgpu_xcp.1/xcp"
        children = [
            self.start("wait-reject-xcp", target, nonroot=role)
            for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            self.publish(
                self.parts[1], 0x11110001, 0x40
            )  # Same ID as partition 0, different object.
            for child in children:
                child.communicate("continue\n", timeout=30)
                self.assertEqual(child.returncode, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    def test_set_key_writes_efivarfs_without_amdgpu(self):
        self.components()
        (self.sys / EFIVAR).parent.mkdir(parents=True, mode=0o755)
        for existing in (False, True):
            with self.subTest(existing=existing):
                (self.sys / EFIVAR).unlink(missing_ok=True)
                if existing:
                    self.efivar(A, flags=0, mode=0o644)
                self.run_probe(
                    "key-info",
                    "KEY_ERROR" if not existing else "SUCCESS",
                    hashlib.sha256(A).hexdigest()[:16],
                    "0",
                )
                self.run_probe(
                    "components", "temporary" if not existing else "keyed", A.hex()
                )
                self.run_probe("set-key", B.hex())
                self.assertEqual(
                    (self.sys / EFIVAR).read_bytes(),
                    bytes([7, 0, 0, 0, 1, 1, 0, 0]) + B,
                )
                self.assertEqual(
                    stat.S_IMODE((self.sys / EFIVAR).stat().st_mode), 0o600
                )
                self.run_probe(
                    "key-info", "SUCCESS", hashlib.sha256(B).hexdigest()[:16], "1"
                )
                self.run_probe("components", "keyed", B.hex())
                self.run_probe("components", "temporary", nonroot=True)

    def test_node_key_sources(self):
        self.components()
        fingerprint = {k: hashlib.sha256(k).hexdigest()[:16] for k in (A, B)}
        cases = [
            # (cuid_seed state or None for no amdgpu, efivar key/flags, expected)
            ("provisioned", (B, 0), (A, "1")),
            ("unprovisioned", (B, 1), (A, "0")),
            (None, (B, 1), (B, "1")),
            (None, (B, 0), (B, "0")),
        ]
        for state, (efikey, flags), (key, provisioned) in cases:
            with self.subTest(state=state, flags=flags):
                if state:
                    self.driver(key=A, state=state)
                self.efivar(efikey, flags=flags)
                try:
                    self.run_probe("key-info", "SUCCESS", fingerprint[key], provisioned)
                    self.run_probe("components", "keyed", key.hex())
                    self.run_probe("key-info", "PERMISSION_DENIED", nonroot=True)
                finally:
                    for device in self.devices.values():
                        for name in (
                            "cuid_seed",
                            "cuid_seed_state",
                            "cuid_primary",
                            "cuid_derived",
                        ):
                            (device / name).unlink(missing_ok=True)

    def test_malformed_key_variable_leaves_no_key(self):
        self.components()
        for version, flags in ((2, 1), (1, 0x80)):
            with self.subTest(version=version, flags=flags):
                self.efivar(B, flags=flags, version=version)
                self.run_probe("key-info", "KEY_ERROR")
                self.run_probe("components", "temporary")
        (self.sys / EFIVAR).write_bytes(bytes([7, 0, 0, 0, 1, 1, 0, 0]) + B[:31])
        self.run_probe("key-info", "KEY_ERROR")
        self.run_probe("components", "temporary")

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

    def test_unpublished_whole_gpu_is_temporary(self):
        forms = ["bdf=0000:03:00.0", "/dev/dri/renderD128", "/sys/class/drm/card0"]
        for primary_only in (False, True):
            if primary_only:
                self.publish(self.devices["0000:03:00.0"], 0x11110001)
                (self.devices["0000:03:00.0"] / "cuid_derived").unlink()
            for role in (False, True):
                with self.subTest(primary_only=primary_only, nonroot=role):
                    self.run_probe("temporary-gpu", *forms, nonroot=role)

    def test_temporary_gpu_takes_the_driver_value_once_published(self):
        primary = vectors.pack_primary(0x11110001, 0, 1, 0x73A3, 0x1002, vectors.GPU)
        _, derived = vectors.derive(A, primary)
        expected = vectors.uuid_str(vectors.to_uuidv8(derived))
        children = [
            self.start("wait-publish", expected, nonroot=role) for role in (False, True)
        ]
        try:
            for child in children:
                self.assertEqual(child.stdout.readline().strip(), "ready")
            self.publish(self.devices["0000:03:00.0"], 0x11110001)
            for child in children:
                child.communicate("continue\n", timeout=30)
                self.assertEqual(child.returncode, 0)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()

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
