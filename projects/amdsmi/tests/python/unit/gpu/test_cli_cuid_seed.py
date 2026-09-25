#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CUID seed length checks, output leak checks and node-level CLI reporting.

The C library is stubbed: these tests never provision a node. Output checks
reject the synthetic seed and every consecutive eight-byte window in raw or
hex form. Reporting tests use the real CLI and logger across two GPUs.
Binding tests load this tree's py-interface and skip only when libamd_smi.so
cannot be loaded.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

# ``common.common`` bootstraps the real amdsmi package at import time, which
# fails on a stale or mismatched install. The CLI classes below fully stub
# ``amdsmi`` and only need ``amdsmi_path`` to locate the *installed* CLI
# fallback, so degrade gracefully, as test_cli_set_clk_limit.py does. Exception
# rather than ImportError: a stale install raises AttributeError out of
# build_type_lists() rather than failing to import.
try:
    from common.common import amdsmi_path
except Exception:  # pragma: no cover - harness/install unavailable or stale
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _resolve_cli_dir():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        if cli_dir and os.path.isfile(os.path.join(cli_dir, "subcommands", "set_value.py")):
            return cli_dir
    return None


_CLI_DIR = _resolve_cli_dir()
SET_VALUE_PATH = os.path.join(_CLI_DIR, "subcommands", "set_value.py") if _CLI_DIR else ""
_SOURCE_PY_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)
_PY_INTERFACE_PACKAGE = "amdsmi_py_interface_under_test"

SEED_SIZE = 32

# Distinctive, so a leak is unmistakable in a captured stream: every octet is
# unique and none of it is 0x00 or 0xff, which turn up in unrelated output.
SEED_32 = bytes(range(0x40, 0x40 + SEED_SIZE))

# What the stubbed library reports after provisioning. Not the fingerprint of
# SEED_32 (nothing here computes one), so that "the command reported what the
# library told it" cannot be satisfied by a command that computes its own.
PROVISIONED_FINGERPRINT = "1c2d3e4f50617283"


def _install_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``set_value.py`` imports cleanly."""
    amdsmi_pkg = types.ModuleType("amdsmi")
    interface = types.ModuleType("amdsmi.amdsmi_interface")
    exception = types.ModuleType("amdsmi.amdsmi_exception")
    wrapper = types.ModuleType("amdsmi.amdsmi_wrapper")

    # Constants set_value.py binds at import time.
    interface.AMDSMI_MAX_PPT_LIMIT = 0
    interface.AMDSMI_MAX_UTIL = 100
    interface.AMDSMI_CUID_SEED_SIZE = SEED_SIZE
    interface.AMDSMI_CUID_SEED_FINGERPRINT_SIZE = 8
    interface.amdsmi_wrapper = wrapper
    wrapper.AMDSMI_STATUS_API_FAILED = 7
    wrapper.AMDSMI_STATUS_NO_PERM = 10
    wrapper.AMDSMI_STATUS_INVAL = 1
    wrapper.AMDSMI_STATUS_IO = 12

    class _StubLibraryException(Exception):
        """Carries an error code, which the CLI branches on."""

        def __init__(self, err_code=1):
            super().__init__(f"stub amdsmi error {err_code}")
            self.err_code = err_code

        def get_error_code(self):
            return self.err_code

        def get_error_info(self, detailed=True):
            return str(self)

    exception.AmdSmiLibraryException = _StubLibraryException
    interface.AmdSmiLibraryException = _StubLibraryException

    amdsmi_pkg.amdsmi_interface = interface
    amdsmi_pkg.amdsmi_exception = exception

    sys.modules["amdsmi"] = amdsmi_pkg
    sys.modules["amdsmi.amdsmi_interface"] = interface
    sys.modules["amdsmi.amdsmi_exception"] = exception
    sys.modules["amdsmi.amdsmi_wrapper"] = wrapper
    return interface


def _load_set_value_module():
    if _CLI_DIR and _CLI_DIR not in sys.path:
        sys.path.insert(0, _CLI_DIR)
    spec = importlib.util.spec_from_file_location("set_value_cuid_under_test", SET_VALUE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RecordingLogger:
    """``self.logger`` stub that keeps everything the command published.

    ``output`` is the complete input to every renderer (human-readable, JSON and
    CSV are pure functions of it), so a seed octet that is not in here and not in
    the process's own streams cannot appear in any of them.
    """

    def __init__(self):
        self.format = "human"
        self.output = {}
        self.printed = []

    def print_output(self, *args, **kwargs):
        self.printed.append(dict(self.output))

    def store_output(self, device, key, value):
        self.output[key] = value

    def clear_multiple_devices_output(self):
        pass


class _CuidSeedTestBase(unittest.TestCase):
    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
    )

    @classmethod
    def setUpClass(cls):
        if not SET_VALUE_PATH:
            raise unittest.SkipTest("amd-smi CLI set_value.py not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()
        cls.module = _load_set_value_module()

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def setUp(self):
        self.set_calls = []
        self.interface.amdsmi_set_cuid_seed = self.set_calls.append
        # A node whose key an administrator has just set.
        self.interface.amdsmi_get_cuid_seed_info = lambda: {
            "provisioned": True,
            "fingerprint": PROVISIONED_FINGERPRINT,
        }
        self.logger = _RecordingLogger()
        self.cmd = self.module.SetValueCommands()
        self.cmd.logger = self.logger

    def _seed_file(self, payload):
        import tempfile

        handle = tempfile.NamedTemporaryFile(prefix="cuid-seed-", delete=False)
        handle.write(payload)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def _provision(self, payload, from_stdin=False):
        """Drive the real ``_set_cuid_seed`` and return (stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        if from_stdin:
            saved_stdin = sys.stdin
            sys.stdin = types.SimpleNamespace(buffer=io.BytesIO(payload))
            self.addCleanup(setattr, sys, "stdin", saved_stdin)
            source = "-"
        else:
            source = self._seed_file(payload)
        with redirect_stdout(out), redirect_stderr(err):
            self.cmd._set_cuid_seed(source)
        return source, out.getvalue(), err.getvalue()


class TestCuidSeedLengthIsEnforced(_CuidSeedTestBase):
    """A seed that is not exactly 32 octets is refused, and nothing changes."""

    def test_sixteen_octet_seed_is_refused(self):
        # Half a seed: a truncated file, not a weaker secret.
        path = self._seed_file(b"\x01" * 16)
        with self.assertRaises(ValueError) as caught:
            self.cmd._set_cuid_seed(path)
        message = str(caught.exception)
        self.assertIn("exactly 32 bytes", message)
        self.assertIn("got 16", message)
        # The error names the file: that is why the check is duplicated here
        # rather than left to the binding.
        self.assertIn(path, message)
        self.assertEqual(self.set_calls, [], "a refused seed must not reach the library")
        self.assertEqual(self.logger.output, {}, "a refused seed must not report a new state")

    def test_sixty_four_octet_seed_is_refused(self):
        # Two seeds concatenated, or a hex-encoded one saved as bytes. Silently
        # truncating to the first 32 octets would provision something nobody
        # chose.
        path = self._seed_file(b"\x02" * 64)
        with self.assertRaises(ValueError) as caught:
            self.cmd._set_cuid_seed(path)
        message = str(caught.exception)
        self.assertIn("exactly 32 bytes", message)
        self.assertIn("got 64", message)
        self.assertIn(path, message)
        self.assertEqual(self.set_calls, [], "a refused seed must not reach the library")
        self.assertEqual(self.logger.output, {}, "a refused seed must not report a new state")

    def test_short_seed_on_stdin_is_refused(self):
        # stdin is the other accepted source, and the one an operator reaches
        # for when piping from a secret store, so it gets the same check.
        saved_stdin = sys.stdin
        sys.stdin = types.SimpleNamespace(buffer=io.BytesIO(b"\x03" * 16))
        self.addCleanup(setattr, sys, "stdin", saved_stdin)
        with self.assertRaises(ValueError) as caught:
            self.cmd._set_cuid_seed("-")
        self.assertIn("exactly 32 bytes", str(caught.exception))
        self.assertEqual(self.set_calls, [])

    def test_exactly_thirty_two_octets_is_accepted(self):
        # The control. Without it the two refusals above would also pass
        # against a command that refused everything.
        _source, _out, _err = self._provision(SEED_32)
        self.assertEqual(self.set_calls, [SEED_32])
        self.assertEqual(self.logger.output["seed_provisioned"], True)


class TestCuidSeedNeverReachesOutput(_CuidSeedTestBase):
    """Reject whole-seed and eight-byte-window leaks in captured output."""

    def _assert_no_seed_material(self, blob, where):
        self.assertNotIn(SEED_32.hex(), blob.lower(), f"whole seed as hex in {where}")
        self.assertNotIn(SEED_32.decode("latin-1"), blob, f"whole seed verbatim in {where}")
        # Any eight consecutive octets of a 256-bit secret is a quarter of it
        # and enough to confirm a guess, so a partial leak is a leak. Eight is
        # also long enough not to collide with unrelated output by chance.
        for start in range(0, SEED_SIZE - 8 + 1):
            window = SEED_32[start : start + 8]
            self.assertNotIn(
                window.hex(), blob.lower(), f"seed[{start}:{start + 8}] hex in {where}"
            )
            self.assertNotIn(
                window.decode("latin-1"), blob, f"seed[{start}:{start + 8}] verbatim in {where}"
            )

    def test_no_seed_octet_in_any_output_stream(self):
        source, out, err = self._provision(SEED_32)

        # It really was provisioned; otherwise this asserts about nothing.
        self.assertEqual(self.set_calls, [SEED_32])
        self.assertTrue(self.logger.printed, "the command should report the new state")

        for name, blob in (
            ("stdout", out),
            ("stderr", err),
            ("logger.output", repr(self.logger.output)),
            ("logger.output as JSON", json.dumps(self.logger.output, default=repr)),
            ("printed payloads", repr(self.logger.printed)),
        ):
            self._assert_no_seed_material(blob, name)

    def test_no_seed_octet_in_any_output_stream_from_stdin(self):
        _source, out, err = self._provision(SEED_32, from_stdin=True)
        self.assertEqual(self.set_calls, [SEED_32])
        for name, blob in (
            ("stdout", out),
            ("stderr", err),
            ("logger.output", repr(self.logger.output)),
            ("logger.output as JSON", json.dumps(self.logger.output, default=repr)),
        ):
            self._assert_no_seed_material(blob, name)

    def test_what_is_reported_is_the_fingerprint_and_the_state(self):
        # The positive half: the command is useless if it reports nothing, and
        # "nothing was leaked" is trivially true of a command that prints
        # nothing. These two keys, and no third one carrying the secret.
        self._provision(SEED_32)
        self.assertEqual(sorted(self.logger.output), ["seed_fingerprint", "seed_provisioned"])
        self.assertEqual(self.logger.output["seed_fingerprint"], PROVISIONED_FINGERPRINT)


class TestCuidSeedFailureReportsWhatTheNodeHolds(_CuidSeedTestBase):
    """A failed provisioning says whether the seed was stored anyway.

    The key can be stored before the refresh that follows fails, so an error
    does not by itself mean nothing changed.
    """

    def _fail_with(self, code):
        stub_exception = sys.modules["amdsmi.amdsmi_exception"].AmdSmiLibraryException

        def _raise(seed):
            self.set_calls.append(seed)
            raise stub_exception(code)

        self.interface.amdsmi_set_cuid_seed = _raise

    def _node_holds(self, provisioned, fingerprint):
        self.interface.amdsmi_get_cuid_seed_info = lambda: {
            "provisioned": provisioned,
            "fingerprint": fingerprint,
        }

    def test_stored_but_unpublished_seed_is_reported_and_still_fails(self):
        self._fail_with(self.interface.amdsmi_wrapper.AMDSMI_STATUS_IO)
        self._node_holds(True, hashlib.sha256(SEED_32).digest()[:8].hex())
        with self.assertRaises(sys.modules["amdsmi.amdsmi_exception"].AmdSmiLibraryException):
            self._provision(SEED_32)
        self.assertEqual(self.set_calls, [SEED_32])
        self.assertTrue(self.logger.printed, "the stored state must be reported before the error")
        self.assertEqual(
            self.logger.output["seed_fingerprint"], hashlib.sha256(SEED_32).digest()[:8].hex()
        )
        self.assertIs(self.logger.output["seed_provisioned"], True)
        self.assertIn("node key was stored", self.logger.output["seed_publication"])
        self.assertIn("FAILED", self.logger.output["seed_publication"])

    def test_failure_that_left_the_old_seed_reports_nothing(self):
        # The store still holds another key: reporting its fingerprint next to
        # the error would read as the new seed having landed.
        self._fail_with(self.interface.amdsmi_wrapper.AMDSMI_STATUS_API_FAILED)
        self._node_holds(True, PROVISIONED_FINGERPRINT)
        with self.assertRaises(sys.modules["amdsmi.amdsmi_exception"].AmdSmiLibraryException):
            self._provision(SEED_32)
        self.assertEqual(self.logger.output, {})
        self.assertEqual(self.logger.printed, [])

    def test_unprovisioned_store_reports_nothing(self):
        self._fail_with(self.interface.amdsmi_wrapper.AMDSMI_STATUS_API_FAILED)
        self._node_holds(False, hashlib.sha256(SEED_32).digest()[:8].hex())
        with self.assertRaises(sys.modules["amdsmi.amdsmi_exception"].AmdSmiLibraryException):
            self._provision(SEED_32)
        self.assertEqual(self.logger.output, {})

    def test_lack_of_privilege_is_named_and_reports_nothing(self):
        self._fail_with(self.interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM)
        self._node_holds(True, hashlib.sha256(SEED_32).digest()[:8].hex())
        with self.assertRaises(PermissionError):
            self._provision(SEED_32)
        self.assertEqual(self.logger.output, {})

    def test_refused_seed_is_named_and_reports_nothing(self):
        self._fail_with(self.interface.amdsmi_wrapper.AMDSMI_STATUS_INVAL)
        self._node_holds(True, PROVISIONED_FINGERPRINT)
        with self.assertRaisesRegex(ValueError, "refused"):
            self._provision(SEED_32)
        self.assertEqual(self.logger.output, {})


class _BindingTestBase(unittest.TestCase):
    """This tree's Python binding, loaded against the built libamd_smi.so."""

    @classmethod
    def setUpClass(cls):
        # This tree's binding under a private package name, so neither an
        # installed amdsmi nor the stub the CLI classes register can stand in.
        # Loaded without the package __init__, which needs the generated
        # _version.py.
        if not os.path.isfile(os.path.join(_SOURCE_PY_INTERFACE_DIR, "amdsmi_interface.py")):
            raise unittest.SkipTest(f"no py-interface at {_SOURCE_PY_INTERFACE_DIR}")
        package = types.ModuleType(_PY_INTERFACE_PACKAGE)
        package.__path__ = [_SOURCE_PY_INTERFACE_DIR]
        sys.modules[_PY_INTERFACE_PACKAGE] = package
        try:
            interface = importlib.import_module(_PY_INTERFACE_PACKAGE + ".amdsmi_interface")
            exception = importlib.import_module(_PY_INTERFACE_PACKAGE + ".amdsmi_exception")
        except Exception:
            cls._forget_package()
            raise
        if interface.amdsmi_wrapper._loaded_lib_path is None:
            cls._forget_package()
            raise unittest.SkipTest("libamd_smi.so could not be loaded")
        cls.interface = interface
        cls.parameter_exception = exception.AmdSmiParameterException

    @classmethod
    def tearDownClass(cls):
        cls._forget_package()

    @staticmethod
    def _forget_package():
        for name in [n for n in sys.modules if n.split(".")[0] == _PY_INTERFACE_PACKAGE]:
            del sys.modules[name]


class TestCuidSeedLengthEnforcedInTheBinding(_BindingTestBase):
    """The same refusal in the Python binding, below the CLI.

    A binding caller never runs ``_set_cuid_seed``, so with the check only in the
    CLI a script calling ``amdsmi.amdsmi_set_cuid_seed(b"...")`` would hand a
    short buffer to a C entry point that reads 32 octets from it.
    """

    def setUp(self):
        self.calls = []

        def _record(buffer):
            # Nothing is provisioned here: a real one re-keys the whole node.
            self.calls.append(bytes(buffer))
            return 0  # AMDSMI_STATUS_SUCCESS

        # create=True: the loaded libamd_smi.so may predate the symbol, and the
        # check under test runs before the binding is reached.
        patcher = mock.patch.object(
            self.interface.amdsmi_wrapper, "amdsmi_set_cuid_seed", _record, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_wrong_length_seeds_are_refused_before_the_library(self):
        for length in (0, 16, 31, 33, 64):
            with self.subTest(length=length):
                with self.assertRaises(self.parameter_exception):
                    self.interface.amdsmi_set_cuid_seed(b"\x05" * length)
        self.assertEqual(self.calls, [], "a refused seed must not reach the library")

    def test_thirty_two_octets_reaches_the_library_unchanged(self):
        self.interface.amdsmi_set_cuid_seed(SEED_32)
        self.assertEqual(self.calls, [SEED_32])


class TestCuidComponentsBinding(_BindingTestBase):
    """amdsmi_get_cuid_components decodes every entry, and lists a component
    that appears between its count call and its fill call."""

    def _fake(self, totals):
        """Stand in for the C call; totals[k] is the node's size at call k."""
        wrapper = self.interface.amdsmi_wrapper
        self.calls = []

        def _entry(count_ref, components):
            total = totals[min(len(self.calls), len(totals) - 1)]
            self.calls.append(bool(components))
            count = count_ref._obj
            capacity = count.value if components else 0
            for i in range(min(capacity, total)):
                entry = components[i]
                entry.info.derived = f"0000000{i}-0000-8000-8000-000000000000".encode()
                entry.info.component_type = wrapper.AMDSMI_CUID_COMPONENT_GPU
                entry.info.source = wrapper.AMDSMI_CUID_SOURCE_DRIVER
                entry.bdf = f"0000:0{i}:00.0".encode()
                entry.vendor_id = 0x1002
            count.value = total
            if components and capacity < total:
                return wrapper.AMDSMI_STATUS_INSUFFICIENT_SIZE
            return wrapper.AMDSMI_STATUS_SUCCESS

        patcher = mock.patch.object(wrapper, "amdsmi_get_cuid_components", _entry, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_entries_are_decoded(self):
        self._fake([2])
        components = self.interface.amdsmi_get_cuid_components()
        self.assertEqual([c["bdf"] for c in components], ["0000:00:00.0", "0000:01:00.0"])
        self.assertEqual(components[1]["derived"], "00000001-0000-8000-8000-000000000000")
        self.assertEqual(components[0]["component_type"], "GPU")
        self.assertEqual(components[0]["source"], "DRIVER")
        self.assertIs(components[0]["auxiliary"], False)
        self.assertEqual(components[0]["vendor_id"], 0x1002)
        self.assertEqual(components[0]["primary"], "")

    def test_a_component_appearing_between_calls_is_listed(self):
        self._fake([1, 2])
        components = self.interface.amdsmi_get_cuid_components()
        self.assertEqual(self.calls, [False, True, False, True])
        self.assertEqual(len(components), 2)


class _FakeHelpers:
    """The slice of ``AMDSMIHelpers`` that `static` and the logger reach for.

    ``handle_gpus`` delegates to the real implementation: it is the loop that
    turns one invocation into one ``static_gpu`` call per device, and a
    reimplementation of it here would be the very thing under test.
    """

    def __init__(self, helpers_module):
        self._real = helpers_module.AMDSMIHelpers

    def handle_gpus(self, args, logger, subcommand):
        return self._real.handle_gpus(self, args, logger, subcommand)

    def get_gpu_cuid_info(self, device_handle, include_primary=False):
        return self._real.get_gpu_cuid_info(self, device_handle, include_primary)

    def get_cuid_seed_state(self):
        return self._real.get_cuid_seed_state()

    def _get_driver_cuid_seed_state(self, device_handle, cuid):
        return self._real._get_driver_cuid_seed_state(self, device_handle, cuid)

    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle

    def os_info(self):
        return ("linux", "x86_64")

    def check_required_groups(self):
        pass

    def is_linux(self):
        return True

    def is_baremetal(self):
        return True

    def is_virtual_os(self):
        return False

    def is_hypervisor(self):
        return False

    def is_amd_hsmp_initialized(self):
        return False

    def is_amdgpu_initialized(self):
        return True


class TestStaticCuidOutputShape(unittest.TestCase):
    """`amd-smi static --cuid`: the fixed field names, and one seed per node."""

    # Two of them, because "once for the invocation" and "once per GPU" are the
    # same output on a one-GPU node. Neither is 0: static_gpu treats a falsy
    # device handle as "no device selected".
    GPU_HANDLES = [1, 2]

    _SAVED_MODULE_NAMES = (
        "amdsmi",
        "amdsmi.amdsmi_interface",
        "amdsmi.amdsmi_exception",
        "amdsmi.amdsmi_wrapper",
        "amdsmi_init",
        "amdsmi_helpers",
        "amdsmi_logger",
        "amdsmi_cli_exceptions",
        "BDF",
    )

    @classmethod
    def setUpClass(cls):
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI not found (source or installed)")
        cls._saved_modules = {name: sys.modules.get(name) for name in cls._SAVED_MODULE_NAMES}
        cls.interface = _install_fake_amdsmi()

        # amdsmi_helpers pulls in amdsmi_init, which initialises the real
        # library at import time and exits the interpreter when no driver is
        # loaded. Stand in for it with the two names amdsmi_helpers reads.
        fake_init = types.ModuleType("amdsmi_init")
        fake_init.AMDSMI_INIT_FLAG = 0
        fake_init.AMD_VENDOR_ID = 0x1002
        fake_init.amdsmi_interface = cls.interface
        fake_init.amdsmi_exception = sys.modules["amdsmi.amdsmi_exception"]
        sys.modules["amdsmi_init"] = fake_init

        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)
        import amdsmi_helpers
        import amdsmi_logger

        cls.helpers_module = amdsmi_helpers
        cls.logger_module = amdsmi_logger

        spec = importlib.util.spec_from_file_location(
            "static_cuid_under_test", os.path.join(_CLI_DIR, "subcommands", "static.py")
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @classmethod
    def tearDownClass(cls):
        for name, saved in cls._saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved

    def setUp(self):
        self.seed_calls = []
        self.interface.amdsmi_get_cuid_seed_info = self._seed_info
        self.interface.amdsmi_get_gpu_cuid_info = lambda handle: {
            "primary": "",
            "derived": f"deadbeef-0000-8000-0000-00000000000{handle}",
            "component_type": "GPU",
            "auxiliary": False,
            "source": "DRIVER",
        }
        self.interface.amdsmi_get_gpu_asic_info = lambda handle: {}

    def _seed_info(self):
        self.seed_calls.append(1)
        return {"provisioned": True, "fingerprint": PROVISIONED_FINGERPRINT}

    def _run(self, output_format, **arg_overrides):
        """Run one `amd-smi static` invocation and return what it printed."""
        helpers = _FakeHelpers(self.helpers_module)
        command = self.module.StaticCommands()
        command.helpers = helpers
        command.logger = self.logger_module.AMDSMILogger(
            format=output_format, destination="stdout", helpers=helpers
        )
        command.group_check_printed = True
        command.device_handles = list(self.GPU_HANDLES)
        command.cpu_handles = []
        # NIC discovery probes the host; this invocation is about GPUs.
        command._static_nics = lambda *args, **kwargs: False

        selectors = dict.fromkeys(
            (
                "asic",
                "bus",
                "vbios",
                "driver",
                "ras",
                "vram",
                "cache",
                "board",
                "process_isolation",
                "clock",
                "mem_carveout",
                "partition",
                "limit",
                "soc_pstate",
                "xgmi_plpd",
                "profile",
                "numa",
                "dfc_ucode",
                "fb_info",
                "num_vf",
                "cuid",
                "cuid_primary",
            ),
            False,
        )
        selectors.update(gpu=None, cpu=None, nic=None)
        selectors.update(arg_overrides)

        captured = io.StringIO()
        with redirect_stdout(captured):
            command.static(argparse.Namespace(**selectors))
        return captured.getvalue()

    def test_json_reports_the_seed_once_at_the_top_level(self):
        document = json.loads(self._run("json", cuid=True))

        self.assertEqual(document["seed_provisioned"], True)
        self.assertEqual(document["seed_fingerprint"], PROVISIONED_FINGERPRINT)
        self.assertEqual(len(document["gpu_data"]), len(self.GPU_HANDLES))
        for gpu_block in document["gpu_data"]:
            self.assertNotIn("cuid_seed", gpu_block)
            self.assertNotIn("seed_provisioned", gpu_block)
            self.assertNotIn("seed_fingerprint", gpu_block)
            self.assertNotIn("seed_provisioned", gpu_block["cuid"])
            self.assertNotIn("seed_fingerprint", gpu_block["cuid"])

    def test_json_gpu_blocks_carry_identity_and_observation_metadata(self):
        document = json.loads(self._run("json", cuid=True))

        for gpu_block in document["gpu_data"]:
            self.assertEqual(
                sorted(gpu_block["cuid"]),
                [
                    "auxiliary",
                    "component_type",
                    "cuid_metadata_status",
                    "derived_cuid",
                    "effective_seed",
                    "identifier_kind",
                    "primary_cuid",
                    "source",
                ],
            )
            self.assertEqual(gpu_block["cuid"]["effective_seed"], "unknown")
        self.assertNotIn("seed_info_scope", document)

    def test_the_library_is_asked_for_the_seed_once_for_the_invocation(self):
        # Nothing here is memoised, so a second read would mean a second
        # emission.
        self._run("json", cuid=True)
        self.assertEqual(len(self.seed_calls), 1)

    def test_human_readable_prints_the_seed_once_for_two_gpus(self):
        printed = self._run("human_readable", cuid=True)

        self.assertEqual(printed.count("SEED_PROVISIONED"), 1)
        self.assertEqual(printed.count("SEED_FINGERPRINT"), 1)
        self.assertEqual(printed.count("DERIVED_CUID"), len(self.GPU_HANDLES))
        self.assertIn(PROVISIONED_FINGERPRINT, printed)
        self.assertNotIn("CUID_SEED", printed)

    def test_csv_carries_the_seed_names_as_columns_once(self):
        printed = self._run("csv", cuid=True)

        self.assertEqual(printed.count("seed_provisioned,seed_fingerprint"), 1)
        self.assertIn(f"True,{PROVISIONED_FINGERPRINT}", printed)
        # The per-device table is its own block and keeps the per-device names.
        self.assertIn("gpu,derived_cuid,primary_cuid,component_type,auxiliary,source", printed)

    def test_cuid_primary_alone_still_reports_the_seed(self):
        # --cuid-primary selects a field of the CUID block, so it implies the
        # block, and the block includes the node's seed state.
        document = json.loads(self._run("json", cuid_primary=True))

        self.assertEqual(document["seed_fingerprint"], PROVISIONED_FINGERPRINT)

    def test_no_seed_without_the_cuid_block(self):
        document = json.loads(self._run("json", asic=True))

        self.assertNotIn("seed_provisioned", document)
        self.assertNotIn("seed_fingerprint", document)
        self.assertEqual(self.seed_calls, [])
        for gpu_block in document["gpu_data"]:
            self.assertNotIn("cuid", gpu_block)

    def test_an_unreadable_seed_says_so_rather_than_reading_unprovisioned(self):
        # A caller without the privilege to read the seed store learns nothing
        # about the node. "False" would be a claim, and the wrong one.
        no_perm = self.interface.AmdSmiLibraryException(
            self.interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        )

        def _refuse():
            self.seed_calls.append(1)
            raise no_perm

        self.interface.amdsmi_get_cuid_seed_info = _refuse
        document = json.loads(self._run("json", cuid=True))

        self.assertEqual(document["seed_provisioned"], "N/A (requires root)")
        self.assertEqual(document["seed_fingerprint"], "N/A (requires root)")


if __name__ == "__main__":
    unittest.main()
