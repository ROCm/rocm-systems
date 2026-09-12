#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for CUID seed provisioning and reporting.

Three requirements, tested at the layers that enforce them.

**A seed of any length but 32 octets is refused, and nothing is provisioned.**
``amdsmi_set_cuid_seed()`` takes ``const uint8_t[AMDSMI_CUID_SEED_SIZE]`` and so
has no length to check. The check lives twice above it, in the CLI (so the error
names the file the operator passed) and in the Python binding (so a binding
caller cannot bypass the CLI), and both are exercised here.

**No octet of the seed reaches any output stream.** ``amd-smi`` output is pasted
into tickets, so the provisioning command must report the fingerprint and never
the secret. The library call is stubbed out, so nothing here provisions
anything; a real provisioning re-keys the whole node.

**The seed's state is reported once for the invocation, under the names the
machine-readable contract fixes.** The seed is a property of the node, so
``amd-smi static --cuid`` reports it beside the per-GPU blocks rather than
inside each of them, as ``seed_provisioned`` and ``seed_fingerprint``. Those
tests drive the real ``StaticCommands`` and ``AMDSMILogger`` across two GPUs.

Every class stubs the C library, so they run without GPU hardware and without a
compiled ``amdsmi``. The binding-level class needs a real importable ``amdsmi``
and skips when there is none.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout

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

SEED_SIZE = 32

# Distinctive, so a leak is unmistakable in a captured stream: every octet is
# unique and none of it is 0x00 or 0xff, which turn up in unrelated output.
SEED_32 = bytes(range(0x40, 0x40 + SEED_SIZE))

# What the stubbed library reports after provisioning. Neither the fingerprint
# of SEED_32 (nothing here computes one) nor the canonical fallback fingerprint
# be8937fba7ed4e6f, so that "the command reported what the library told it"
# cannot be satisfied by a command that reports a constant.
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
    interface.amdsmi_wrapper = wrapper
    wrapper.AMDSMI_STATUS_NO_PERM = 10

    class _StubLibraryException(Exception):
        """Carries an error code, which the CLI branches on."""

        def __init__(self, err_code=1):
            super().__init__(f"stub amdsmi error {err_code}")
            self.err_code = err_code

        def get_error_code(self):
            return self.err_code

        def get_error_info(self):
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
        # A node that has just been provisioned, and therefore *not*
        # fingerprinting the public fallback seed. Not be8937fba7ed4e6f, the
        # canonical fallback fingerprint: a stub echoing that constant would
        # also be satisfied by a command printing it from its own source.
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
    """No octet of a provisioned seed appears in anything the command emits."""

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


class TestCuidSeedLengthEnforcedInTheBinding(unittest.TestCase):
    """The same refusal in the Python binding, below the CLI.

    A binding caller never runs ``_set_cuid_seed``, so with the check only in the
    CLI a script calling ``amdsmi.amdsmi_set_cuid_seed(b"...")`` would hand a
    short buffer to a C entry point that reads 32 octets from it.
    """

    @classmethod
    def setUpClass(cls):
        # The real package, not the stub the CLI classes install. The sibling
        # classes restore sys.modules in tearDownClass, so whichever runs first,
        # this import is the real one.
        try:
            from amdsmi import amdsmi_exception, amdsmi_interface
        except Exception as e:  # pragma: no cover - no amdsmi installed
            raise unittest.SkipTest(f"amdsmi package not importable: {e}")
        if not isinstance(getattr(amdsmi_interface, "__file__", None), str):
            raise unittest.SkipTest("a stubbed amdsmi is loaded in this interpreter")
        if not hasattr(amdsmi_interface, "amdsmi_set_cuid_seed") or not hasattr(
            amdsmi_interface.amdsmi_wrapper, "amdsmi_set_cuid_seed"
        ):
            # An amdsmi from before this change does not carry the check under
            # test.
            raise unittest.SkipTest(
                f"installed amdsmi ({amdsmi_interface.__file__}) predates the CUID seed API"
            )
        cls.interface = amdsmi_interface
        cls.parameter_exception = amdsmi_exception.AmdSmiParameterException

    def setUp(self):
        self.calls = []
        self.saved = self.interface.amdsmi_wrapper.amdsmi_set_cuid_seed

        def _record(buffer):
            # Nothing is provisioned here: a real one re-keys the whole node.
            self.calls.append(bytes(buffer))
            return 0  # AMDSMI_STATUS_SUCCESS

        self.interface.amdsmi_wrapper.amdsmi_set_cuid_seed = _record
        self.addCleanup(setattr, self.interface.amdsmi_wrapper, "amdsmi_set_cuid_seed", self.saved)

    def test_wrong_length_seeds_are_refused_before_the_library(self):
        for length in (0, 16, 31, 33, 64):
            with self.subTest(length=length):
                with self.assertRaises(self.parameter_exception):
                    self.interface.amdsmi_set_cuid_seed(b"\x05" * length)
        self.assertEqual(self.calls, [], "a refused seed must not reach the library")

    def test_thirty_two_octets_reaches_the_library_unchanged(self):
        self.interface.amdsmi_set_cuid_seed(SEED_32)
        self.assertEqual(self.calls, [SEED_32])


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

    def test_json_gpu_blocks_carry_the_five_per_device_names(self):
        document = json.loads(self._run("json", cuid=True))

        for gpu_block in document["gpu_data"]:
            self.assertEqual(
                sorted(gpu_block["cuid"]),
                ["auxiliary", "component_type", "derived_cuid", "primary_cuid", "source"],
            )

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
