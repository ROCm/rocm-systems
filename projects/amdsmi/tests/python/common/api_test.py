#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Declarative per-API driver for the unit suites.

Builds on the shared session in :mod:`common.common`: each test declares its
parameters and the driver derives the invalid-argument and valid-argument calls
from them. Imports flow one way, this module depends on ``common`` and never
the reverse.
"""

import ctypes
import itertools
import os
import unittest

from common.common import PASS, VERBOSITY_VERBOSE, Common, amdsmi, verbose

BAD_HANDLE = None
BAD_INT = "not-an-int"
BAD_STR = 0
BAD_ENUM = 0xDEAD
BAD_SEQUENCE = 0
BAD_BYTES = 0
OUT_OF_RANGE = 0xFFFFFFFF

# ctypes refuses an argument it cannot marshal, so the C entry point is never
# reached; that is still a rejection.
_BINDING_REJECTIONS = (ctypes.ArgumentError, TypeError, ValueError, OverflowError)

# ctypes converts any object to c_bool by truthiness, so a wrong-typed value
# silently becomes True and the call proceeds to the device. Such a parameter
# cannot be reject-tested at all.
_UNREJECTABLE_CTYPES = (ctypes.c_bool,)

# Fixed-width markers the library reports in place of a reading it could not take.
_SENTINEL_INTS = frozenset({0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF})
_SENTINEL_STRS = frozenset({"N/A", ""})

_UNEXPECTED_SUCCESS = (
    "\tTEST FAILURE, AMDSMI API Returned  0, AMDSMI_STATUS_SUCCESS\n"
    "\t              AMDSMI API Expected a rejection"
)


class Param:
    """One API argument, described as ``(label, value)`` pairs.

    bad:      the invalid values ``reject()`` drives this argument with
    sweep:    values replayed while a *different* argument is invalid
    accepted: the values ``expect()`` drives this argument with
    """

    def __init__(self, name, valid, bad, sweep=(), accepted=()):
        self.name = name
        self.bad = list(bad)
        self.sweep = list(sweep) or [valid]
        self.accepted = list(accepted) or [valid]


class Handle(Param):
    """A processor handle; every live device is driven in turn."""

    def __init__(self, name, handles, labels=None):
        labels = labels or [str(i) for i in range(len(handles))]
        values = list(zip(labels, handles))
        super().__init__(name, values[0], [("invalid", BAD_HANDLE)], sweep=values, accepted=values)


def opaque(name):
    """A non-processor handle (socket, event, node, BDF).

    The valid value is itself invalid, so this only suits an API whose sole
    argument is this handle; otherwise the other arguments are never the reason
    the call is refused.
    """
    return Param(name, ("invalid", BAD_HANDLE), [("invalid", BAD_HANDLE)])


def integer(name, valid=0, bounds=False):
    """An integer argument.

    ``bounds`` adds an out-of-range value. That value is type-correct, so it
    reaches the library instead of being refused by the binding: only set it on
    a getter, and never on an argument that sizes an allocation. ``reject_only``
    enforces this.
    """
    bad = [("bad-type", BAD_INT)]
    if bounds:
        bad.append(("out-of-range", OUT_OF_RANGE))
    return Param(name, (str(valid), valid), bad)


def text(name, valid=""):
    return Param(name, (repr(valid), valid), [("bad-type", BAD_STR)])


def enum(name, type_list):
    """An enum argument driven from one of the enum tables above.

    ``reject()`` replays every member, including the ones the table marks as
    rejected: the deliberately invalid argument is refused first, so no member
    reaches the device. ``expect()`` uses only the accepted members.
    """
    members = [(member, value) for member, value, _ in type_list]
    accepted = [(member, value) for member, value, cond in type_list if cond == PASS]
    return Param(
        name,
        (accepted or members)[0],
        [("bad-type", BAD_ENUM)],
        sweep=members,
        accepted=accepted or members,
    )


def _printable(value):
    """Replace ctypes handles with their address; json.dumps cannot encode them."""
    if isinstance(value, ctypes.c_void_p):
        return hex(value.value or 0)
    if isinstance(value, dict):
        return {key: _printable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_printable(item) for item in value]
    return value


def _inspect(value, path="result", problems=None, sentinels=None):
    """Collect structural defects and not-populated markers from an API payload."""
    problems = [] if problems is None else problems
    sentinels = [] if sentinels is None else sentinels

    if value is None:
        problems.append(f"{path} is None")
    elif isinstance(value, (bytes, bytearray)):
        problems.append(f"{path} is undecoded {type(value).__name__}")
    elif isinstance(value, ctypes.c_void_p):
        # An opaque handle is the payload for the discovery APIs; only a null
        # one is meaningless.
        if not value.value:
            sentinels.append(path)
    elif isinstance(value, (ctypes._SimpleCData, ctypes.Structure)):
        # _SimpleCData is a CPython internal, but there is no public base class
        # covering every scalar ctypes type.
        problems.append(f"{path} leaked a raw ctypes {type(value).__name__}")
    elif isinstance(value, dict):
        if not value:
            problems.append(f"{path} is an empty dict")
        for key, item in value.items():
            _inspect(item, f"{path}.{key}", problems, sentinels)
    elif isinstance(value, (list, tuple)):
        # An empty list is a real answer (no processes, no bad pages); callers
        # that require entries assert on the payload themselves.
        for i, item in enumerate(value):
            _inspect(item, f"{path}[{i}]", problems, sentinels)
    elif isinstance(value, bool):
        # A flag carries no sentinel meaning.
        pass
    elif isinstance(value, int) and value in _SENTINEL_INTS:
        sentinels.append(path)
    elif isinstance(value, str) and value in _SENTINEL_STRS:
        sentinels.append(path)

    return problems, sentinels


def _leaf(path):
    """Field name ending an _inspect path such as ``result[0].uuid``."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _field_names(value, prefix="result", names=None):
    """Leaf field names reachable in a payload."""
    names = set() if names is None else names
    if isinstance(value, dict):
        for key, item in value.items():
            names.add(key)
            _field_names(item, key, names)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _field_names(item, prefix, names)
    else:
        names.add(prefix)
    return names


def _unpopulated_required(data, sentinels, required):
    """Fields the caller declared must carry real data but do not."""
    if not required:
        return []
    if required is True:
        return [f"{path} is a sentinel, not data" for path in sentinels]
    # A named field that is absent means the requirement stopped applying, so
    # report it rather than letting a typo pass as coverage.
    present = _field_names(data)
    unfilled = {_leaf(path) for path in sentinels}
    problems = []
    for name in required:
        if name not in present:
            problems.append(f"required field {name!r} is not in the payload")
        elif name in unfilled:
            problems.append(f"{name} is a sentinel, not data")
    return problems


def _call_label(func_name, names, slots):
    args = ", ".join(f"{name}={label}" for name, (label, _) in zip(names, slots))
    return f"\t### {func_name}({args})"


def _unrejectable_positions(func_name):
    """Argument positions whose ctypes type coerces rather than refuses."""
    binding = getattr(amdsmi.amdsmi_wrapper, func_name, None)
    argtypes = getattr(binding, "argtypes", None) or []
    return {i for i, kind in enumerate(argtypes) if kind in _UNREJECTABLE_CTYPES}


def _arg_sets(params, bad_idx, bad):
    """Yield one slot list per call for a single invalid argument.

    Each remaining axis is swept on its own rather than as a full product: the
    invalid argument is refused by the first guard it reaches, so crossing the
    other axes with each other multiplies calls without adding coverage.
    """
    base = [param.sweep[0] for param in params]
    base[bad_idx] = bad

    swept = False
    for i, param in enumerate(params):
        if i == bad_idx or len(param.sweep) <= 1:
            continue
        swept = True
        for value in param.sweep:
            row = list(base)
            row[i] = value
            yield row
    if not swept:
        yield base


class ApiTest:
    """Drives one AMD SMI API, negatively or positively."""

    def __init__(self, common_obj):
        self.common = common_obj

    def _resolve(self, func_name):
        func = getattr(amdsmi, func_name, None)
        if func is None:
            raise unittest.SkipTest(f"Missing amdsmi API in amdsmi_interface.py: {func_name}")
        return func

    def reject(self, func_name, *params):
        """Drive *func_name* with one invalid argument at a time; all must fail."""
        func = self._resolve(func_name)
        unrejectable = _unrejectable_positions(func_name)
        names = [param.name for param in params]
        accepted = []
        for idx, param in enumerate(params):
            if idx in unrejectable:
                raise AssertionError(
                    f"{func_name}({param.name}) is a ctypes c_bool: every value coerces"
                    " to True, so driving it here would reach the device instead of"
                    " being refused. Cover this argument in the functional tier."
                )
            for bad in param.bad:
                for slots in _arg_sets(params, idx, bad):
                    label = _call_label(func_name, names, slots)
                    if self._call_bad(func, label, slots):
                        accepted.append(label.strip())
        self.common.print("")
        if accepted:
            raise AssertionError(
                f"{func_name} accepted invalid arguments:\n  " + "\n  ".join(accepted)
            )

    def expect(
        self,
        func_name,
        *params,
        validate=True,
        skip_when=None,
        require_populated=(),
        require_success=False,
    ):
        """Drive *func_name* with valid arguments only; all must succeed.

        Repeated for every device and every accepted enum value. Returns the
        payload of the last successful call so a caller can add its own checks.
        Pass ``validate=False`` for an API that returns nothing by design, and
        ``skip_when`` to drop argument combinations the API does not define.
        ``require_populated`` names fields that must hold real data instead of a
        sentinel; ``require_success`` turns "nothing succeeded" from a skip into
        a failure, for APIs every supported device is expected to answer.
        """
        func = self._resolve(func_name)
        names = [param.name for param in params]
        failures = []
        attempted = unsupported = succeeded = 0
        last = None
        for slots in itertools.product(*[param.accepted for param in params]):
            if skip_when and skip_when([label for label, _ in slots]):
                continue
            attempted += 1
            label = _call_label(func_name, names, slots)
            try:
                data = func(*[value for _, value in slots])
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(label, e, self.common.PASS):
                    failures.append(f"{label.strip()} -> {e}")
                else:
                    unsupported += 1
                continue
            except amdsmi.AmdSmiException as e:
                self.common.print(f"{label}\n\tTEST FAILURE, {type(e).__name__}: {e}")
                failures.append(f"{label.strip()} -> {type(e).__name__}: {e}")
                continue

            self.common.print(label, _printable(data))
            problems, sentinels = _inspect(data) if validate else ([], [])
            if sentinels:
                self.common.print(f"\tNOT POPULATED ({len(sentinels)}): {', '.join(sentinels[:8])}")
            problems += _unpopulated_required(data, sentinels, require_populated)
            if problems:
                for problem in problems:
                    self.common.print(f"\tTEST FAILURE, invalid payload: {problem}")
                failures.append(f"{label.strip()} -> {'; '.join(problems)}")
                continue
            self.common.check_ret("", "", self.common.PASS)
            succeeded += 1
            last = data

        self.common.print(
            f"\t{succeeded} call(s) returned AMDSMI_STATUS_SUCCESS,"
            f" {unsupported} not supported on this platform"
        )
        self.common.print("")
        if failures:
            raise AssertionError(f"{func_name} failed:\n  " + "\n  ".join(failures))
        # A run that asserted nothing must never report as a pass.
        if attempted == 0:
            message = f"{func_name}: every argument combination was excluded"
            if require_success:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        if succeeded == 0:
            message = f"{func_name} is not supported on this platform"
            if require_success:
                raise AssertionError(message)
            raise unittest.SkipTest(message)
        return last

    def _call_bad(self, func, label, slots):
        """Return True when the API wrongly accepted the invalid argument."""
        args = [value for _, value in slots]
        try:
            data = func(*args)
        except amdsmi.AmdSmiLibraryException as e:
            return self.common.check_ret(label, e, self.common.ANY_FAIL)
        except amdsmi.AmdSmiException as e:
            return self.common.check_ret(label, e, self.common.FAIL)
        except _BINDING_REJECTIONS as e:
            self.common.print(
                f"{label}\n\tTEST SUCCESS, AMDSMI API rejected the argument"
                f" ({type(e).__name__}), AMDSMI_STATUS_INVAL"
            )
            return False
        self.common.print(label, _printable(data))
        self.common.print(_UNEXPECTED_SUCCESS)
        return True


def _nic_handles():
    handles = []
    for socket in amdsmi.amdsmi_get_socket_handles():
        for nic_type in (
            amdsmi.AmdSmiProcessorType.AMD_AINIC,
            amdsmi.AmdSmiProcessorType.AMD_BRCM_NIC,
        ):
            try:
                found = amdsmi.amdsmi_get_processor_handles_by_type(socket, nic_type)
            except (amdsmi.AmdSmiLibraryException, AttributeError):
                continue
            handles.extend(found["processor_handles"])
    return handles


def _cpu_handles():
    try:
        return amdsmi.amdsmi_get_cpu_handles()["processor_handles"]
    except amdsmi.AmdSmiLibraryException:
        return []
    except AttributeError:
        # A library built without ESMI never binds the CPU entry points.
        return []


class ApiTestCase(unittest.TestCase):
    """Live-hardware fixture for the API suites.

    ``HANDLE_KIND`` selects which processor handles the suite drives -- "gpu",
    "cpu" or "nic". When the platform exposes none of that kind the rejection
    path still runs against a null handle, since every API must be offered an
    invalid argument somewhere; only the read path is skipped.
    """

    HANDLE_KIND = "gpu"

    @classmethod
    def setUpClass(cls):
        # One session per suite rather than per test: enumeration is the same
        # for every method and dominates the runtime of a rejection-only test.
        cls.common = Common(verbose, probe_devices=False)
        cls.common.amdsmi_smart_init()
        cls.common.processors = amdsmi.amdsmi_get_processor_handles()

        if cls.HANDLE_KIND == "gpu":
            handles = cls.common.processors
        elif cls.HANDLE_KIND == "cpu":
            handles = _cpu_handles()
        else:
            handles = _nic_handles()

        cls.has_device = bool(handles)
        if cls.has_device:
            cls.handle = Handle(cls.HANDLE_KIND, handles)
        else:
            # A null handle satisfies the interface's isinstance() guard, so the
            # argument under test is what gets refused.
            cls.handle = Handle(
                cls.HANDLE_KIND, [amdsmi.amdsmi_wrapper.amdsmi_processor_handle()], ["absent"]
            )

    @classmethod
    def tearDownClass(cls):
        cls._shut_down()

    @staticmethod
    def _shut_down():
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        # Gates only the UMA-carveout and TTM setters; every other setter is safe
        # here solely because reject_only() drives binding-refused values.
        os.environ["AMDSMI_DRY_RUN"] = "1"
        self.addCleanup(os.environ.pop, "AMDSMI_DRY_RUN", None)
        self.api = ApiTest(self.common)

    def _announce(self):
        # print_func_name() reads the caller's frame, which would name the
        # helper rather than the test once the call is one level down.
        if self.common.verbose == VERBOSITY_VERBOSE:
            print(f"\n## {self._testMethodName}()", flush=True)

    def both(self, func_name, *params, needs_peer=False, **expect_kwargs):
        """Reject every invalid argument, then require the valid ones to succeed."""
        self._announce()
        self.api.reject(func_name, *params)
        self._require_device(func_name)
        if needs_peer:
            self._require_peer(func_name)
        return self.api.expect(func_name, *params, **expect_kwargs)

    def reject_only(self, func_name, *params):
        """For an API that changes state: only the rejection path is safe here."""
        for param in params:
            if any(value is OUT_OF_RANGE for _, value in param.bad):
                raise AssertionError(
                    f"{func_name}({param.name}): an out-of-range value is type-correct, so"
                    " it reaches the device. A state-changing API must be driven only with"
                    " values the binding refuses -- drop bounds=True."
                )
        self._announce()
        self.api.reject(func_name, *params)

    def expect_only(self, func_name, *params, **expect_kwargs):
        """For an API with no invalid form, such as a zero-argument getter."""
        self._announce()
        self._require_device(func_name)
        return self.api.expect(func_name, *params, **expect_kwargs)

    def _require_device(self, func_name):
        """Stop before the read path when the platform has no device to read."""
        if not self.has_device:
            raise unittest.SkipTest(
                f"no {self.HANDLE_KIND} device: {func_name} rejection verified,"
                " read path not applicable"
            )

    def _require_peer(self, func_name):
        """A pair API needs two devices; on one device every pair is excluded."""
        count = len(self.common.processors)
        if count < 2:
            raise unittest.SkipTest(
                f"{func_name}: needs two devices to form a pair, found {count};"
                " rejection verified, read path not applicable"
            )

    def prerequisite(self, func_name, *args):
        """Fetch a value another API needs, skipping when it is unavailable here."""
        try:
            return getattr(amdsmi, func_name)(*args)
        except amdsmi.AmdSmiLibraryException as e:
            raise unittest.SkipTest(f"{func_name} unavailable: {e}") from e
