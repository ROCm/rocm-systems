import os
import re
import select
import sys
import tempfile
import unittest.mock
from subprocess import CalledProcessError, PIPE, Popen, TimeoutExpired, run
import time
import hashlib
import logging
import shutil
import signal

DEFAULT_TIMEOUT = 60

LOGGER_NAME = "rocm-debug-agent-test"


def logger():
    """Return the diagnostic logger used by every test helper in this module."""
    return logging.getLogger(LOGGER_NAME)


def log_program_output(out_str, err_str):
    """Log captured stdout/stderr under clearly-labeled sections.

    The two streams are emitted separately because `subprocess.PIPE` buffers
    them independently, so the relative ordering between a stdout line and a
    stderr line cannot be reliably reconstructed after the fact. Labeling
    each section avoids the ambiguity of an undifferentiated dump.
    """
    out_str = (out_str or "").rstrip()
    err_str = (err_str or "").rstrip()
    if not out_str and not err_str:
        return
    parts = []
    if out_str:
        parts.append(f"--- stdout ---\n{out_str}")
    if err_str:
        parts.append(f"--- stderr ---\n{err_str}")
    logger().info("\n".join(parts))


def run_and_communicate(
    test_name,
    args="0",
    debug_agent_options="",
    timeout=DEFAULT_TIMEOUT,
):
    # Prepare command
    program = "./rocm-debug-agent-test"
    cmd = [program]
    if isinstance(args, (list, tuple)):
        cmd += [str(a) for a in args]
    else:
        cmd.append(str(args))
    with unittest.mock.patch.dict(
        os.environ,
        (
            {"ROCM_DEBUG_AGENT_OPTIONS": debug_agent_options}
            if debug_agent_options
            else {}
        ),
    ):
        p = Popen(cmd, stdout=PIPE, stderr=PIPE)

        def handle_failure(reason):
            logger().info("Test %s FAIL: %s", test_name, reason)
            p.kill()
            try:
                # Kill first, then communicate to flush any remaining output
                output, err = p.communicate()
            except Exception:
                output, err = b"", b""
            log_program_output(output.decode("utf-8"), err.decode("utf-8"))
            return None, None, False

        try:
            output, err = p.communicate(timeout=timeout)
        except TimeoutExpired:
            return handle_failure("Timeout reached during communicate.")
        except Exception:
            return handle_failure("Unknown exception.")

        out_str = output.decode("utf-8")
        err_str = err.decode("utf-8")
        log_program_output(out_str, err_str)
        return out_str, err_str, True


def check_errors(check_list, out_str, err_str):
    _log = logger()
    all_strings_found = True
    for i, check_str in enumerate(check_list, start=1):
        if not (check_str.search(err_str)):
            all_strings_found = False
            _log.info("Pattern %d/%d NOT found: %s", i, len(check_list), check_str.pattern)

    return all_strings_found


def filter_warnings(err_str):
    """Filter out warnings wich are expected on some archs."""
    return "\n".join(
        [
            line
            for line in err_str.split("\n")
            if not (
                "Precise memory not supported for all the agents" in line
                or "architecture not supported" in line
                or "Warning: Resource leak detected" in line
                or "rocm-dbgapi: warning: Cannot locate the amdgpu.ids file." in line
            )
        ]
    )


# set up
if len(sys.argv) != 2:
    raise Exception(
        "ERROR: Please specify test binary location. For example: $python3.6 run_test.py ./build"
    )
else:
    test_binary_directory = sys.argv[1]
    agent_library_directory = os.path.abspath(test_binary_directory) + "/.."
    if not "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = agent_library_directory
    else:
        os.environ["LD_LIBRARY_PATH"] += ":" + agent_library_directory
    os.environ["HSA_TOOLS_LIB"] = "librocm-debug-agent.so.2"
    os.chdir(test_binary_directory)

    # Set up file-only logging for diagnostic output (stdout/stderr dumps).
    # The script may be installed under a read-only ROCm prefix, so prefer the
    # already-chdir'd test binary directory and fall back to the system temp
    # directory if neither is writable.
    log = logger()
    log.setLevel(logging.DEBUG)
    log_path = os.path.abspath("run-test.log")
    try:
        _fh = logging.FileHandler(log_path, mode="w")
    except OSError:
        log_path = os.path.join(tempfile.gettempdir(), "run-test.log")
        _fh = logging.FileHandler(log_path, mode="w")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_fh)
    log.propagate = False
    print(f"Diagnostic log: {log_path}")

    # pre test to check if librocm-debug-agent.so.2 can be found
    out_str, err_str, success = run_and_communicate(
        test_name="0: default", args="0"
    )

    if success and filter_warnings(err_str):
        print(err_str)
        if '"librocm-debug-agent.so.2" failed to load' in err_str:
            print(
                "ERROR: Cannot find librocm-debug-agent.so.2, please set its location with environment variable LD_LIBRARY_PATH"
            )
        sys.exit(1)


def test_baseline(*, name, args, **_):
    out_str, err_str, success = run_and_communicate(
        test_name=name,
        args=args,
    )

    if not success:
        return False

    # Only log but not throw for err_str, since debug build has print
    # out that could be ignored
    if filter_warnings(err_str):
        logger().info("output:\n%s", err_str)

    return True


def test_snapshot_code_object(*, name, args, **_):
    out_str, err_str, success = run_and_communicate(
        test_name=name,
        args=args,
    )
    if not success:
        return False

    found_error = False

    # If the debug agent did not capture the code object on load, it should
    # not be able to open it on exception, leading to the following warning:
    #
    #    rocm-debug-agent: warning: elf_getphdrnum failed for `memory://226967#offset=0x1651d4f0&size=3456'
    #    rocm-debug-agent: warning: could not open code_object_1
    if "could not open code_object" in err_str:
        found_error = True

    # If the code object was not properly loaded, we should not have any
    # disassembly in the output
    if (
        "Disassembly:\n" not in err_str
        and "Disassembly for function kernel_abort:\n" not in err_str
    ):
        found_error = True

    if found_error:
        logger().info("Snapshot code object test failed: code_object error or missing disassembly")

    return not found_error


def _file_checksum(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_symbol_with_readelf(path: str, symbol: str) -> bool:
    # Use readelf -s (symbol table). Return True if symbol name appears.
    try:
        res = run(["readelf", "-sW", path], stdout=PIPE, stderr=PIPE, check=True)
        out = res.stdout.decode(errors="ignore")
        return symbol in out
    except (CalledProcessError):
        return False


def test_save_code_objects(*, name, args, **_):
    if not shutil.which("readelf"):
        logger().info(
            "Tool readelf not found, skipping %s", name
        )
        unsupported_tests.add(test_save_code_objects)
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        run_and_communicate(
            test_name=name,
            args=args,
            debug_agent_options=f"--save-code-objects={tmpdir} --load-all-code-objects",
        )

        try:
            code_objects = os.listdir(tmpdir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Temporary directory not found: {tmpdir}"
            ) from e

        if len(code_objects) == 0:
            logger().info(
                "No code object found in %s", tmpdir
            )
            return False

        # Filter to files that contain the target symbol in their
        # symbol table.
        full_paths = [
            os.path.join(tmpdir, f)
            for f in code_objects
            if os.path.isfile(os.path.join(tmpdir, f))
        ]
        with_symbol = []
        symbol_to_find = "saved_test_kernel"
        for path in full_paths:
            if _has_symbol_with_readelf(path, symbol_to_find):
                with_symbol.append(path)

        _log = logger()
        if len(with_symbol) != 2:
            _log.info(
                "Expected exactly 2 code objects containing symbol"
                " '%s', found %d", symbol_to_find, len(with_symbol)
            )
            _log.info(
                "All saved files:\n\t%s", "\n\t".join(code_objects)
            )
            _log.info(
                "Files with symbol:\n\t%s",
                "\n\t".join(os.path.basename(p) for p in with_symbol)
            )
            return False

        # Compare contents via checksum.
        checksums = [(_file_checksum(p), p) for p in with_symbol]
        unique_sums = {cs for cs, _ in checksums}
        if len(unique_sums) != 1:
            _log.info(
                "The two code objects containing the symbol do not"
                " have identical contents"
            )
            for cs, pth in checksums:
                _log.info("%s -> %s", os.path.basename(pth), cs)
            return False

        return True


def test_output_redirection(*, name, args, patterns=(), **_):
    check_list = [re.compile(s) for s in patterns]

    with tempfile.TemporaryDirectory() as tmpdir:
        with unittest.mock.patch.dict(
            os.environ, {"ROCM_DEBUG_AGENT_OPTIONS": f"-o {tmpdir}/output_log.txt"}
        ):

            # Start process, ignore what function return since everything
            # is written to the file
            run_and_communicate(
                test_name=name,
                args=args,
                debug_agent_options=f"-o {tmpdir}/output_log.txt",
            )

            # Read the output log
            with open(f"{tmpdir}/output_log.txt", "r") as f:
                log_contents = f.read()

            all_output_string_found = True
            _log = logger()
            for check_str in check_list:
                if not check_str.search(log_contents):
                    all_output_string_found = False
                    _log.info('"%s" Not Found in output_log.txt.', check_str.pattern)

            if not all_output_string_found:
                _log.info("Full output log contents:\n%s", log_contents)

            return all_output_string_found


def test_sigquit(*, args, patterns=(), **_):
    check_list = [re.compile(s) for s in patterns]

    LOOP_TIMEOUT = DEFAULT_TIMEOUT  # seconds

    p = Popen(["./rocm-debug-agent-test", args], stdout=PIPE, stderr=PIPE)

    kernel_started = False
    wave_seen = False
    timeout_seen = False

    consumed_out = []
    consumed_err = []

    deadline = time.monotonic() + LOOP_TIMEOUT
    streams_to_read = [p.stdout, p.stderr]

    while time.monotonic() < deadline and streams_to_read:

        rlist, _, _ = select.select(streams_to_read, [], [], 1)
        if not rlist:
            continue

        for r in rlist:
            line = r.readline()

            if line == b"":
                # Reading "" means that we reached EOF on this stream.
                # Remove it from the streams of interest so every stream
                # can be fully flushed before we exit the loop.
                del streams_to_read[streams_to_read.index(r)]
                continue

            s = line.decode("utf-8")
            if r is p.stdout:
                consumed_out.append(s)
            else:
                consumed_err.append(s)

            if not kernel_started and "Kernel started" in s:
                kernel_started = True
                os.kill(p.pid, signal.SIGQUIT)
                # We give our program LOOP_TIMEOUT secs to start the kernel.
                # Once we know that the kernel is running, we give it
                # an extra LOOP_TIMEOUT seconds to process SIGQUIT.
                deadline = deadline + LOOP_TIMEOUT

            if kernel_started:
                if s.lstrip().startswith(
                    "Disassembly for function sigquit_kern(int*):"
                ):
                    wave_seen = True
                    break
            if "Timeout reached. Exiting." in s:
                timeout_seen = True
        if wave_seen or timeout_seen:
            break

    p.terminate()
    try:
        output, err = p.communicate(timeout=3)
    except TimeoutExpired:
        logger().info(
            "Timeout reached during final communicate."
        )
        output, err = b"", b""
    except Exception:
        logger().info(
            "Unexpected exception during final communicate."
        )
        output, err = b"", b""

    out_str = "".join(consumed_out) + output.decode("utf-8")
    err_str = "".join(consumed_err) + err.decode("utf-8")
    _log = logger()

    if not kernel_started:
        _log.info("Timeout waiting for 'Kernel started'. Terminating process.")
        log_program_output(out_str, err_str)
        return False

    if timeout_seen or not wave_seen:
        if timeout_seen:
            _log.info("Timeout reached. Exiting. Failing test.")
        else:
            _log.info(
                "Loop timed out without receiving expected message. "
                "Failing test."
            )
        log_program_output(out_str, err_str)
        return False

    all_output_string_found = True
    for check_str in check_list:
        if not (check_str.search(err_str)):
            all_output_string_found = False
            _log.info('"%s" Not Found in dump.', check_str.pattern)

    if not all_output_string_found:
        log_program_output(out_str, err_str)

    return all_output_string_found


def test_debug_info_comparison(*, name, args, patterns=(), **_):
    # `args` selects the build *with* debug info; the no-debug-info build is
    # exposed by the test binary as a separate fixed argument.
    NO_DEBUG_INFO_ARGS = "7"

    check_list = [re.compile(s) for s in patterns]

    out_str_debug, err_str_debug, success_debug = run_and_communicate(
        test_name=f"{name} (debug info)",
        args=args,
    )
    out_str_no_debug, err_str_no_debug, success_no_debug = run_and_communicate(
        test_name=f"{name} (no debug info)",
        args=NO_DEBUG_INFO_ARGS,
    )

    if not (success_no_debug and success_debug):
        return False

    # Every source-line pattern must be present in the debug-info disassembly,
    # and none of them should appear in the no-debug-info disassembly.
    # (check_list already contains compiled regex objects; no need to re-compile.)
    all_present_in_debug = all(p.search(err_str_debug) for p in check_list)
    none_present_in_no_debug = not any(p.search(err_str_no_debug) for p in check_list)
    return all_present_in_debug and none_present_in_no_debug


def test_eager_code_object_save(*, name, args, **_):
    with tempfile.TemporaryDirectory() as tmpdir:
        run_and_communicate(
            test_name=f"{name} (lazy)",
            args=args,
            debug_agent_options=f"--save-code-objects={tmpdir}",
        )

        if len(os.listdir(tmpdir)) != 0:
            logger().info(
                "Code object saved, while code object saving should have been lazy"
            )
            return False

        run_and_communicate(
            test_name=f"{name} (eager)",
            args=args,
            debug_agent_options=f"--save-code-objects={tmpdir} -c",
        )

        if len(os.listdir(tmpdir)) == 0:
            logger().info("Missing saved code objects")
            return False
    return True


def test_lazy_loading_and_eager_incompatible(*, name, args, patterns=(), **_):
    check_list = [re.compile(s) for s in patterns]

    # The two flags must reject each other in either order.
    for opts in ("--load-all-code-objects --lazy", "--lazy --load-all-code-objects"):
        out, err, success = run_and_communicate(
            test_name=f"{name} ({opts})",
            args=args,
            debug_agent_options=opts,
        )
        if not success or not check_errors(check_list, out, err):
            logger().info("Failed to detect -c and -z incompatibility")
            return False
    return True


# ==============================================================================
# Test Definitions
#
# Each entry fully describes a test.  For tests with custom logic, 'function'
# points to the existing handler above.  For simple pattern-check tests that
# don't exist yet, 'patterns' alone is sufficient (handled by run_test).
#
# Supported keys:
#   name           - unique test identifier (used in output)
#   description    - human-readable summary
#   args           - argument(s) passed to the test binary
#   options        - ROCM_DEBUG_AGENT_OPTIONS value (optional; default: empty)
#   timeout        - per-test timeout in seconds (optional; default: DEFAULT_TIMEOUT)
#   patterns       - list of regex strings to match against stderr
#   function       - custom handler invoked with every other entry in this
#                    test definition spread as keyword arguments. Handlers
#                    typically declare named keyword-only parameters (e.g.
#                    `def test_xxx(*, name, args, patterns=(), **_):`) and let
#                    `**_` swallow fields they do not consume. Omit this key
#                    for default pattern-matching behaviour.
#   abort_on_fail  - if True, abort the entire suite when this test fails
# ==============================================================================

TEST_DEFINITIONS = [
    {
        'name': 'test_baseline',
        'description': 'Fault-free run produces no diagnostic output',
        'args': '0',
        'function': test_baseline,
    },
    {
        'name': 'test_assert_trap',
        'description': 'Assert trap dumps faulting wave with registers and disassembly',
        'args': '1',
        'patterns': [
            r"HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception\.",
            r"\(stopped, reason: ASSERT_TRAP\)",
            r"exec: (00000000)?00000001",
            r"s0:",
            r"v0:",
            r"0x0000: 22222222 11111111",
            r"Disassembly for function vector_add_assert_trap\(int\*, int\*, int\*\)",
        ],
    },
    {
        'name': 'test_memory_violation',
        'description': 'Memory violation dumps faulting wave with registers and disassembly',
        'args': '2',
        'patterns': [
            r"\(stopped, reason: MEMORY_VIOLATION\)",
            r"exec: (ffffffff)?ffffffff",
            r"s0:",
            r"v0:",
            r"0x0000: 22222222 11111111",
            r"Disassembly for function vector_add_memory_fault\(int\*, int\*, int\*\)",
        ],
    },
    {
        'name': 'test_snapshot_code_object',
        'description': 'Code object captured at load time is still readable on exception',
        'args': '3',
        'function': test_snapshot_code_object,
    },
    {
        'name': 'test_save_code_objects',
        'description': '--save-code-objects writes complete, identical copies to disk',
        'args': '4',
        'function': test_save_code_objects,
    },
    {
        'name': 'test_output_redirection',
        'description': '-o redirects the wave dump to a file instead of stderr',
        'args': '1',
        'function': test_output_redirection,
        'patterns': [
            r"s0:",
            r"v0:",
            r"0x0000: 22222222 11111111",
            r"Disassembly for function vector_add_assert_trap\(int\*, int\*, int\*\)",
        ],
    },
    {
        'name': 'test_help',
        'description': '-h prints the usage banner and exits',
        'args': '0',
        'options': '-h',
        'timeout': 10,
        'patterns': [
            r"ROCdebug-agent usage",
        ],
    },
    {
        'name': 'test_log_level',
        'description': '-l info enables rocm-dbgapi diagnostic logging',
        'args': '1',
        'options': '-l info',
        'patterns': [
            r"rocm-dbgapi",
        ],
    },
    {
        'name': 'test_all_waves',
        'description': '--all dumps every active wave, not only the faulting one',
        'args': '5',
        'options': '--all',
        'patterns': [
            r"wave_1",
            r"wave_2",
            r"wave_3",
            r"wave_4",
            r"wave_5",
            r"wave_6",
            r"wave_7",
            r"wave_8",
        ],
    },
    {
        'name': 'test_sigquit',
        'description': 'SIGQUIT during a running kernel triggers a wave dump',
        'args': '6',
        'function': test_sigquit,
        'patterns': [
            r"s0:",
            r"v0:",
            r"Disassembly for function sigquit_kern\(int\*\)",
        ],
    },
    {
        'name': 'test_debug_info',
        'description': 'Disassembly shows source lines only when debug info is present',
        'args': '1',
        'function': test_debug_info_comparison,
        'patterns': [
            r"c\[gid\] = a\[gid\] \+ b\[gid\] \+ \(lds_check\[0\] >> 32\);",
            r"if \(gid == 0\)",
        ],
    },
    {
        'name': 'test_eager_vs_lazy',
        'description': '-c saves code objects eagerly; default lazy mode saves none without an event',
        'args': '4',
        'function': test_eager_code_object_save,
    },
    {
        'name': 'test_lazy_loading_and_eager_incompatible',
        'description': '--load-all-code-objects and --lazy are rejected together (either order)',
        'args': '4',
        'function': test_lazy_loading_and_eager_incompatible,
        'patterns': [
            r'"--load-all-code-objects" and "--lazy" are mutually exclusive',
        ],
    },
]


def log_test_header(label, width=72):
    """Write a visually distinct banner for a test run to the diagnostic log."""
    _log = logger()
    _log.info("")
    _log.info("=" * width)
    _log.info("  %s", label)
    _log.info("=" * width)


def run_test(test_def):
    """Dispatch a single test definition.

    If 'function' is present, call it with every test_def field (except
    'function' itself) spread as a keyword argument, so handlers can declare
    in their signature exactly which fields they consume.  Otherwise fall
    back to generic pattern matching against stderr.
    """
    if 'function' in test_def:
        handler_kwargs = {k: v for k, v in test_def.items() if k != 'function'}
        return test_def['function'](**handler_kwargs)

    # Generic path: run binary with args/options, check stderr patterns.
    check_list = [re.compile(s) for s in test_def.get('patterns', [])]
    out_str, err_str, success = run_and_communicate(
        test_name=test_def['name'],
        args=test_def['args'],
        debug_agent_options=test_def.get('options', ''),
        timeout=test_def.get('timeout', DEFAULT_TIMEOUT),
    )
    if not success:
        return False
    if not check_list:
        return True
    return check_errors(check_list, out_str, err_str)


unsupported_tests = set()

test_success = True
failed_tests = []
total_pass = 0
total_fail = 0
total_unsupported = 0

# TODO: if more environment variables need sweeping in the future, replace this
# hard-coded loop with a table of (env_var, possible_values) entries and run the
# Cartesian product of their values, so adding a new variable is a one-line change.
for deferred_loading in (None, "1", "0"):
    with unittest.mock.patch.dict("os.environ"):
        if deferred_loading is None:
            mode_label = "HIP_ENABLE_DEFERRED_LOADING unset"
            if "HIP_ENABLE_DEFERRED_LOADING" in os.environ:
                del os.environ["HIP_ENABLE_DEFERRED_LOADING"]
        else:
            mode_label = f"HIP_ENABLE_DEFERRED_LOADING={deferred_loading}"
            os.environ["HIP_ENABLE_DEFERRED_LOADING"] = deferred_loading

        abort = False
        _log = logger()
        for test_def in TEST_DEFINITIONS:
            name = test_def['name']
            desc = test_def['description']
            label = f"{name}: {desc} [{mode_label}]"

            log_test_header(label)
            result = run_test(test_def)

            func = test_def.get('function')
            if func and func in unsupported_tests:
                verdict = f"UNSUPPORTED: {label}"
                print(verdict)
                _log.info(verdict)
                total_unsupported += 1
            elif result:
                verdict = f"PASS: {label}"
                print(verdict)
                _log.info(verdict)
                total_pass += 1
            else:
                verdict = f"FAIL: {label}"
                print(verdict)
                _log.info(verdict)
                total_fail += 1
                failed_tests.append(verdict)
                test_success = False

                if test_def.get('abort_on_fail', False):
                    msg = (
                        f"\n*** abort_on_fail set for {name} — "
                        "aborting remaining tests ***\n"
                    )
                    print(msg)
                    _log.info(msg)
                    abort = True
                    break

        if abort:
            break

total = total_pass + total_fail + total_unsupported
print()
print("=" * 60)
print()
print(f"Total:       {total}")
print(f"PASS:        {total_pass}")
print(f"FAIL:        {total_fail}")
print(f"UNSUPPORTED: {total_unsupported}")

if failed_tests:
    print()
    print("Failed:")
    for entry in failed_tests:
        print(f"  - {entry}")

print()
print("=" * 60)

if test_success:
    print("\nOVERALL: PASS")
else:
    print("\nOVERALL: FAIL")
    sys.exit(1)
