import os
import re
import select
import sys
import tempfile
import unittest.mock
from subprocess import Popen, PIPE, TimeoutExpired
import time
import hashlib
import shutil
from subprocess import Popen, PIPE, CalledProcessError, run
import signal

DEFAULT_TIMEOUT = 30


def run_and_communicate(
    test_name,
    args="0",
    debug_agent_options="",
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
            print(f"Test {test_name} FAIL: {reason}")
            p.kill()
            try:
                # Kill first, then communicate to flush any remaining output
                output, err = p.communicate()
            except Exception:
                output, err = b"", b""
            print("rocm-debug-agent test print out.")
            print(output.decode("utf-8"))
            print("rocm-debug-agent test error message.")
            print(err.decode("utf-8"))
            return None, None, False

        try:
            output, err = p.communicate(timeout=DEFAULT_TIMEOUT)
        except TimeoutExpired:
            return handle_failure("Timeout reached during communicate.")
        except Exception:
            return handle_failure("Unknown exception.")

        out_str = output.decode("utf-8")
        err_str = err.decode("utf-8")
        return out_str, err_str, True


def check_errors(check_list, out_str, err_str):
    all_strings_found = True
    for check_str in check_list:
        if not (check_str.search(err_str)):
            all_strings_found = False
            print('"', check_str, '" Not Found in dump.')

    if not all_strings_found:
        print("rocm-debug-agent test print out.")
        print(out_str)
        print("rocm-debug-agent test error message.")
        print(err_str)

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
    print("Test binary directory: ", os.path.abspath(test_binary_directory))
    agent_library_directory = os.path.abspath(test_binary_directory) + "/.."
    if not "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = agent_library_directory
    else:
        os.environ["LD_LIBRARY_PATH"] += ":" + agent_library_directory
    os.environ["HSA_TOOLS_LIB"] = "librocm-debug-agent.so.2"
    os.chdir(test_binary_directory)
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


# test 0
def check_test_0():
    print("Starting rocm-debug-agent-test 0")

    out_str, err_str, success = run_and_communicate(
        test_name="0: default", args="0"
    )

    if not success:
        return False

    # Only print but not throw for err_str, since debug build has print
    # out could be ignored
    if filter_warnings(err_str):
        print(err_str)

    return True


# test 1
def check_test_1():
    print("Starting rocm-debug-agent test 1")

    check_list = [
        re.compile(s)
        for s in [
            "HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception\\.",
            "\\(stopped, reason: ASSERT_TRAP\\)",
            "exec: (00000000)?00000001",
            #                  'status: 00012061',
            #                  'trapsts: 20000000',
            #                  'm0: 00000000',
            "s0:",
            "v0:",
            "0x0000: 22222222 11111111",  # First uint64_t in LDS is '1111111122222222'
            "Disassembly for function vector_add_assert_trap\\(int\\*, int\\*, int\\*\\)",
            #                  'vector_add_assert_trap.cpp:', # Debug info may not be available on some older distributions
            #                  '53          __builtin_trap ();', # Source files not always available (When install tests from package)
            #                  's_trap 2'
        ]
    ]

    out_str, err_str, success = run_and_communicate(
        test_name="1: debug_trap", args="1"
    )

    if success:
        return check_errors(check_list, out_str, err_str)
    else:
        return False


# test 2
def check_test_2():
    print("Starting rocm-debug-agent test 2")

    check_list = [
        re.compile(s)
        for s in [
            #                  'System event \(HSA_AMD_GPU_MEMORY_FAULT_EVENT\)',
            #                  'Faulting page: 0x',
            "\\(stopped, reason: MEMORY_VIOLATION\\)",
            "exec: (ffffffff)?ffffffff",
            #                  'status: 00012461',
            #                  'trapsts: 30000100',
            #                  'm0: 00001008',
            "s0:",
            "v0:",
            "0x0000: 22222222 11111111",  # First uint64_t in LDS is '1111111122222222'
            "Disassembly for function vector_add_memory_fault\\(int\\*, int\\*, int\\*\\)",
            #                  'vector_add_memory_fault.cpp:', Debug info may not be available on some older distributions
            #                  'global_store_dword' # Without precise memory, we can't guarantee that
        ]
    ]

    out_str, err_str, success = run_and_communicate(
        test_name="2: memory violation", args="2"
    )
    if success:
        return check_errors(check_list, out_str, err_str)
    else:
        return False


# test 3: snapshot code object on load
def check_test_3():
    print("Starting rocm-debug-agent test 3")

    out_str, err_str, success = run_and_communicate(
        test_name="3: snapshot code object on load", args="3"
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
        print("rocm-debug-agent test print out.")
        print(out_str)
        print("rocm-debug-agent test error message.")
        print(err_str)

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


# test 4: save code object on disk
def check_test_4():
    if not shutil.which("readelf"):
        print(
            "Tool readelf not found, could not run test 4, but don't know if"
            " that is an error."
        )
        unsupported_tests.append(check_test_4)
        return True

    print("Starting rocm-debug-agent test 4")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_and_communicate(
            test_name="4: save code objects",
            args="4",
            debug_agent_options=f"-p --save-code-objects={tmpdir}",
        )

        try:
            code_objects = os.listdir(tmpdir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Temporary directory not found: {tmpdir}"
            ) from e

        if len(code_objects) == 0:
            print(f"No code object found in {tmpdir}")
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

        if len(with_symbol) != 2:
            print(
                f"Expected exactly 2 code objects containing symbol"
                f" '{symbol_to_find}', found {len(with_symbol)}"
            )
            print(
                "All saved files:\n\t{}".format("\n\t".join(code_objects))
            )
            print(
                "Files with symbol:\n\t{}".format(
                    "\n\t".join(os.path.basename(p) for p in with_symbol)
                )
            )
            return False

        # Compare contents via checksum.
        checksums = [(_file_checksum(p), p) for p in with_symbol]
        unique_sums = {cs for cs, _ in checksums}
        if len(unique_sums) != 1:
            print(
                "The two code objects containing the symbol do not"
                " have identical contents"
            )
            for cs, pth in checksums:
                print(f"{os.path.basename(pth)} -> {cs}")
            return False

        return True


# test 5
def check_test_5():
    print("Starting rocm-debug-agent test 5")

    check_list = [
        re.compile(s)
        for s in [
            "s0:",
            "v0:",
            "0x0000: 22222222 11111111",  # First uint64_t in LDS is '1111111122222222'
            "Disassembly for function vector_add_assert_trap\\(int\\*, int\\*, int\\*\\)",
        ]
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        with unittest.mock.patch.dict(
            os.environ, {"ROCM_DEBUG_AGENT_OPTIONS": f"-o {tmpdir}/output_log.txt"}
        ):

            # Start process, ignore what function return since everything
            # is written to the file
            run_and_communicate(
                "5: -o option",
                args="1",
                debug_agent_options=f"-o {tmpdir}/output_log.txt",
            )

            # Read the output log
            with open(f"{tmpdir}/output_log.txt", "r") as f:
                log_contents = f.read()

            all_output_string_found = True
            for check_str in check_list:
                if not check_str.search(log_contents):
                    all_output_string_found = False
                    print(f'"{check_str}" Not Found in output_log.txt.')

            if not all_output_string_found:
                print("Full output log contents:")
                print(log_contents)

            return all_output_string_found


# test 6
def check_test_6():
    print("Starting rocm-debug-agent test 6")

    check_list = [
        re.compile(s)
        for s in [
            "ROCdebug-agent usage",
        ]
    ]

    out_str, err_str, success = run_and_communicate(
        test_name="6: help", args="0", debug_agent_options="-h"
    )

    if success:
        return check_errors(check_list, out_str, err_str)
    else:
        return False


# test 7
def check_test_7():
    print("Starting rocm-debug-agent test 7")

    check_list = [
        re.compile(s)
        for s in [
            "rocm-dbgapi",
        ]
    ]

    out_str, err_str, success = run_and_communicate(
        test_name="7: -l option", args="1", debug_agent_options="-l info"
    )

    if success:
        return check_errors(check_list, out_str, err_str)
    else:
        return False


# test 8
def check_test_8():
    print("Starting rocm-debug-agent test 8")

    check_list = [
        re.compile(s)
        for s in [
            "wave_1",
            "wave_2",
            "wave_3",
            "wave_4",
            "wave_5",
            "wave_6",
            "wave_7",
            "wave_8",
        ]
    ]

    out_str, err_str, success = run_and_communicate(
        test_name="8: --all option", args="5", debug_agent_options="--all"
    )

    if success:
        return check_errors(check_list, out_str, err_str)
    else:
        return False


# test 9 SIGQUIT
def check_test_9():
    print("Starting rocm-debug-agent test 9")

    check_list = [
        re.compile(s)
        for s in [
            "s0:",
            "v0:",
            "Disassembly for function sigquit_kern\\(int\\*\\)",
        ]
    ]

    LOOP_TIMEOUT = DEFAULT_TIMEOUT  # seconds

    p = Popen(["./rocm-debug-agent-test", "6"], stdout=PIPE, stderr=PIPE)

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
                # We give our program 30 secs to start the kernel.
                # Once we know that the kernel is running, we give it
                # extra 30 seconds to process SIGQUIT.
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
        print("Timeout reached during final communicate.")
        output, err = b"", b""
    except Exception:
        print("Unexpected exception during final communicate.")
        output, err = b"", b""

    out_str = "".join(consumed_out) + output.decode("utf-8")
    err_str = "".join(consumed_err) + err.decode("utf-8")

    if not kernel_started:
        print("Timeout waiting for 'Kernel started'. Terminating process.")
        print("rocm-debug-agent test print out.")
        print(out_str)
        print("rocm-debug-agent test error message.")
        print(err_str)
        return False

    if timeout_seen or not wave_seen:
        if timeout_seen:
            print("Timeout reached. Exiting. Failing test.")
        else:
            print(
                (
                    "Loop timed out without receiving expected message. "
                    "Failing test."
                )
            )
        print("rocm-debug-agent test print out.")
        print(out_str)
        print("rocm-debug-agent test error message.")
        print(err_str)
        return False

    all_output_string_found = True
    for check_str in check_list:
        if not (check_str.search(err_str)):
            all_output_string_found = False
            print('"', check_str, '" Not Found in dump.')

    if not all_output_string_found:
        print("rocm-debug-agent test print out.")
        print(out_str)
        print("rocm-debug-agent test error message.")
        print(err_str)

    return all_output_string_found


# test 10 check for difference in outputs for same program compiled with and
# without -ggdb flag
def check_test_10():
    print("Starting rocm-debug-agent test 10")

    check_list = [
        re.compile(re.escape(s))
        for s in ["c[gid] = a[gid] + b[gid] + (lds_check[0] >> 32);", "if (gid == 0)"]
    ]

    out_str_debug, err_str_debug, success_debug = run_and_communicate(
        test_name="10: debug info", args="1"
    )
    out_str_no_debug, err_str_no_debug, success_no_debug = run_and_communicate(
        test_name="10: no debug info", args="7"
    )

    if not (success_no_debug and success_debug):
        return False

    # check if string is in dissasembly of code with debug info but not in other one
    found_in_debug = False
    found_in_no_debug = False
    for check_str in check_list:
        pattern = re.compile(check_str)
        if pattern.search(err_str_debug):
            found_in_debug = True
        if pattern.search(err_str_no_debug):
            found_in_no_debug = True

    return found_in_debug and not found_in_no_debug


test_success = True
unsupported_tests = []

for deferred_loading in (None, "1", "0"):
    with unittest.mock.patch.dict("os.environ"):
        if deferred_loading is None:
            print(f"### Testing without HIP_ENABLE_DEFERRED_LOADING")
            if "HIP_ENABLE_DEFERRED_LOADING" in os.environ:
                del os.environ["HIP_ENABLE_DEFERRED_LOADING"]
        else:
            print(f"### Testing with HIP_ENABLE_DEFERRED_LOADING={deferred_loading}")
            os.environ["HIP_ENABLE_DEFERRED_LOADING"] = deferred_loading

        test_list = [
            check_test_0,
            check_test_1,
            check_test_2,
            check_test_3,
            check_test_4,
            check_test_5,
            check_test_6,
            check_test_7,
            check_test_8,
            check_test_9,
            check_test_10,
        ]

        for i, test in enumerate(test_list, start=0):
            result = test()
            test_success &= result
            if test in unsupported_tests:
                print(f"Test {i} UNSUPPORTED")
            elif result:
                print(f"Test {i} PASS")
            else:
                print(f"Test {i} FAIL")

if test_success:
    print("rocm-debug-agent test Pass!")
else:
    raise Exception("rocm-debug-agent test fail!")
