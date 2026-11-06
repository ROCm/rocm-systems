import argparse
import os
import re
import signal
import subprocess
import sys


def run(options):
    environment = dict(os.environ)
    existing = environment.get("LD_PRELOAD", "")
    environment["LD_PRELOAD"] = ":".join(
        item for item in (existing, options.client) if item
    )
    environment["ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS"] = str(
        int(options.mode == "signal-less")
    )
    environment["ROCPROFILER_QUEUE_INTERPOSITION"] = str(int(options.mode != "expanded"))
    environment["ROCPROFILER_LOG_LEVEL"] = "info"
    process = subprocess.Popen(
        [options.app, options.order, options.test_case],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        try:
            output, _ = process.communicate(timeout=5)
            print(output)
        except subprocess.TimeoutExpired:
            print("[flush-contract] FAIL unreapable child; stop GPU testing")
            return 1
        print("[flush-contract] FAIL process timeout")
        return 1
    print(output)
    if process.returncode == 77:
        return 77
    if process.returncode != 0 or "[flush-contract] PASS" not in output:
        print("[flush-contract] FAIL native contract check")
        return 1
    # A mode that silently fell back to another completion path would still deliver its records
    # and look like a pass, so the intended path is confirmed from the SDK's own log before any
    # count is trusted. Absence of the decision line means the log format moved and this check is
    # no longer checking anything -- that is a failure, not a skip.
    interposition = re.findall(
        r"\[queue-interposition\] ROCPROFILER_QUEUE_INTERPOSITION=(true|false)", output
    )
    if not interposition:
        print("[flush-contract] FAIL no queue-interposition decision in log")
        return 1
    wanted = "false" if options.mode == "expanded" else "true"
    if interposition[-1] != wanted:
        print(
            "[flush-contract] FAIL wrong completion path:"
            f" wanted interposition={wanted}, got {interposition[-1]}"
        )
        return 1

    if options.mode == "signal-less":
        summaries = re.findall(
            r"signal-less summary: registered=(\d+) eop-proven=(\d+).*?emitted=(\d+) "
            r"no-timing=(\d+) register-refused=(\d+) cap-evicted=(\d+); "
            r"teardown finalized (\d+) deferred and stranded (\d+)",
            output,
        )
        if not summaries:
            # Distinguish "this device/driver never armed signal-less" from "the summary line
            # changed shape". The former is a capability skip; the latter would silently retire
            # every signal-less case to a green skip, which is the one outcome this matrix must
            # never produce.
            if "signal-less summary" in output:
                print("[flush-contract] FAIL unparsable signal-less summary")
                return 1
            print("[flush-contract] SKIP no signal-less summary emitted")
            return 77
        if all(int(row[0]) == 0 for row in summaries):
            print("[flush-contract] SKIP no confirmed signal-less registrations")
            return 77
        expected = (
            252
            if options.test_case == "repeat"
            else 85 if options.test_case == "unfinished" else 84
        )
        if len(summaries) != 1 or tuple(map(int, summaries[0])) != (
            expected,
            expected,
            expected,
            0,
            0,
            0,
            0,
            0,
        ):
            print("[flush-contract] FAIL incomplete signal-less accounting", summaries)
            return 1
    print("[flush-contract] VERIFIED", options.mode, options.order, options.test_case)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--client", required=True)
    parser.add_argument(
        "--mode", choices=("signal", "expanded", "signal-less"), required=True
    )
    parser.add_argument("--order", choices=("stop-first", "flush-first"), required=True)
    parser.add_argument(
        "--case",
        dest="test_case",
        choices=("tail", "delayed", "unfinished", "reentry", "repeat", "concurrent"),
        required=True,
    )
    sys.exit(run(parser.parse_args()))
