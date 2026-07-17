# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Automated coverage for the command-line help surface of the rocprof-sys
launchers (``rocprof-sys-run`` and ``rocprof-sys-sample``).

Validates every help / informational invocation: the help entry points
(``--version`` / ``--help`` / ``--help=all`` / compact help), every
``--help=<topic>`` / ``--help=<domain>`` page, the preset listing and the
``--explain=<preset>`` pages.

Each case is a ``pytest.param`` of ``(run_args, pass_regex)`` consumed by the
single parametrized test below, which runs it against every target in
``TARGETS``. Patterns use the ``{prog}`` placeholder wherever the program name
is embedded in the output; it is substituted with the target at runtime. Add or
adjust coverage by editing the ``HELP_CASES`` list.
"""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.cli_help]

# Both launchers share the same help/flag surface; the embedded mark selects the
# matching runner category so a single test body covers run + sample.
TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


# ----------------------------------------------------------------------------
# Shared pattern groups (kept here to avoid repeating them across cases)
# ----------------------------------------------------------------------------

# Sections printed by the compact help screens (--help / -h / -?).
_COMPACT = [
    r"Usage: {prog}",
    r"QUICK START",
    r"DOMAIN FLAGS",
    r"COMMON OPTIONS",
    r"HELP TOPICS",
    r"EXAMPLES",
]

# Common scaffolding printed by every --explain=<preset> page.
_EXPLAIN = [
    r"Description:",
    r"Use case:",
    r"Environment Variables:",
    r"ROCPROFSYS_\w+ = ",
]


def _explain(preset: str, category: str) -> list[str]:
    """pass_regex for an ``--explain=<preset>`` page (preset-specific + shared)."""
    return [
        rf"Preset: {preset}",
        rf"Category:\s+{category}",
        rf"Usage: {{prog}} --preset={preset}",
    ] + _EXPLAIN


# ----------------------------------------------------------------------------
# Expected-output cases: (run_args, pass_regex)
# ----------------------------------------------------------------------------

HELP_CASES = [
    # --- Domain help pages (--help=<domain>) ------------------------------
    pytest.param(
        ["--help=cpu"],
        [
            r"CPU OPTIONS \(",
            r"-H, --host",
            r"-f, --sampling-freq",
            r"-t, --tids",
            r"--sampling-wait",
            r"--sampling-duration",
            r"--sample-cputime",
            r"--sample-realtime",
            r"--sample-overflow",
            r"-C, --cpu-events",
            r"-G, --gpu-events",
            r"--cpu",
            r"See also",
        ],
        id="domain_cpu",
    ),
    pytest.param(
        ["--help=gpu"],
        [
            r"GPU OPTIONS \(",
            r"-D, --device",
            r"--use-amd-smi",
            r"--process-freq",
            r"--process-wait",
            r"--process-duration",
            r"--gpus",
            r"--ai-nics",
            r"-G, --gpu-events",
            r"--gpu",
            r"See also",
        ],
        id="domain_gpu",
    ),
    pytest.param(
        ["--help=parallel"],
        [
            r"PARALLEL OPTIONS \(",
            r"-I, --include",
            r"-E, --exclude",
            r"--parallel",
            r"See also",
        ],
        id="domain_parallel",
    ),
    pytest.param(
        ["--help=rocm"],
        [
            r"ROCM OPTIONS \(",
            r"-T, --trace",
            r"--use-amd-smi",
            r"--selected-regions",
            r"--gpus",
            r"--ai-nics",
            r"--hsa-interrupt",
            r"--rocm",
            r"kfd_events",
            r"See also",
        ],
        id="domain_rocm",
    ),
    # --- Preset explanations (--explain=<preset>) -------------------------
    pytest.param(
        ["--explain=balanced"], _explain("balanced", "general"), id="explain_balanced"
    ),
    pytest.param(
        ["--explain=detailed"], _explain("detailed", "general"), id="explain_detailed"
    ),
    pytest.param(
        ["--explain=profile-mpi"],
        _explain("profile-mpi", "hpc"),
        id="explain_profile-mpi",
    ),
    pytest.param(
        ["--explain=profile-only"],
        _explain("profile-only", "general"),
        id="explain_profile-only",
    ),
    pytest.param(
        ["--explain=runtime-trace"],
        _explain("runtime-trace", "tracing"),
        id="explain_runtime-trace",
    ),
    pytest.param(
        ["--explain=sys-trace"], _explain("sys-trace", "tracing"), id="explain_sys-trace"
    ),
    pytest.param(
        ["--explain=trace-gpu"], _explain("trace-gpu", "gpu"), id="explain_trace-gpu"
    ),
    pytest.param(
        ["--explain=trace-hpc"], _explain("trace-hpc", "hpc"), id="explain_trace-hpc"
    ),
    pytest.param(
        ["--explain=trace-hw-counters"],
        _explain("trace-hw-counters", "gpu"),
        id="explain_trace-hw-counters",
    ),
    pytest.param(
        ["--explain=trace-openmp"],
        _explain("trace-openmp", "hpc"),
        id="explain_trace-openmp",
    ),
    pytest.param(
        ["--explain=workload-trace"],
        _explain("workload-trace", "gpu"),
        id="explain_workload-trace",
    ),
    # --- Full option dump (--help=all): assert the stable section headers.
    # Individual flags are intentionally not asserted here: many are
    # conditionally compiled (AMD SMI, CI/debug build options, etc.), so the
    # bracketed section headers are the portable, build-independent signal.
    pytest.param(
        ["--help=all"],
        [
            r"Options:",
            r"-h, -\?, --help(?![-\w])",
            r"--version(?![-\w])",
            r"\[DEBUG OPTIONS\]",
            r"\[MODE OPTIONS\]",
            r"\[GENERAL OPTIONS\]",
            r"\[BACKEND OPTIONS\]",
            r"\[PARALLELISM OPTIONS\]",
            r"\[TRACING OPTIONS\]",
            r"\[PROFILE OPTIONS\]",
            r"\[HOST/DEVICE \(PROCESS SAMPLING\) OPTIONS\]",
            r"\[GENERAL SAMPLING OPTIONS\]",
            r"\[SAMPLING TIMER OPTIONS\]",
            r"\[ADVANCED SAMPLING OPTIONS\]",
            r"\[HARDWARE COUNTER OPTIONS\]",
            r"\[CATEGORY OPTIONS\]",
            r"\[IO OPTIONS\]",
            r"\[PERFETTO OPTIONS\]",
            r"\[TIMEMORY OPTIONS\]",
            r"\[ROCM OPTIONS\]",
            r"\[MISCELLANEOUS OPTIONS\]",
            r"\[LIBROCPROF-SYS OPTIONS\]",
            r"\[PRESET OPTIONS\]",
            r"\[DOMAIN OPTIONS\]",
            r"\[EXPORT OPTIONS\]",
        ],
        id="all",
    ),
    # --- Compact help screens (--help / -h / -?) --------------------------
    pytest.param(["--help"], _COMPACT, id="compact_long"),
    pytest.param(["-h"], _COMPACT, id="compact_short_h"),
    pytest.param(["-?"], _COMPACT, id="compact_short_q"),
    # --- Preset listing (--list-presets) ----------------------------------
    pytest.param(
        ["--list-presets"],
        [
            r"Available Presets:",
            r"Usage: {prog} --preset=<name>",
            r"{prog} --explain=<name>",
            r"general:",
            r"gpu:",
            r"hpc:",
            r"tracing:",
            r"\bbalanced\b",
            r"\bdetailed\b",
            r"\bprofile-only\b",
            r"\btrace-gpu\b",
            r"\btrace-hw-counters\b",
            r"\bworkload-trace\b",
            r"\bprofile-mpi\b",
            r"\btrace-hpc\b",
            r"\btrace-openmp\b",
            r"\bruntime-trace\b",
            r"\bsys-trace\b",
        ],
        id="list_presets",
    ),
    # --- Topic help pages (--help=<topic>) --------------------------------
    pytest.param(
        ["--help=backend"],
        [
            r"{prog} --help=backend",
            r"-I, --include",
            r"-E, --exclude",
            r"--use-amd-smi",
            r"--use-causal",
            r"--use-code-coverage",
            r"--use-kokkosp",
            r"--use-mpip",
            r"--use-rcclp",
            r"--use-rocpd",
            r"--use-sampling",
            r"--use-shmem",
            r"--use-ucx",
            r"--trace-thread-barriers",
            r"--trace-thread-join",
            r"--trace-thread-locks",
            r"--use-process-sampling",
            r"--trace-thread-rw-locks",
            r"--trace-thread-spin-locks",
            r"--use-unified-memory-profiling",
            r"See also",
        ],
        id="topic_backend",
    ),
    pytest.param(
        ["--help=counters"],
        [
            r"{prog} --help=counters",
            r"-C, --cpu-events",
            r"-G, --gpu-events",
            r"See also",
        ],
        id="topic_counters",
    ),
    pytest.param(
        ["--help=debug"],
        [
            r"{prog} --help=debug",
            r"--log-level",
            r"--monochrome",
            r"--debug",
            r"-v, --verbose",
            r"--log-file",
            r"--dl-verbose",
            r"--perfetto-annotations",
            r"--kokkosp-kernel-logger",
            r"--sampling-allocator-size",
            r"--kokkosp-name-length-max",
            r"--kokkosp-prefix",
            r"--ci",
        ],
        id="topic_debug",
    ),
    pytest.param(
        ["--help=general"],
        [
            r"{prog} --help=general",
            r"-c, --config",
            r"-o, --output",
            r"-T, --trace",
            r"-L, --trace-legacy",
            r"-P, --profile",
            r"-F, --flat-profile",
            r"-H, --host",
            r"-D, --device",
            r"-w, --wait",
            r"-d, --duration",
            r"--periods",
            r"--rank-filter-id",
            r"--rank-filter-output",
            r"--rank-filter-logs",
            r"See also",
        ],
        id="topic_general",
    ),
    pytest.param(
        ["--help=misc"],
        [
            r"{prog} --help=misc",
            r"-i, --inlines",
            r"--hsa-interrupt",
            r"See also",
        ],
        id="topic_misc",
    ),
    pytest.param(
        # --help=preset is curated (get_help_topic_map) to render PRESET +
        # DOMAIN + EXPORT sections together; assert the three section headers
        # plus every left-column flag they list.
        ["--help=preset"],
        [
            r"{prog} --help=preset",
            r"\[PRESET OPTIONS\]",
            r"\[DOMAIN OPTIONS\]",
            r"\[EXPORT OPTIONS\]",
            r"--list-presets",
            r"--explain",
            r"--preset",
            r"--gpu",
            r"--rocm",
            r"--cpu",
            r"--parallel",
            r"--export-config",
            r"See also",
        ],
        id="topic_preset",
    ),
    pytest.param(
        ["--help=process"],
        [
            r"{prog} --help=process",
            r"--process-freq",
            r"--process-wait",
            r"--process-duration",
            r"--cpus",
            r"--gpus",
            r"--ai-nics",
            r"See also",
        ],
        id="topic_process",
    ),
    pytest.param(
        ["--help=profiling"],
        [
            r"{prog} --help=profiling",
            r"--profile-format",
            r"--profile-diff",
            r"See also",
        ],
        id="topic_profiling",
    ),
    pytest.param(
        ["--help=sampling"],
        [
            r"{prog} --help=sampling",
            r"-f, --sampling-freq",
            r"-t, --tids",
            r"--sampling-wait",
            r"--sampling-duration",
            r"--sample-cputime",
            r"--sample-realtime",
            r"--sample-overflow",
            r"--sampling-ainics",
            r"--sampling-cputime-delay",
            r"--sampling-cputime-freq",
            r"--sampling-cputime-signal",
            r"--sampling-cputime-tids",
            r"--sampling-include-inlines",
            r"--sampling-keep-internal",
            r"--sampling-overflow-event",
            r"--sampling-overflow-freq",
            r"--sampling-overflow-signal",
            r"--sampling-overflow-tids",
            r"--sampling-realtime-delay",
            r"--sampling-realtime-freq",
            r"--sampling-realtime-signal",
            r"--sampling-realtime-tids",
            r"See also",
        ],
        id="topic_sampling",
    ),
    pytest.param(
        ["--help=tracing"],
        [
            r"{prog} --help=tracing",
            r"--trace-file",
            r"--trace-buffer-size",
            r"--trace-fill-policy",
            r"--trace-wait",
            r"--trace-duration",
            r"--trace-periods",
            r"--selected-regions",
            r"--trace-clock-id",
            r"See also",
        ],
        id="topic_tracing",
    ),
    # --- Error / version handling -----------------------------------------
    pytest.param(
        ["--help=does-not-exist"],
        [
            r"Unknown help topic 'does-not-exist'",
            r"Available topics",
        ],
        id="unknown_topic",
    ),
    pytest.param(
        ["--version"],
        [r"{prog} v\d+\.\d+\.\d+"],
        id="version",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("cli-help")
class TestCliHelp(RocprofsysTest):
    """Validate every help / informational invocation of the rocprof-sys launchers."""

    @pytest.mark.parametrize("target", TARGETS)
    @pytest.mark.parametrize("run_args, pass_regex", HELP_CASES)
    def test(self, target, run_args, pass_regex):
        # Substitute the program name into name-dependent patterns.
        pass_regex = [p.replace("{prog}", target) for p in pass_regex]

        # rocprof-sys-run additionally exposes an [EXECUTION OPTIONS] section in
        # the full dump; rocprof-sys-sample does not.
        if target == "rocprof-sys-run" and run_args == ["--help=all"]:
            pass_regex = pass_regex + [r"\[EXECUTION OPTIONS\]"]

        result = self.run_test(
            "baseline",
            target=target,
            run_args=run_args,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.parametrize("target", TARGETS)
    def test_explain_invalid(self, target):
        """Negative: --explain with an unknown preset exits non-zero and
        directs the user to --list-presets.

        Kept out of HELP_CASES because that suite assumes a zero exit code;
        an invalid --explain is a hard failure (unlike an unknown --help topic).
        """
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--explain=invalid-preset-xyz"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"not found", r"--list-presets"],
            use_abort_fail_regex=False,  # negative test intentionally exits non-zero
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_explain_requires_value(self, target):
        """Negative: --explain with no value is an argument-validation error
        (distinct from an unknown preset name)."""
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--explain="],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            use_abort_fail_regex=False,  # negative test intentionally exits non-zero
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_unrecognized_option(self, target):
        """Negative: an unknown command-line option is rejected with a non-zero exit."""
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--does-not-exist-flag"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            use_abort_fail_regex=False,  # negative test intentionally exits non-zero
        )
