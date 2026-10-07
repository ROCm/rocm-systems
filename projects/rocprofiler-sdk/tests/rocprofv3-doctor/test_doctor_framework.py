#!/usr/bin/env python3

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

"""GPU-free unit tests for the rocprofv3-doctor check framework.

Covers the registry (ordering, duplicate/cycle detection), the runner
(skip-propagation, exception isolation), filtering, and the JSON/exit-code
contracts. No test here touches the real machine.
"""

import json

import pytest

from conftest import FakeAccessor


@pytest.fixture
def framework(rocprofv3_package):
    from rocprofv3 import doctor_registry, doctor_result

    return (doctor_registry, doctor_result)


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------
def test_doctor_framework_result_defaults(framework):
    _, doctor_result = framework
    result = doctor_result.Result(doctor_result.STATUS_PASS)
    assert result.detail == ""
    assert result.remediation == ""
    assert result.data == {}


def test_doctor_framework_result_rejects_bad_status(framework):
    _, doctor_result = framework
    with pytest.raises(ValueError):
        doctor_result.Result("bogus")


def test_doctor_framework_result_data_is_copied(framework):
    """A caller mutating the dict it passed in must not alter the result."""
    _, doctor_result = framework
    payload = {"a": 1}
    result = doctor_result.Result(doctor_result.STATUS_PASS, data=payload)
    payload["a"] = 2
    assert result.data["a"] == 1

    result.to_dict()["data"]["a"] = 99
    assert result.data["a"] == 1


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def _dummy(status_module, status):
    def fn(accessor):
        return status_module.Result(status)

    return fn


def test_doctor_framework_registry_ordering(framework):
    doctor_registry, doctor_result = framework
    registry = doctor_registry.Registry()
    for check_id, order in (("b.two", 20), ("a.one", 20), ("c.zero", 5)):
        registry.register(
            id=check_id, group=check_id.split(".")[0], title=check_id, order=order
        )(_dummy(doctor_result, doctor_result.STATUS_PASS))

    ids = [check.id for check in registry.get_checks()]
    assert ids == ["c.zero", "a.one", "b.two"]


def test_doctor_framework_registry_rejects_duplicate_id(framework):
    doctor_registry, doctor_result = framework
    registry = doctor_registry.Registry()
    registry.register(id="x.one", group="x", title="x")(
        _dummy(doctor_result, doctor_result.STATUS_PASS)
    )
    with pytest.raises(ValueError):
        registry.register(id="x.one", group="x", title="x")(
            _dummy(doctor_result, doctor_result.STATUS_PASS)
        )


def test_doctor_framework_registry_unknown_dependency(framework):
    doctor_registry, doctor_result = framework
    registry = doctor_registry.Registry()
    registry.register(id="x.one", group="x", title="x", depends=["nope.missing"])(
        _dummy(doctor_result, doctor_result.STATUS_PASS)
    )
    with pytest.raises(ValueError):
        registry.validate_dependency_graph()


def test_doctor_framework_registry_cycle_detected(framework):
    doctor_registry, doctor_result = framework
    registry = doctor_registry.Registry()
    registry.register(id="x.a", group="x", title="a", depends=["x.b"])(
        _dummy(doctor_result, doctor_result.STATUS_PASS)
    )
    registry.register(id="x.b", group="x", title="b", depends=["x.a"])(
        _dummy(doctor_result, doctor_result.STATUS_PASS)
    )
    with pytest.raises(ValueError):
        registry.validate_dependency_graph()


def test_doctor_framework_global_registry_is_valid(framework):
    """The real check catalog must have a consistent dependency graph."""
    from rocprofv3 import doctor

    doctor.validate_registry()
    assert len(doctor.get_checks()) > 0


def test_doctor_framework_global_registry_ids_unique(framework):
    from rocprofv3 import doctor

    ids = [check.id for check in doctor.get_checks()]
    assert len(ids) == len(set(ids))


def test_doctor_framework_global_registry_sorted(framework):
    from rocprofv3 import doctor

    checks = doctor.get_checks()
    keys = [(check.order, check.id) for check in checks]
    assert keys == sorted(keys)


# ----------------------------------------------------------------------
# runner: skip-propagation and isolation
# ----------------------------------------------------------------------
def _build_registry(doctor_registry, doctor_result, specs):
    """specs: list of (id, depends, callable)."""
    registry = doctor_registry.Registry()
    order = 0
    for check_id, depends, fn in specs:
        order += 1
        registry.register(
            id=check_id,
            group=check_id.split(".")[0],
            title=check_id,
            depends=depends,
            order=order,
        )(fn)
    return registry


def test_doctor_framework_skip_propagation(framework):
    doctor_registry, doctor_result = framework
    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.a", None, _dummy(doctor_result, doctor_result.STATUS_FAIL)),
            ("g.b", ["g.a"], _dummy(doctor_result, doctor_result.STATUS_PASS)),
        ],
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    statuses = dict((check.id, result.status) for check, result in outcomes)

    assert statuses["g.a"] == doctor_result.STATUS_FAIL
    assert statuses["g.b"] == doctor_result.STATUS_SKIP
    detail = dict((c.id, r.detail) for c, r in outcomes)["g.b"]
    assert "g.a" in detail, "skip reason must name the failing dependency"


def test_doctor_framework_skip_propagates_transitively(framework):
    """One root cause must not produce a cascade of red further downstream."""
    doctor_registry, doctor_result = framework
    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.a", None, _dummy(doctor_result, doctor_result.STATUS_FAIL)),
            ("g.b", ["g.a"], _dummy(doctor_result, doctor_result.STATUS_PASS)),
            ("g.c", ["g.b"], _dummy(doctor_result, doctor_result.STATUS_PASS)),
        ],
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    statuses = dict((check.id, result.status) for check, result in outcomes)

    assert statuses["g.c"] == doctor_result.STATUS_SKIP
    fails = [s for s in statuses.values() if s == doctor_result.STATUS_FAIL]
    assert len(fails) == 1


def test_doctor_framework_warn_does_not_skip_dependents(framework):
    doctor_registry, doctor_result = framework
    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.a", None, _dummy(doctor_result, doctor_result.STATUS_WARN)),
            ("g.b", ["g.a"], _dummy(doctor_result, doctor_result.STATUS_PASS)),
        ],
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    statuses = dict((check.id, result.status) for check, result in outcomes)
    assert statuses["g.b"] == doctor_result.STATUS_PASS


def test_doctor_framework_exception_isolation(framework):
    doctor_registry, doctor_result = framework

    def boom(accessor):
        raise RuntimeError("synthetic failure")

    registry = _build_registry(doctor_registry, doctor_result, [("g.a", None, boom)])
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    _, result = outcomes[0]

    # a broken check is a doctor bug, not a finding about the system
    assert result.status == doctor_result.STATUS_ERROR
    assert "traceback" in result.data
    assert "synthetic failure" in result.data["traceback"]


def test_doctor_framework_exception_does_not_stop_later_checks(framework):
    doctor_registry, doctor_result = framework

    def boom(accessor):
        raise RuntimeError("synthetic failure")

    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.a", None, boom),
            ("g.b", None, _dummy(doctor_result, doctor_result.STATUS_PASS)),
        ],
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    statuses = dict((check.id, result.status) for check, result in outcomes)
    assert statuses["g.b"] == doctor_result.STATUS_PASS


def test_doctor_framework_non_result_return_is_an_error(framework):
    doctor_registry, doctor_result = framework
    registry = _build_registry(
        doctor_registry, doctor_result, [("g.a", None, lambda accessor: "oops")]
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    assert outcomes[0][1].status == doctor_result.STATUS_ERROR


# ----------------------------------------------------------------------
# --only / --skip filtering
# ----------------------------------------------------------------------
def _assert_matches_plus_prerequisites(selected, matches):
    """Every selected check matches, or is a prerequisite of one that does."""
    chosen = [check for check in selected if matches(check)]
    assert chosen
    needed = set()
    for check in selected:
        needed.update(check.depends)
    for check in selected:
        assert matches(check) or check.id in needed, check.id


def test_doctor_framework_only_by_group(framework):
    from rocprofv3 import doctor

    selected = doctor.select_checks(only=["driver"])
    _assert_matches_plus_prerequisites(selected, lambda c: c.group == "driver")


def test_doctor_framework_only_by_id_glob(framework):
    from rocprofv3 import doctor

    selected = doctor.select_checks(only=["driver.kfd-*"])
    _assert_matches_plus_prerequisites(selected, lambda c: c.id.startswith("driver.kfd-"))


def test_doctor_framework_skip_by_group(framework):
    from rocprofv3 import doctor

    selected = doctor.select_checks(skip=["smoke"])
    assert all(check.group != "smoke" for check in selected)


def test_doctor_framework_multiple_only_patterns_are_ored(framework):
    from rocprofv3 import doctor

    selected = doctor.select_checks(only=["driver", "python"])
    _assert_matches_plus_prerequisites(
        selected, lambda c: c.group in ("driver", "python")
    )
    groups = set(check.group for check in selected)
    assert {"driver", "python"} <= groups


def test_doctor_framework_smoke_is_opt_in(framework):
    from rocprofv3 import doctor

    default_groups = set(check.group for check in doctor.select_checks())
    assert "smoke" not in default_groups

    enabled = doctor.select_checks(include_default_disabled=True)
    assert any(check.group == "smoke" for check in enabled)


def test_doctor_framework_skip_wins_over_only(framework):
    from rocprofv3 import doctor

    selected = doctor.select_checks(only=["driver"], skip=["driver.kfd-device"])
    assert all(check.id != "driver.kfd-device" for check in selected)


# ----------------------------------------------------------------------
# exit codes
# ----------------------------------------------------------------------
def _outcomes(doctor_registry, doctor_result, statuses):
    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.c{}".format(idx), None, _dummy(doctor_result, status))
            for idx, status in enumerate(statuses)
        ],
    )
    return doctor_registry.run_checks(FakeAccessor(), registry.get_checks())


def test_doctor_framework_exit_zero_when_all_pass(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    outcomes = _outcomes(doctor_registry, doctor_result, [doctor_result.STATUS_PASS] * 3)
    assert doctor.exit_code(outcomes) == 0


def test_doctor_framework_exit_zero_when_only_warnings(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    outcomes = _outcomes(
        doctor_registry,
        doctor_result,
        [doctor_result.STATUS_PASS, doctor_result.STATUS_WARN],
    )
    assert doctor.exit_code(outcomes) == 0


def test_doctor_framework_exit_zero_when_only_skips(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    outcomes = _outcomes(
        doctor_registry,
        doctor_result,
        [doctor_result.STATUS_PASS, doctor_result.STATUS_SKIP],
    )
    assert doctor.exit_code(outcomes) == 0


def test_doctor_framework_exit_one_when_any_fails(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    outcomes = _outcomes(
        doctor_registry,
        doctor_result,
        [doctor_result.STATUS_PASS, doctor_result.STATUS_FAIL],
    )
    assert doctor.exit_code(outcomes) == 1


# ----------------------------------------------------------------------
# JSON schema shape
# ----------------------------------------------------------------------
@pytest.fixture
def healthy_report(framework):
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    accessor = make_healthy_accessor()
    outcomes = doctor.run(accessor)
    return doctor.render_json(
        outcomes,
        tool_version="1.4.0",
        rocm_version="10.0.0",
        rocm_root=accessor.rocm_root,
    )


def test_doctor_framework_json_is_parseable(healthy_report):
    json.loads(healthy_report)


def test_doctor_framework_json_schema_version(healthy_report):
    assert json.loads(healthy_report)["schema_version"] == 1


def test_doctor_framework_json_top_level_fields(healthy_report):
    data = json.loads(healthy_report)
    for key in (
        "schema_version",
        "tool_version",
        "rocm_version",
        "rocm_root",
        "timestamp_utc",
        "summary",
        "checks",
    ):
        assert key in data, key


def test_doctor_framework_json_summary_adds_up(healthy_report):
    summary = json.loads(healthy_report)["summary"]
    total = sum(summary[key] for key in ("pass", "warn", "fail", "skip"))
    assert total == summary["total"]


def test_doctor_framework_json_check_fields(healthy_report):
    data = json.loads(healthy_report)
    assert data["checks"]
    for entry in data["checks"]:
        for key in (
            "id",
            "group",
            "title",
            "severity",
            "status",
            "detail",
            "remediation",
            "data",
        ):
            assert key in entry, "{} missing {}".format(entry.get("id"), key)


def test_doctor_framework_json_status_values_are_valid(healthy_report):
    data = json.loads(healthy_report)
    for entry in data["checks"]:
        assert entry["status"] in ("pass", "warn", "fail", "skip")
        assert entry["severity"] in ("error", "warning", "info")


def test_doctor_framework_json_covers_every_selected_check(framework, healthy_report):
    from rocprofv3 import doctor

    expected = set(check.id for check in doctor.select_checks())
    actual = set(entry["id"] for entry in json.loads(healthy_report)["checks"])
    assert actual == expected


# ----------------------------------------------------------------------
# text rendering
# ----------------------------------------------------------------------
def test_doctor_framework_text_report_has_no_color_by_default(framework):
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    outcomes = doctor.run(make_healthy_accessor())
    text = doctor.render_text(outcomes, use_color=False)
    assert "\033[" not in text
    assert "Summary:" in text


def test_doctor_framework_text_report_colors_when_asked(framework):
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    outcomes = doctor.run(make_healthy_accessor())
    text = doctor.render_text(outcomes, use_color=True)
    assert "\033[" in text


def test_doctor_framework_quiet_hides_passing_checks(framework):
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    outcomes = doctor.run(make_healthy_accessor())
    quiet = doctor.render_text(outcomes, quiet=True)
    assert "[ PASS ]" not in quiet


def test_doctor_framework_list_checks_includes_known_ids(framework):
    from rocprofv3 import doctor

    listing = doctor.render_check_list()
    for check_id in ("install.rocm-root", "driver.kfd-device", "python.version"):
        assert check_id in listing


# ----------------------------------------------------------------------
# selection and scheduling contract (design review 2026-10-06)
# ----------------------------------------------------------------------
def test_doctor_framework_unmatched_only_is_an_error(framework):
    """An --only typo must not produce an empty run that exits 0."""
    from rocprofv3 import doctor

    with pytest.raises(doctor.SelectionError) as info:
        doctor.select_checks(only=["install", "does-not-exist"])
    assert "does-not-exist" in str(info.value)
    assert "install" not in str(info.value).replace("does-not-exist", "")


def test_doctor_framework_only_pulls_in_prerequisites(framework):
    from rocprofv3 import doctor

    ids = [check.id for check in doctor.select_checks(only=["runtime.libraries-load"])]
    assert "runtime.libraries-load" in ids
    assert "install.sdk-library" in ids
    assert "install.rocm-root" in ids


def test_doctor_framework_skipped_prerequisite_skips_dependent(framework):
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    outcomes = doctor.run(
        make_healthy_accessor(),
        only=["runtime.libraries-load"],
        skip=["install.sdk-library"],
    )
    results = dict((check.id, result) for check, result in outcomes)
    assert "install.sdk-library" not in results
    result = results["runtime.libraries-load"]
    assert result.status == doctor.STATUS_SKIP
    assert "excluded from this run" in result.detail


def test_doctor_framework_prerequisites_run_first_regardless_of_order(framework):
    doctor_registry, doctor_result = framework
    registry = doctor_registry.Registry()
    seen = []

    def record(name, status):
        def fn(accessor):
            seen.append(name)
            return doctor_result.Result(status)

        return fn

    # "g.late" has the lower order number but depends on "g.early"
    registry.register(id="g.late", group="g", title="late", depends=["g.early"], order=1)(
        record("late", doctor_result.STATUS_PASS)
    )
    registry.register(id="g.early", group="g", title="early", order=2)(
        record("early", doctor_result.STATUS_FAIL)
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    assert seen == ["early"]
    statuses = dict((check.id, result.status) for check, result in outcomes)
    assert statuses["g.late"] == doctor_result.STATUS_SKIP


def test_doctor_framework_error_skips_dependents(framework):
    doctor_registry, doctor_result = framework

    def boom(accessor):
        raise RuntimeError("bug")

    registry = _build_registry(
        doctor_registry,
        doctor_result,
        [
            ("g.a", None, boom),
            ("g.b", ["g.a"], _dummy(doctor_result, doctor_result.STATUS_PASS)),
        ],
    )
    outcomes = doctor_registry.run_checks(FakeAccessor(), registry.get_checks())
    assert [result.status for _, result in outcomes] == [
        doctor_result.STATUS_ERROR,
        doctor_result.STATUS_SKIP,
    ]


def test_doctor_framework_exit_code_failure_outranks_check_error(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    check = doctor_registry.Check("g.a", "g", "a", doctor_result.SEV_ERROR, None, [], 1)
    error = doctor_result.Result(doctor_result.STATUS_ERROR)
    fail = doctor_result.Result(doctor_result.STATUS_FAIL)
    ok = doctor_result.Result(doctor_result.STATUS_PASS)
    assert doctor.exit_code([(check, ok), (check, error)]) == doctor.EXIT_TOOL_ERROR
    assert doctor.exit_code([(check, error), (check, fail)]) == doctor.EXIT_FAILURES
    assert doctor.exit_code([(check, ok)]) == doctor.EXIT_OK


def test_doctor_framework_only_selecting_smoke_runs_it(framework):
    """Naming an opt-in check with --only runs it -- no second flag needed --
    and its output directory is owned by the run and removed on success."""
    from rocprofv3 import doctor
    from conftest import make_healthy_accessor

    accessor = make_healthy_accessor(runs={"--kernel-trace": (0, "", "")})
    accessor._files.add("/bin/true")
    outcomes = doctor.run(accessor, only=["smoke.rocprofv3-launcher"])
    results = dict((check.id, result) for check, result in outcomes)
    result = results["smoke.rocprofv3-launcher"]
    assert result.status == doctor.STATUS_PASS, result.detail
    assert "not verified" in result.detail
    assert accessor.temp_dirs and accessor.removed_trees == accessor.temp_dirs
    launch = [cmd for cmd, _ in accessor.calls if "--kernel-trace" in cmd][0]
    assert launch[launch.index("-d") + 1] == accessor.temp_dirs[0]


def test_doctor_framework_every_check_declares_a_probe(framework):
    from rocprofv3 import doctor

    doctor_registry, doctor_result = framework
    for check in doctor.get_checks():
        assert check.probe in doctor_result.PROBE_VALUES, check.id
    probes = dict((check.id, check.probe) for check in doctor.get_checks())
    assert probes["runtime.libraries-load"] == doctor_result.PROBE_PROCESS
    assert probes["counters.avail-enumeration"] == doctor_result.PROBE_GPU
    assert probes["install.rocm-root"] == doctor_result.PROBE_PASSIVE


def test_doctor_framework_json_reports_probe_and_errors(healthy_report):
    data = json.loads(healthy_report)
    assert "error" in data["summary"]
    for entry in data["checks"]:
        assert entry["probe"] in ("passive", "process", "gpu")


def test_doctor_framework_rocprofv3_doctor_after_separator_is_application_arg(
    rocprofv3_path,
):
    """Review P1: `rocprofv3 ... -- app --doctor` must profile app, not divert
    to the doctor; only rocprofv3's own options are inspected."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rocprofv3_launcher", rocprofv3_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    dispatched = []
    parsed = []

    class Parsed(Exception):
        pass

    def fake_parse(argv=None):
        parsed.append(argv)
        raise Parsed()

    launcher.dispatch_doctor = lambda args: dispatched.append(args) or 0
    launcher.parse_arguments = fake_parse

    with pytest.raises(Parsed):
        launcher.main(["--kernel-trace", "--", "app", "--doctor"])
    assert dispatched == []
    assert parsed == [["--kernel-trace", "--", "app", "--doctor"]]

    assert launcher.main(["--doctor", "--format", "json"]) == 0
    assert dispatched == [["--doctor", "--format", "json"]]


def _load_launcher(rocprofv3_path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, rocprofv3_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_doctor_framework_rocprofv3_doctor_execs_the_libexec_script(
    rocprofv3_path, monkeypatch
):
    """rocprofv3 replaces itself with the doctor (no lingering parent process),
    found under libexec, with --doctor removed from the forwarded options."""
    launcher = _load_launcher(rocprofv3_path, "rocprofv3_launcher_exec")
    calls = []

    def fake_execv(path, argv):
        calls.append((path, argv))
        raise SystemExit(0)

    monkeypatch.setattr(launcher.os, "execv", fake_execv)
    with pytest.raises(SystemExit):
        launcher.main(["--doctor", "--only", "install"])

    (path, argv) = calls[0]
    script = argv[1]
    assert path == argv[0]
    assert "libexec" in script.split("/")
    assert script.endswith(("rocprofv3-doctor", "rocprofv3-doctor.py"))
    assert argv[2:] == ["--only", "install"]


def test_doctor_framework_rocprofv3_doctor_reports_missing_script(
    rocprofv3_path, monkeypatch, capsys
):
    launcher = _load_launcher(rocprofv3_path, "rocprofv3_launcher_missing")
    monkeypatch.setattr(launcher, "find_doctor_script", lambda: None)
    assert launcher.main(["--doctor"]) == 2
    assert "not installed" in capsys.readouterr().err
