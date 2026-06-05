#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Pure-Python causal profiling validation logic.

Ported from validate-causal-json.py (rocprofiler-systems/tests/).
Provides data processing, speedup computation, and validation helpers
consumed by CausalReader. Also serves as the standalone ``rocprof-sys-causal-print``
utility via the ``main()`` entry point below (this module is self-contained —
standard library only — so it runs as a single-file script).
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def mean(_data):
    return sum(_data) / float(len(_data)) if len(_data) > 0 else 0.0


def stddev(_data, num_stddev=1.0):
    if len(_data) == 0:
        return 0.0
    _mean = mean(_data)
    _variance = sum([((x - _mean) ** 2) for x in _data]) / float(len(_data))
    return float(num_stddev) * math.sqrt(_variance)


def simpsons_rule(a, b, fa, fb):
    """Simple numerical integration via Simpson's rule.

    https://en.m.wikipedia.org/wiki/Simpson%27s_rule
    """
    slope = (fb - fa) / (b - a)
    fm = fa + (0.5 * (b - a) * slope)
    factor = (b - a) / 6.0
    return factor * (fa + (4.0 * fm) + fb)


# ---------------------------------------------------------------------------
# Data model classes
# ---------------------------------------------------------------------------


class validation(object):
    def __init__(self, _exp_re, _pp_re, _virt, _expected, _tolerance):
        self.experiment_filter = re.compile(_exp_re)
        self.progress_pt_filter = re.compile(_pp_re)
        self.virtual_speedup = int(_virt)
        self.program_speedup = float(_expected)
        self.tolerance = float(_tolerance)

    def validate(
        self,
        _exp_name,
        _pp_name,
        _virt_speedup,
        _prog_speedup,
        _prog_speedup_stddev,
        _base_speedup_stddev,
        _ci=False,
        stderr_buf=None,
    ):
        if (
            not re.search(self.experiment_filter, _exp_name)
            or not re.search(self.progress_pt_filter, _pp_name)
            or _virt_speedup != self.virtual_speedup
        ):
            return None

        _tolerance = self.tolerance
        _reason = "[unspecified reason]"
        if _ci is True:
            _tolerance += max([_base_speedup_stddev, _prog_speedup_stddev])
            _reason = "results obtained on a shared CI system... potentially artificially deflating speedup predictions"
        elif _base_speedup_stddev > self.tolerance:
            _tolerance += math.sqrt(_base_speedup_stddev)
            _reason = (
                f"large standard deviation of the baseline ({_base_speedup_stddev:.3f})"
            )
        elif _prog_speedup_stddev > 1.0:
            _tolerance += math.sqrt(_prog_speedup_stddev)
            _reason = f"large standard deviation of the program speedup ({_prog_speedup_stddev:.3f})"

        if _tolerance > self.tolerance:
            msg = f"    [{_exp_name}][{_pp_name}][{_virt_speedup}] Tolerance increased: {_reason} ({self.tolerance:.3f} increased to {_tolerance:.3f})...\n"
            if stderr_buf is not None:
                stderr_buf.write(msg)
            else:
                sys.stderr.write(msg)

        def _compute(_speedup_v, _tolerance_v):
            return _speedup_v >= (
                self.program_speedup - _tolerance_v
            ) and _speedup_v <= (self.program_speedup + _tolerance_v)

        return _compute(_prog_speedup, _tolerance)


class throughput_point(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.delta = []
        self.duration = []

    def __iadd__(self, _data):
        self.delta += [float(_data[0])]
        self.duration += [float(_data[1])]
        return self

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def get_data(self):
        return [x / y for x, y in zip(self.duration, self.delta)]

    def mean(self):
        return sum(self.duration) / sum(self.delta)


class latency_point(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.arrivals = []
        self.departures = []
        self.duration = []

    def __iadd__(self, _data):
        self.arrivals += [float(_data[0])]
        self.departures += [float(_data[1])]
        self.duration += [float(_data[2])]
        return self

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def get_data(self):
        return [y / x for x, y in zip(self.arrivals, self.duration)]

    def get_difference(self):
        _duration = sum(self.duration)
        return [x / _duration for x in self.duration]

    def mean(self):
        rate = sum(self.arrivals) / sum(self.duration)
        return sum(self.get_difference()) / rate


class line_speedup(object):
    def __init__(self, _name="", _prog="", _exp_data=None, _exp_base=None):
        self.name = _name
        self.prog = _prog
        self.data = _exp_data
        self.base = _exp_base

    def virtual_speedup(self):
        if self.data is None or self.base is None:
            return 0.0
        return self.data.speedup

    def compute_speedup(self):
        if self.data is None or self.base is None:
            return 0.0
        return ((self.base.mean() - self.data.mean()) / self.base.mean()) * 100

    def compute_speedup_stddev(self):
        if self.data is None or self.base is None:
            return 0.0
        _data = []
        _base = self.base.mean()
        for ditr in self.data.get_data():
            _data += [((_base - ditr) / _base) * 100]
        return stddev(_data)

    def get_name(self):
        return ":".join(
            [
                os.path.basename(x) if os.path.isfile(x) else x
                for x in self.name.split(":")
            ]
        )

    def __str__(self):
        if self.data is None or self.base is None:
            return f"{self.name}"
        _line_speedup = self.compute_speedup()
        _line_stddev = self.compute_speedup_stddev()
        _name = self.get_name()
        return f"[{_name}][{self.prog}][{self.data.speedup:3}] speedup: {_line_speedup:6.1f} +/- {_line_stddev:6.2f} %"

    def __eq__(self, rhs):
        return (
            self.name == rhs.name
            and self.prog == rhs.prog
            and self.data == rhs.data
            and self.base == rhs.base
        )

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        if self.name != rhs.name:
            return self.name < rhs.name
        elif self.prog != rhs.prog:
            return self.prog < rhs.prog
        elif self.data != rhs.data:
            return self.data < rhs.data
        elif self.base != rhs.base:
            return self.base < rhs.base
        return False


class experiment_progress(object):
    def __init__(self, _data):
        self.data = _data

    def get_impact(self):
        speedup_c = [float(x.compute_speedup()) for x in self.data]
        speedup_v = [float(x.virtual_speedup()) for x in self.data]
        impact = []
        for i in range(len(self.data) - 1):
            impact += [
                simpsons_rule(
                    speedup_v[i], speedup_v[i + 1], speedup_c[i], speedup_c[i + 1]
                )
            ]
        return [sum(impact), mean(impact), stddev(impact)]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        _impact_v = self.get_impact()
        _name = self.data[0].get_name()
        _prog = self.data[0].prog
        _impact = [
            f"[{_name}][{_prog}][sum]  impact: {_impact_v[0]:6.1f}",
            f"[{_name}][{_prog}][avg]  impact: {_impact_v[1]:6.1f} +/- {_impact_v[2]:6.2f}",
        ]
        return "\n".join([f"{x}" for x in self.data] + _impact)

    def __lt__(self, rhs):
        self.data.sort()
        return self.get_impact()[0] < rhs.get_impact()[0]


# ---------------------------------------------------------------------------
# Data processing functions
# ---------------------------------------------------------------------------


def process_samples(data, _data):
    if not _data:
        return data
    for record in _data["rocprofsys"]["causal"]["records"]:
        for samp in record["samples"]:
            _info = samp["info"]
            _count = samp["count"]
            _func = _info["dfunc"]
            if _func not in data:
                data[_func] = 0
            data[_func] += _count
            for dwarf_entry in _info["dwarf_info"]:
                _name = "{}:{}".format(dwarf_entry["file"], dwarf_entry["line"])
                if _name not in data:
                    data[_name] = 0
                data[_name] += _count
    return data


def process_data(data, _data, experiments_filter, progress_points_filter):
    def find_or_insert(_data, _value, _type):
        if _value not in _data:
            if _type == "throughput":
                _data[_value] = throughput_point(_value)
            elif _type == "latency":
                _data[_value] = latency_point(_value)
        return _data[_value]

    if not _data:
        return data

    _selection_filter = re.compile(experiments_filter)
    _progresspt_filter = re.compile(progress_points_filter)

    for record in _data["rocprofsys"]["causal"]["records"]:
        for exp in record["experiments"]:
            _speedup = exp["virtual_speedup"]
            _duration = exp["duration"]
            _file = exp["selection"]["info"]["file"]
            _line = exp["selection"]["info"]["line"]
            _func = exp["selection"]["info"]["dfunc"]
            _sym_addr = exp["selection"]["symbol_address"]
            _selected = ":".join([_file, f"{_line}"]) if _sym_addr == 0 else _func
            if not re.search(_selection_filter, _selected):
                continue
            if _selected not in data:
                data[_selected] = {}
            for pts in exp["progress_points"]:
                _name = pts["name"]
                if not re.search(_progresspt_filter, _name):
                    continue
                if _name not in data[_selected]:
                    data[_selected][_name] = {}
                if "delta" in pts:
                    _delt = pts["delta"]
                    if _delt > 0:
                        itr = find_or_insert(
                            data[_selected][_name], _speedup, "throughput"
                        )
                        itr += [_delt, _duration]
                    elif "arrival" in pts and pts["arrival"] > 0:
                        itr = find_or_insert(
                            data[_selected][_name], _speedup, "latency"
                        )
                        itr += [pts["arrival"], pts["departure"], _duration]
                else:
                    _delt = pts["laps"]
                    if _delt > 0:
                        itr = find_or_insert(data[_selected][_name], _speedup, "throughput")
                        itr += [_delt, _duration]

    return data


def compute_speedups(_data, speedups, min_experiments):
    data = {}
    for selected, pitr in _data.items():
        if selected not in data:
            data[selected] = {}
        for progpt, ditr in pitr.items():
            data[selected][progpt] = OrderedDict(sorted(ditr.items()))

    ret = []
    for selected, pitr in _data.items():
        for progpt, ditr in pitr.items():
            if 0 not in ditr.keys():
                continue
            for spdup, itr in ditr.items():
                if len(speedups) > 0 and spdup not in speedups:
                    continue
                if spdup != itr.speedup:
                    raise ValueError(f"in {selected}: {spdup} != {itr.speedup}")
                if len(itr) >= min_experiments:
                    _val = line_speedup(selected, progpt, itr, ditr[0])
                    ret.append(_val)

    ret.sort()
    _last_name = None
    _last_prog = None
    result = []
    for itr in ret:
        if itr.name != _last_name or itr.prog != _last_prog:
            result.append([])
        result[-1].append(itr)
        _last_name = itr.name
        _last_prog = itr.prog

    _data2 = []
    for itr in result:
        _data2.append(experiment_progress(itr))

    _data2.sort()
    return _data2


def get_validations(validate_tuples):
    """Build validation objects from a list of (exp_re, pp_re, virt, expected, tol) tuples."""
    data = []
    for v in validate_tuples:
        data.append(validation(v[0], v[1], v[2], v[3], v[4]))
    return data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_causal_json(
    json_paths: list[Path],
    ci_mode: bool = False,
    experiments: str = ".*",
    progress_points: str = ".*",
    speedups: Optional[list[int]] = None,
    min_experiments: int = 2,
    num_points: int = 5,
    stddev_factor: float = 1.0,
    samples_pct: float = 95.0,
    validations: Optional[list[tuple[str, str, int, float, float]]] = None,
) -> tuple[bool, str, str]:
    """Validate causal profiling JSON output files.

    Args:
        json_paths: List of paths to causal JSON files.
        ci_mode: If True, increase tolerance for CI environments.
        experiments: Regex filter for experiment names.
        progress_points: Regex filter for progress point names.
        speedups: List of virtual speedup values to report. Empty = all.
        min_experiments: Minimum experiments per speedup entry.
        num_points: Minimum number of data points per experiment group.
        stddev_factor: Number of standard deviations to report.
        samples_pct: Report samples within this percentage of peak (0, 100].
        validations: List of (exp_re, pp_re, virt_speedup, expected_speedup, tolerance).

    Returns:
        Tuple of (is_valid, stdout, stderr).
    """
    if speedups is None:
        speedups = []
    if validations is None:
        validations = []

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Validate samples_pct range
    if not (samples_pct > 0.0 and samples_pct <= 100.0):
        stderr_buf.write(
            f"Invalid samples_pct value: {samples_pct}. Supported range: 0.0 < x <= 100.0\n"
        )
        return False, stdout_buf.getvalue(), stderr_buf.getvalue()

    percent_samples_threshold = 1.0 - (samples_pct / 100.0)

    num_speedups = len(speedups)
    effective_num_points = num_points
    if num_speedups > 0 and effective_num_points > num_speedups:
        effective_num_points = num_speedups

    data = {}
    samp = {}
    for json_path in json_paths:
        with open(json_path, "r") as f:
            inp_data = json.load(f)
        data = process_data(data, inp_data, experiments, progress_points)
        samp = process_samples(samp, inp_data)

    # Print samples section
    stdout_buf.write("Samples:\n")
    if samp:
        width = max([int(math.log10(x) + 1) for _, x in samp.items()])
        samp_peak = max([count for _, count in samp.items()])
        for name, count in sorted(samp.items(), key=lambda x: x[1], reverse=True):
            if count >= samp_peak * percent_samples_threshold:
                stdout_buf.write(f"    {count:{width}} :: {name}\n")

    results = compute_speedups(data, speedups, min_experiments)

    stdout_buf.write("\n")
    stdout_buf.write("Experiments:\n")
    for itr in results:
        if len(itr) < effective_num_points:
            continue
        stdout_buf.write("\n")
        stdout_buf.write(
            "{}\n".format("\n".join([f"    {x}" for x in f"{itr}".split("\n")]))
        )

    # Run validations
    validation_objs = get_validations(validations)
    expected_validations = len(validation_objs)
    correct_validations = 0
    is_valid = True

    if expected_validations > 0:
        stdout_buf.write(f"\nPerforming {expected_validations} validations...\n\n")
        for eitr in results:
            _experiment = eitr.data[0].get_name()
            _progresspt = eitr.data[0].prog
            _base_speedup_stddev = eitr.data[0].compute_speedup_stddev()
            for ditr in eitr.data:
                _virt_speedup = ditr.virtual_speedup()
                _prog_speedup = ditr.compute_speedup()
                _prog_speedup_stddev = ditr.compute_speedup_stddev()
                for vitr in validation_objs:
                    _v = vitr.validate(
                        _experiment,
                        _progresspt,
                        _virt_speedup,
                        _prog_speedup,
                        _prog_speedup_stddev,
                        _base_speedup_stddev,
                        ci_mode,
                        stderr_buf=stderr_buf,
                    )
                    if _v is None:
                        continue

                    if _v is True:
                        correct_validations += 1
                    else:
                        stderr_buf.write(
                            f"\n    [{_experiment}][{_progresspt}][{_virt_speedup}] failed validation: {_prog_speedup:8.3f} != {vitr.program_speedup} +/- {vitr.tolerance}\n\n"
                        )

    if expected_validations != correct_validations:
        stderr_buf.write(
            f"\nCausal profiling predictions not validated. Expected {expected_validations}, found {correct_validations}\n"
        )
        is_valid = False
    elif expected_validations > 0:
        stdout_buf.write(f"Causal profiling predictions validated: {expected_validations}\n")

    return is_valid, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point (rocprof-sys-causal-print + in-process pytest validator)
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """Run causal validation/printing from CLI args. Returns 0 on success, 1 on failure.

    Reproduces the argument surface of the former tests/validate-causal-json.py so
    this module can back both the shipped ``rocprof-sys-causal-print`` utility and
    the in-process pytest ``validate_causal_json`` wrapper.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", type=str, default=".*")
    parser.add_argument("-p", "--progress-points", type=str, default=".*")
    parser.add_argument("-n", "--num-points", type=int, default=5)
    parser.add_argument("-m", "--min-experiments", type=int, default=2)
    parser.add_argument("-i", "--input", nargs="*", type=str, required=True)
    parser.add_argument("-s", "--speedups", nargs="*", type=int, default=[])
    parser.add_argument("-d", "--stddev", type=float, default=1.0)
    parser.add_argument("-v", "--validate", nargs="*", type=str, default=[])
    parser.add_argument("--samples", type=float, default=95.0)
    parser.add_argument("--ci", action="store_true")

    args = parser.parse_args(argv)

    vals = args.validate or []
    if len(vals) % 5 != 0:
        sys.stderr.write(
            "--validate requires groups of 5 values: "
            "{experiment} {progress_point} {virtual_speedup} {expected_speedup} {tolerance}\n"
        )
        return 1
    validations = [tuple(vals[i : i + 5]) for i in range(0, len(vals), 5)]

    is_valid, out, err = validate_causal_json(
        json_paths=[Path(p) for p in args.input],
        ci_mode=args.ci,
        experiments=args.experiments,
        progress_points=args.progress_points,
        speedups=args.speedups,
        min_experiments=args.min_experiments,
        num_points=args.num_points,
        stddev_factor=args.stddev,
        samples_pct=args.samples,
        validations=validations,
    )

    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
