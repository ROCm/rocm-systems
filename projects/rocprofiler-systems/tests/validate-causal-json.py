#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import re
import sys
import json
import math
import argparse
from collections import OrderedDict


num_stddev = 1.0


def mean(_data):
    return sum(_data) / float(len(_data)) if len(_data) > 0 else 0.0


def stddev(_data):
    if len(_data) == 0:
        return 0.0
    _mean = mean(_data)
    _variance = sum([((x - _mean) ** 2) for x in _data]) / float(len(_data))
    return float(num_stddev) * math.sqrt(_variance)


def simpsons_rule(a, b, fa, fb):
    """Simple numerical integration via Simpson's rule

    https://en.m.wikipedia.org/wiki/Simpson%27s_rule
    """
    slope = (fb - fa) / (b - a)
    # f(x) at midpoint
    fm = fa + (0.5 * (b - a) * slope)

    factor = (b - a) / 6.0
    # print(
    #    f"[{a:8.3f} : {b:8.3f}|{fa:8.3f} : {fb:8.3f}][slope={slope:8.3f}] {factor:8.3f} * ({fa:8.3f} + (4.0 * {fm:8.3f}) + {fb:8.3f})"
    # )
    return factor * (fa + (4.0 * fm) + fb)


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
            """On GitHub Action servers, you typically only get two CPUs, which may be one
            core with two hyperthreads. The hyperthreading can causes the speedup potential
            to drop. Furthermore, these are typically shared resources so the runtime may
            vary significantly. Thus, always account for stddev to prevent failures due to
            these causes
            """
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
            sys.stderr.write(
                f"    [{_exp_name}][{_pp_name}][{_virt_speedup}] Tolerance increased: {_reason} ({self.tolerance:.3f} increased to {_tolerance:.3f})...\n"
            )

        def _compute(_speedup_v, _tolerance_v):
            return _speedup_v >= (self.program_speedup - _tolerance_v) and _speedup_v <= (
                self.program_speedup + _tolerance_v
            )

        return _compute(_prog_speedup, _tolerance)


class throughput_point(object):
    def __init__(self, _speedup):
        self.speedup = _speedup
        self.delta = []
        self.duration = []

    def __iadd__(self, _data):
        self.delta += [float(_data[0])]
        self.duration += [float(_data[1])]

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

    def __len__(self):
        return len(self.duration)

    def __eq__(self, rhs):
        return self.speedup == rhs.speedup

    def __neq__(self, rhs):
        return not self == rhs

    def __lt__(self, rhs):
        return self.speedup < rhs.speedup

    def get_data(self):
        _duration = sum(self.duration)
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
        _line_stddev = self.compute_speedup_stddev()  # 3 stddev == 99.87%
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


def process_data(data, _data, args):
    def find_or_insert(_data, _value, _type):
        if _value not in _data:
            if _type == "throughput":
                _data[_value] = throughput_point(_value)
            elif _type == "latency":
                _data[_value] = latency_point(_value)
        return _data[_value]

    if not _data:
        return data

    _selection_filter = re.compile(args.experiments)
    _progresspt_filter = re.compile(args.progress_points)

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
                        itr = find_or_insert(data[_selected][_name], _speedup, "latency")
                        itr += [pts["arrival"], pts["departure"], _duration]
                else:
                    _delt = pts["laps"]
                    if _delt > 0:
                        itr = find_or_insert(data[_selected][_name], _speedup)
                        itr += [_delt, _duration]

    return data


def compute_speedups(_data, args):
    data = {}
    for selected, pitr in _data.items():
        if selected not in data:
            data[selected] = {}

        for progpt, ditr in pitr.items():
            data[selected][progpt] = OrderedDict(sorted(ditr.items()))

    from os.path import dirname

    ret = []
    for selected, pitr in _data.items():
        for progpt, ditr in pitr.items():
            if 0 not in ditr.keys():
                continue
            _baseline = ditr[0].mean()
            for speedup, itr in ditr.items():
                if len(args.speedups) > 0 and speedup not in args.speedups:
                    continue
                if speedup != itr.speedup:
                    raise ValueError(f"in {selected}: {speedup} != {itr.speedup}")
                if len(itr) >= args.min_experiments:
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

    _data = []
    for itr in result:
        _data.append(experiment_progress(itr))

    _data.sort()
    return _data


def get_validations(args):
    data = []
    _len = len(args.validate)
    if _len == 0:
        return data
    elif _len % 5 != 0:
        raise ValueError(
            "validation requires format: {experiment regex} {progress-point regex} {virtual-speedup} {expected-speedup} {tolerance} (i.e. 5 args per validation. There are {} extra/missing arguments".format(
                _len % 5
            )
        )

    v = args.validate
    for i in range(int(_len / 5)):
        off = 5 * i
        data.append(
            validation(v[off + 0], v[off + 1], v[off + 2], v[off + 3], v[off + 4])
        )

    return data


def main():
    import argparse

    global num_stddev

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--experiments", type=str, help="Regex for experiments", default=".*"
    )
    parser.add_argument(
        "-p",
        "--progress-points",
        type=str,
        help="Regex for progress points",
        default=".*",
    )
    parser.add_argument(
        "-n", "--num-points", type=int, help="Minimum number of data points", default=5
    )
    parser.add_argument(
        "-m",
        "--min-experiments",
        type=int,
        help="Minimum number of experiments per speedup (e.g. do not display speedups when there are fewer than X experiments at this speedup)",
        default=2,
    )
    parser.add_argument(
        "-i", "--input", type=str, nargs="*", help="Input file(s)", required=True
    )
    parser.add_argument(
        "-s",
        "--speedups",
        type=int,
        help="List of speedup values to report",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "-d",
        "--stddev",
        type=float,
        help="Number of standard deviations to report",
        default=1.0,
    )
    parser.add_argument(
        "-v",
        "--validate",
        type=str,
        nargs="*",
        help="Validate speedup: {experiment regex} {progress-point regex} {virtual-speedup} {expected-speedup} {tolerance}",
        default=[],
    )
    parser.add_argument(
        "--samples",
        type=float,
        help="Report samples within this percentage of the peak (0.0, 100.0] (default: 95 percent)",
        default=95.0,
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="{}. {}".format(
            "Accept speedup predictions when: (A) virtual speedup > 10 and (B) prediction is within the tolerance after being increased by (0.5 * stddev) and (1.0 * stddev)",
            "This is primarily used for the CI where the two threads commonly run on 1 CPU core with 2 hyperthreads (causing the speedup potential to drop)",
        ),
    )

    args = parser.parse_args()

    num_stddev = args.stddev
    num_speedups = len(args.speedups)

    percent_samples = args.samples
    if not percent_samples > 0.0 and not percent_samples <= 100.0:
        raise ValueError(
            f"Invalid samples value: {percent_samples}. Supported range: 0.0 < x <= 100.0"
        )
    percent_samples = 1.0 - (percent_samples / 100.0)

    if num_speedups > 0 and args.num_points > num_speedups:
        args.num_points = num_speedups

    data = {}
    samp = {}
    for inp in args.input:
        with open(inp, "r") as f:
            inp_data = json.load(f)
        data = process_data(data, inp_data, args)
        samp = process_samples(samp, inp_data)

    print("Samples:")
    width = max([int(math.log10(x) + 1) for _, x in samp.items()])
    samp_peak = max([count for _, count in samp.items()])
    for name, count in sorted(samp.items(), key=lambda x: x[1], reverse=True):
        if count >= samp_peak * percent_samples:
            print(f"    {count:{width}} :: {name}")

    results = compute_speedups(data, args)
    print("")
    print("Experiments:")
    for itr in results:
        if len(itr) < args.num_points:
            continue
        print("")
        # split each line, indent each line, and join again into single string
        print("{}".format("\n".join([f"    {x}" for x in f"{itr}".split("\n")])))

    sys.stdout.flush()

    validations = get_validations(args)

    expected_validations = len(validations)
    correct_validations = 0
    if expected_validations > 0:
        print(f"\nPerforming {expected_validations} validations...\n")
        for eitr in results:
            _experiment = eitr.data[0].get_name()
            _progresspt = eitr.data[0].prog
            _base_speedup_stddev = eitr.data[0].compute_speedup_stddev()
            for ditr in eitr.data:
                _virt_speedup = ditr.virtual_speedup()
                _prog_speedup = ditr.compute_speedup()
                _prog_speedup_stddev = ditr.compute_speedup_stddev()
                for vitr in validations:
                    _v = vitr.validate(
                        _experiment,
                        _progresspt,
                        _virt_speedup,
                        _prog_speedup,
                        _prog_speedup_stddev,
                        _base_speedup_stddev,
                        args.ci,
                    )
                    if _v is None:
                        continue

                    if _v is True:
                        correct_validations += 1
                    else:
                        sys.stderr.write(
                            f"\n    [{_experiment}][{_progresspt}][{_virt_speedup}] failed validation: {_prog_speedup:8.3f} != {vitr.program_speedup} +/- {vitr.tolerance}\n\n"
                        )

    if expected_validations != correct_validations:
        sys.stderr.flush()
        sys.stderr.write(
            f"\nCausal profiling predictions not validated. Expected {expected_validations}, found {correct_validations}\n"
        )
        sys.stderr.flush()
        sys.exit(-1)
    elif expected_validations > 0:
        print(f"Causal profiling predictions validated: {expected_validations}")


if __name__ == "__main__":
    main()
