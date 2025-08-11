#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import yaml
import argparse


def fatal_error(msg, exit_code=1):
    sys.stderr.write(f"Fatal error: {msg}\n")
    sys.stderr.flush()
    sys.exit(exit_code)


def parse_yaml(yaml_file):
    try:
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
        return data

    except yaml.YAMLError as exc:
        fatal_error(f"{exc}")


def dump_yaml(data, ostream):
    yaml.dump(
        data,
        ostream,
        default_flow_style=False,
        canonical=False,
        indent=1,
        width=120,
        sort_keys=False,
    )


def convert_yaml_schema(inp):
    data = {"rocprofiler-sdk": {}}
    data["rocprofiler-sdk"]["counters-schema-version"] = inp["schema-version"]
    data["rocprofiler-sdk"]["counters"] = []

    for key, itr in inp.items():
        if key == "schema-version":
            pass
        else:
            entry = {
                "name": key,
                "description": itr["description"],
                "properties": [],
                "definitions": [],
            }
            for arch, aitr in itr["architectures"].items():
                definition = {"architectures": sorted(arch.split("/"))}

                # expression or block + event
                for dkey, ditr in aitr.items():
                    definition[dkey] = ditr

                # add the arch-specific counter definition
                entry["definitions"].append(definition)

            # add the counter entry
            data["rocprofiler-sdk"]["counters"].append(entry)

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input file", type=str, required=True)
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (stdout if not provided)",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args(sys.argv[1:])

    inp = parse_yaml(args.input)
    if "rocprofiler-sdk" not in inp.keys():
        data = convert_yaml_schema(inp)
    else:
        data = inp

    if args.output is None:
        dump_yaml(data, sys.stdout)
    else:
        with open(args.output, "w") as outfile:
            dump_yaml(data, outfile)
