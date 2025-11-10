#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

import os
import re
import sys
import glob
import yaml
import shlex
import shutil
import hashlib
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

# Regex patterns for comment stripping
COMMENT_PATTERNS = {
    "c-like": re.compile(
        r"""
        //.*?$         # line comments
        |              # or
        /\*.*?\*/      # block comments
        """,
        re.MULTILINE | re.DOTALL | re.VERBOSE,
    ),
    "python": re.compile(r"#.*?$", re.MULTILINE),
    "cmake": re.compile(r"#.*?$", re.MULTILINE),
}

GLOBAL_ABIDW_EXTRAS = [
    "--load-all-types",
    "--type-id-style",
    "hash",
]

GLOBAL_ABIDIFF_EXTRAS = []


class BooleanArgAction(argparse.Action):
    """Custom argparse action to handle boolean arguments."""

    def __call__(self, parser, args, value, option_string=None):
        setattr(args, self.dest, strtobool(value))


class Version(object):
    """
    Class to represent a versioning specification from VERSION text file.
    """

    def __init__(self, major, minor, patch, build=None, hash=None) -> None:
        self.major = int(major)
        self.minor = int(minor)
        self.patch = int(patch)
        self.build = f"{build}" if build is not None else None
        self.hash = f"{hash}" if hash is not None else None
        self.validate()

    def validate(self):
        if self.major < 0:
            raise ValueError(f"Major version must be non-negative: {self.major}")
        if self.minor < 0:
            raise ValueError(f"Minor version must be non-negative: {self.minor}")
        if self.patch < 0:
            raise ValueError(f"Patch version must be non-negative: {self.patch}")
        return self

    def write(self, path, hash) -> None:
        _path = Path(path).resolve()
        if not hash:
            raise ValueError(
                f"No hash provided to write to VERSION file ('{_path}'): {hash}"
            )
        self.hash = f"{hash}".lower()
        with open(_path, "w") as ofs:
            _contents = f"""
            {self.major}.{self.minor}.{self.patch}
            # hash: {self.hash}
            """
            import textwrap

            _contents = textwrap.dedent(_contents).strip("\n").strip()
            ofs.write(f"{_contents}\n")

    def value(self):
        _factor = 10000
        for key, itr in zip(
            ["major", "minor", "patch"], [self.major, self.minor, self.patch]
        ):
            if itr >= _factor:
                raise ValueError(
                    f"Version component {key} exceeds maximum value ({_factor - 1}): {itr}"
                )
        return (self.major * _factor * _factor) + (self.minor * _factor) + self.patch

    def __str__(self) -> str:
        _data = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            _build = f"{self.build}".lstrip("-").lstrip(".")
            _data = f"{_data}-{_build}"
        if self.hash:
            _data = f"{_data}~{self.hash}"

        return _data

    def __iadd__(self, other):
        self.major += other.major
        self.minor += other.minor
        self.patch += other.patch
        return self.validate()

    def __add__(self, other):
        return Version(
            self.major + other.major,
            self.minor + other.minor,
            self.patch + other.patch,
            self.build if self.build else other.build,
            self.hash if self.hash else other.hash,
        )

    def __isub__(self, other):
        self.major -= other.major
        self.minor -= other.minor
        self.patch -= other.patch
        return self.validate()

    def __sub__(self, other):
        return Version(
            self.major - other.major,
            self.minor - other.minor,
            self.patch - other.patch,
            self.build if self.build else other.build,
            self.hash if self.hash else other.hash,
        )

    def get(self, name, default=None):
        if not hasattr(self, name) and hasattr(self, name.replace("-", "_")):
            name = name.replace("-", "_")
        return getattr(self, name, default)

    def __eq__(self, other):
        return self.value() == other.value()

    def __lt__(self, other):
        return self.value() < other.value()

    def __le__(self, other):
        return self.value() <= other.value()

    def __gt__(self, other):
        return self.value() > other.value()

    def __ge__(self, other):
        return self.value() >= other.value()


class FileSet(object):
    """
    Class to represent a set of files to include/exclude based on glob patterns.
    """

    def __init__(
        self,
        include_patterns: Iterable[str],
        include_recursive_patterns: Iterable[str],
        exclude_patterns: Iterable[str],
        exclude_recursive_patterns: Iterable[str],
    ) -> None:
        self.include_patterns = include_patterns
        self.include_recursive_patterns = include_recursive_patterns
        self.exclude_patterns = exclude_patterns
        self.exclude_recursive_patterns = exclude_recursive_patterns

        def _glob(_patterns: Iterable[str], _recursive: bool) -> List[Path]:
            # Expand globs
            return [
                pitr
                for itr in _patterns
                for pitr in glob.glob(str(itr), recursive=_recursive)
                if Path(pitr).is_file()
            ]

        self.include_paths = sorted(
            list(
                set(
                    _glob(self.include_patterns, _recursive=False)
                    + _glob(self.include_recursive_patterns, _recursive=True)
                )
            )
        )
        self.exclude_paths = sorted(
            list(
                set(
                    _glob(self.exclude_patterns, _recursive=False)
                    + _glob(self.exclude_recursive_patterns, _recursive=True)
                )
            )
        )

        self.paths = [Path(p) for p in self.include_paths if p not in self.exclude_paths]

    def get_paths(self, absolute=False) -> List[Path]:
        return sorted([p.resolve() if absolute else p for p in self.paths])

    def __iadd__(self, other):
        self.include_patterns += other.include_patterns
        self.include_recursive_patterns += other.include_recursive_patterns
        self.exclude_patterns += other.exclude_patterns
        self.exclude_recursive_patterns += other.exclude_recursive_patterns
        self.include_paths = sorted(list(set(self.include_paths + other.include_paths)))
        self.exclude_paths = sorted(list(set(self.exclude_paths + other.exclude_paths)))
        self.paths = [Path(p) for p in self.include_paths if p not in self.exclude_paths]

        return self

    def __add__(self, other):
        return FileSet(
            self.include_patterns + other.include_patterns,
            self.include_recursive_patterns + other.include_recursive_patterns,
            self.exclude_patterns + other.exclude_patterns,
            self.exclude_recursive_patterns + other.exclude_recursive_patterns,
        )

    def get(self, name, default=None):
        if not hasattr(self, name) and hasattr(self, name.replace("-", "_")):
            name = name.replace("-", "_")
        return getattr(self, name, default)


class VersioningSpec(object):
    """
    Class to represent a versioning specification from versioning.yml.
    """

    def __init__(self, spec_file, args, **kwargs) -> None:

        def _get_file_set(inp, working_dir, section):

            logging.debug(
                f"Generating file set for '{section}' relative to '{working_dir}'...\n\tInput: {inp}"
            )

            def _get_hash_glob_list(key: str) -> List[str]:
                if section not in inp:
                    return []
                return [os.path.join(working_dir, p) for p in inp[section].get(key, [])]

            include = _get_hash_glob_list("include")
            exclude = _get_hash_glob_list("exclude")
            include_recursive = _get_hash_glob_list("recursive-include")
            exclude_recursive = _get_hash_glob_list("recursive-exclude")

            return FileSet(
                include,
                include_recursive,
                exclude,
                exclude_recursive,
            )

        with open(spec_file, "r") as ifs:
            self.spec = yaml.safe_load(ifs)

        # root working directory for relative paths
        root_working_dir = Path(spec_file).resolve().parent

        self.name = self.spec["versioning"].get("name", None)
        self.version = None
        self.build_directory = self.spec["versioning"].get("build-directory", "build")
        self.abidw_args = self.spec["versioning"].get("abidw-args", "")
        self.abidiff_args = self.spec["versioning"].get("abidiff-args", "")

        # command line overrides
        if has_attr(args, "build_directory"):
            self.build_directory = args.build_directory
        if has_attr(args, "abidw_args"):
            self.abidw_args = args.abidw_args
        if has_attr(args, "abidiff_args"):
            self.abidiff_args = args.abidiff_args
        if has_attr(args, "mode"):
            self.mode = args.mode

        self.build_directory = kwargs.get("build_directory", self.build_directory)
        self.abidw_args = kwargs.get("abidw_args", self.abidw_args)
        self.abidiff_args = kwargs.get("abidiff_args", self.abidiff_args)
        self.mode = kwargs.get("mode", self.mode)

        if self.mode is None:
            raise argparse.ArgumentError(None, message="-m / --mode must be specified.")

        for itr in ["source", "build", "install"]:
            _spec = self.spec["versioning"].get(f"{itr}-tree", None)
            working_directory = _spec.get("working-directory", None)

            if working_directory is not None:
                working_directory = (
                    Path(working_directory)
                    if Path(working_directory).is_absolute()
                    else Path(root_working_dir / working_directory).resolve()
                )
            else:
                working_directory = root_working_dir

            if self.mode == "build":
                if itr == "source":
                    # no need to make any modifications
                    pass
                elif itr == "build":
                    # configure relative to the build directory
                    working_directory = (
                        root_working_dir / self.build_directory
                        if not Path(self.build_directory).is_absolute()
                        else self.build_directory
                    )
                elif itr == "install":
                    continue
                else:
                    raise argparse.ArgumentError(
                        None,
                        message=f"Invalid build tree specified: {itr}. Must be 'source' or 'build'.",
                    )
            elif self.mode == "install":
                if itr == "source" or itr == "build":
                    continue
                elif itr == "install":
                    pass
                else:
                    raise argparse.ArgumentError(
                        None,
                        message=f"Invalid build tree specified: {itr}. Must be 'source' or 'build'.",
                    )
            else:
                raise argparse.ArgumentError(
                    None,
                    message=f"Invalid mode specified: {self.mode}. Must be 'build' or 'install'.",
                )

            _require_working_directory_exists = _spec.get(
                "require-working-directory-exists", False
            )

            if _require_working_directory_exists and not os.path.exists(
                working_directory
            ):
                raise RuntimeError(
                    f"Working directory for {itr} does not exist: {working_directory}"
                )

            # read the VERSION file from the reference tree
            if self.version is None:
                _version_file = _spec["version-file"]
                self.version_file = (
                    Path(working_directory) / _version_file
                    if not Path(_version_file).is_absolute()
                    else Path(_version_file)
                )
                self.version = parse_version_file(self.version_file)

            setattr(self, f"{itr}_working_directory", working_directory)
            setattr(
                self,
                f"{itr}_sources",
                _get_file_set(
                    _spec,
                    working_directory,
                    "sources",
                ),
            )
            setattr(
                self,
                f"{itr}_headers",
                _get_file_set(
                    _spec,
                    working_directory,
                    "headers",
                ),
            )
            setattr(
                self,
                f"{itr}_abi_check",
                _get_file_set(
                    _spec,
                    working_directory,
                    "abi-check",
                ),
            )

            # create a temporary file if the working directory has git submodules
            if itr == "source":
                run_cmd(
                    ["git", "submodule", "update", "--init", working_directory],
                )

                submod_rc, submod_out, submod_err = run_cmd(
                    ["git", "submodule", "status", working_directory],
                )

                if submod_rc == 0 and submod_out.strip():
                    submod_out = "\n".join(
                        [
                            "  {}".format(itr.strip())
                            for itr in submod_out.strip().split("\n")
                        ]
                    )
                    logging.info(
                        f"Source tree '{working_directory}' has git submodule(s):\n{submod_out}"
                    )
                    tmpf = tempfile.NamedTemporaryFile(
                        prefix=f"rocm-abi-guard-submodules-{self.name}.", delete=False
                    )
                    with open(tmpf.name, "w") as f:
                        f.write(f"{submod_out}\n")
                    setattr(self, "submodule_file", tmpf.name)
                    setattr(
                        self,
                        f"{itr}_sources",
                        getattr(self, f"{itr}_sources")
                        + FileSet([tmpf.name], [], [], []),
                    )

        setattr(self, "headers", FileSet([], [], [], []))
        setattr(self, "sources", FileSet([], [], [], []))
        setattr(self, "abi_check", FileSet([], [], [], []))

        for aitr in ["headers", "sources", "abi_check"]:
            for titr in ["source", "build", "install"]:
                # if {source,build,install}_{headers,sources,abi_check} exists, add to {headers,sources,abi_check}
                if has_attr(self, f"{titr}_{aitr}"):
                    setattr(
                        self,
                        f"{aitr}",
                        getattr(self, f"{aitr}") + getattr(self, f"{titr}_{aitr}"),
                    )

    def __del__(self):
        if hasattr(self, "submodule_file"):
            try:
                logging.info(f"Removing temporary submodule file '{self.submodule_file}'")
                os.remove(self.submodule_file)
            except Exception as e:
                logging.error(
                    f"Failed to remove temporary submodule file '{self.submodule_file}': {e}"
                )

    def get(self, name, default=None):
        if not hasattr(self, name) and hasattr(self, name.replace("-", "_")):
            name = name.replace("-", "_")
        return getattr(self, name, default)


def has_attr(obj, name):
    return hasattr(obj, name) and getattr(obj, name) is not None


def strtobool(val):
    """Convert a string representation of truth to true or false.
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    if isinstance(val, (list, tuple)):
        if len(val) > 1:
            val_type = type(val).__name__
            raise ValueError(f"invalid truth value {val} (type={val_type})")
        else:
            val = val[0]

    if isinstance(val, bool):
        return val
    elif isinstance(val, str) and val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif isinstance(val, str) and val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        val_type = type(val).__name__
        raise ValueError(f"invalid truth value {val} (type={val_type})")


def get_compute_hash_text(data):

    def _get_readable_text(_text, nwidth=120, indent="\t", line_join="\n"):
        nwidth = 120
        return line_join.join(
            [
                "{}{}".format(indent, _text[i : i + nwidth])
                for i in range(0, len(_text), nwidth)
            ]
        )

    ret = []
    if isinstance(data, (list, tuple, set)):
        ret += [get_compute_hash_text(itr) for itr in data]
    elif isinstance(data, FileSet):
        ret += [get_compute_hash_text(itr) for itr in data.get_paths(absolute=True)]
    elif isinstance(data, Path):
        logging.info(f"Computing hash text for file: '{str(data)}'")
        lang = _language_for_path(data)
        text = data.read_text(encoding="utf-8", errors="ignore")
        pattern = COMMENT_PATTERNS.get(lang)
        if pattern:
            text = re.sub(pattern, "", text)
        # Remove all whitespace
        text = re.sub(r"\s+", "", text)
        logging.debug(
            "Hash text for file '{}':\n{}".format(str(data), _get_readable_text(text))
        )
        ret.append(text)
    elif isinstance(data, str):
        ret.append(data)
    else:
        raise TypeError(f"Unsupported data type for hash text extraction: {type(data)}")

    if not isinstance(data, Path):
        logging.debug(
            "Computed hash text data:\n{}".format(
                "\n".join([_get_readable_text(itr) for itr in ret])
            )
        )

    return ret


def compute_hash(data, *args) -> str:
    """
    Compute a hash for the given data.
    """
    data = get_compute_hash_text(data)
    if len(args) > 0:
        data += get_compute_hash_text([*args])

    _data = "".join([str(itr) for itr in data])
    return hashlib.md5(_data.encode()).hexdigest()


def _flush_streams():
    """
    Flush stdout and stderr streams.
    """
    sys.stdout.flush()
    sys.stderr.flush()


def _language_for_path(path: Path) -> str:
    """
    Determine the programming language for a given file path.
    """
    ext = path.suffix.lower()
    name = path.name
    if ext in {".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hh", ".hxx"}:
        return "c-like"
    if ext in {".py", ".pyi"}:
        return "python"
    if name == "CMakeLists.txt" or ext == ".cmake":
        return "cmake"
    return "c-like"


def run_cmd(
    cmd: List[str], cwd: str | None = None, check: bool = False
) -> Tuple[int, str, str]:
    """
    Run a command in a subprocess and return its exit code, stdout, and stderr.
    """

    _cmd = " ".join([f"{itr}" for itr in cmd])
    logging.info(f"Running command (cwd={str(cwd)}):\n\t'{_cmd}'")

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return proc.returncode, out, err


def parse_version_file(path: Path) -> Version:
    """
    Parse the VERSION file and return a Version object.
    """
    logging.info(f"Parsing VERSION file at: '{path.resolve()}'")
    if not path.is_file():
        sys.exit(f"Missing VERSION file at '{path.resolve()}'.")
    ver = path.read_text().strip()
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)[\.\-]*([a-zA-Z0-9]*)\n# hash: (\w+)", ver)
    if not m:
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", ver)
    if not m:
        sys.exit(f"VERSION must be X.Y.Z, got: {ver}")
    return Version(*m.groups())


def map_by_basename(paths: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Map file basenames to their full paths.
    """
    out = {}
    for p in paths:
        base = os.path.basename(p)
        out[base] = Path(p).resolve()

    # sort dictionary by keys
    out = dict(sorted(out.items()))

    keys = list(out.keys())
    del_keys = []
    dup = {}
    for i, itr in enumerate(keys):
        for j in range(i + 1, len(keys)):
            nitr = keys[j]
            if out[itr] == out[nitr]:
                logging.info(
                    f"Duplicate basename '{itr}' found for paths in '{nitr}':\n\t- {out[itr]}\n\t- {out[nitr]}"
                )
                if itr not in dup:
                    dup[itr] = []
                if nitr not in dup:
                    dup[itr].append(nitr)
                if nitr not in del_keys:
                    del_keys.append(nitr)

    for d in del_keys:
        logging.debug(f"Removing path for '{d}': {out[d]}")
        del out[d]

    return (out, dup)


def split_shell_args(argstr: str) -> List[str]:
    """
    Split a shell-style argument string into a list of arguments.
    """
    # Keep compatibility with typical input like "--headers-only --no-default-suppression"
    argstr = argstr.strip()
    return shlex.split(argstr) if argstr else []


def generate(args) -> str:
    """
    Generate the versioning configuration.
    """
    from jinja2 import Environment, FileSystemLoader

    # Set up Jinja2 environment to load templates from the current directory
    env = Environment(
        loader=FileSystemLoader(
            os.path.join(os.path.dirname(__file__), "templates"),
        ),
        lstrip_blocks=True,
        trim_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template("versioning.yml.j2")

    def patch_dirs_arg(val):
        return val if "*" in val else val + "/**"

    def patch_lib_arg(val):
        _lib_prefix = "lib"
        if val.startswith("lib"):
            _lib_prefix = ""
        return val if "*" in val else f"{_lib_prefix}{val}*.so*"

    _source_dirs = [patch_dirs_arg(d) for d in args.source_dirs]
    _include_dirs = [patch_dirs_arg(d) for d in args.include_dirs]
    _test_dirs = ["'**/test/**'", "'**/tests/**'"]
    _sample_dirs = ["'**/samples/**'"]
    _lib_names = [f"lib{args.project_name}*.so*"] + [
        patch_lib_arg(p) for p in args.library_names
    ]

    logging.warning(f"Library names: {_lib_names}")

    data = {
        "name": f"{args.project_name}",
        "build_directory": args.build_directory,
        "cmake_build_type": "RelWithDebInfo",
        "cmake_generator": "Ninja",
        "cmake_config_args": "",
        "source_tree": {
            "version_file": "VERSION",
            "working_directory": ".",
            "require_working_directory_exists": True,
            "sources": {
                "include": [],
                "exclude": [],
                "recursive_include": _source_dirs,
                "recursive_exclude": _test_dirs + _sample_dirs,
            },
            "headers": {
                "include": [],
                "exclude": [],
                "recursive_include": _include_dirs,
                "recursive_exclude": _test_dirs + _sample_dirs,
            },
            "abi_check": {
                "include": [],
                "exclude": [],
                "recursive_include": [],
                "recursive_exclude": [],
            },
        },
        "build_tree": {
            "version_file": "VERSION",
            "working_directory": ".",
            "require_working_directory_exists": False,
            "sources": {},
            "headers": {},
            "abi_check": {
                "include": [],
                "exclude": [],
                "recursive_include": _lib_names,
                "recursive_exclude": [],
            },
        },
        "install_tree": {
            "version_file": f"share/{args.project_name}/VERSION",
            "working_directory": "../..",
            "require_working_directory_exists": True,
            "sources": [],
            "headers": [
                f"include/{args.project_name}/**/*.h",
                f"include/{args.project_name}/**/*.hpp",
            ],
            "abi_check": {
                "include": [],
                "exclude": [],
                "recursive_include": _lib_names,
                "recursive_exclude": [],
            },
        },
    }

    # Render the template with the loaded data
    rendered_config = template.render(**data)

    # Print the generated YAML configuration
    if not args.quiet:
        print(rendered_config)

    output_file = os.path.join(args.output_directory, args.output_file)
    output_dir = os.path.dirname(output_file)

    logging.warning(f"Generating '{output_file}'...")

    # Make the directory for output file if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Optionally, save the rendered configuration to a new YAML file
    with open(output_file, "w") as f:
        f.write(rendered_config)


def main() -> None:
    """
    Main entry point for the ABI Guard script.
    """

    def add_parser_bool_argument(_parser, *args, **kwargs):
        _parser.add_argument(
            *args,
            **kwargs,
            action=BooleanArgAction,
            nargs="?",
            const=True,
            type=str,
            required=False,
            metavar="BOOL",
        )

    parser = argparse.ArgumentParser(
        description="ABI Guard using libabigail (abidw/abidiff)"
    )

    logging_choices = {
        level.lower(): getattr(logging, level)
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    }

    parser.add_argument(
        "--log-level",
        default=os.environ.get("ROCM_ABI_GUARD_LOG_LEVEL", "warning"),
        choices=list(logging_choices.keys()),
        type=str.lower,
        help="Set the logging level.",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Set the log file.",
        type=str,
    )

    def _add_head_spec_arg(_parser: argparse.ArgumentParser) -> None:
        _parser.add_argument(
            "-i",
            "--head-spec",
            "--input-spec",
            help="Path to versioning.yml file.",
            default=None,
            type=str,
            required=True,
        )

        _parser.add_argument(
            "-m",
            "--mode",
            help="Select build mode to use. build means using source tree and build tree, install means using install tree.",
            choices=["build", "install"],
            default="build",
        )

        _parser.add_argument(
            "--build-directory",
            help="Select/specify build directory (overrides versioning.yml).",
            type=str,
            default="build",
        )

    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser(
        "query",
        help="Read in versioning.yml configuration file and output information",
        add_help=True,
        allow_abbrev=False,
    )

    _add_head_spec_arg(query_parser)

    query_parser.add_argument(
        "field",
        help="Field to query from versioning spec.",
        type=str,
        default=None,
        nargs="?",
    )

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a versioning.yml configuration file from template",
        add_help=True,
        allow_abbrev=False,
    )

    generate_parser.add_argument(
        "-n",
        "--project-name",
        help="Name of the project/package.",
        type=str,
        required=True,
    )

    generate_parser.add_argument(
        "--include-dirs",
        help="Directories containing public API headers.",
        type=str,
        default=[],
        nargs="+",
    )

    generate_parser.add_argument(
        "--source-dirs",
        help="Directories containing source (implementation) files.",
        type=str,
        default=[],
        nargs="+",
    )

    generate_parser.add_argument(
        "--library-names",
        help="Names of the libraries for ABI checking.",
        type=str,
        default=[],
        nargs="+",
    )

    version_parser = subparsers.add_parser(
        "version",
        help="Manipulate VERSION files",
        add_help=True,
        allow_abbrev=False,
    )

    _add_head_spec_arg(version_parser)

    add_parser_bool_argument(
        version_parser,
        "--echo",
        help="Echo current version without modifying.",
        default=False,
    )
    add_parser_bool_argument(
        version_parser,
        "--bump-major",
        help="Bump major version.",
        default=False,
    )
    add_parser_bool_argument(
        version_parser,
        "--bump-minor",
        help="Bump minor version.",
        default=False,
    )
    add_parser_bool_argument(
        version_parser,
        "--bump-patch",
        help="Bump patch version.",
        default=False,
    )
    version_parser.add_argument(
        "--set",
        help="Set version number.",
        type=str,
        default=None,
    )
    version_parser.add_argument(
        "--set-major",
        help="Set major version.",
        type=int,
        default=None,
    )
    version_parser.add_argument(
        "--set-minor",
        help="Set minor version.",
        type=int,
        default=None,
    )
    version_parser.add_argument(
        "--set-patch",
        help="Set patch version.",
        type=int,
        default=None,
    )
    version_parser.add_argument(
        "--set-build",
        help="Set build version.",
        type=int,
        default=None,
    )

    hash_parser = subparsers.add_parser(
        "hash",
        help="Compute hash of source files",
        add_help=True,
        allow_abbrev=False,
    )

    _add_head_spec_arg(hash_parser)

    generate_parser.add_argument(
        "-d",
        "--output-directory",
        help="Path to output directory.",
        default=os.getcwd(),
    )
    generate_parser.add_argument(
        "-o",
        "--output-file",
        help="Name of the output file.",
        default="versioning.yaml",
    )
    generate_parser.add_argument(
        "-q",
        "--quiet",
        help="Suppress printing configuration to stdout.",
        action="store_true",
    )

    abi_generate_parser = subparsers.add_parser(
        "abi-generate", help="Generate a ABI XML using libabigail (abidw)"
    )

    _add_head_spec_arg(abi_generate_parser)

    abi_generate_parser.add_argument(
        "--abidw-args", default="", help="Extra args for abidw."
    )
    abi_generate_parser.add_argument(
        "-d",
        "--output-directory",
        default=os.path.join(os.getcwd(), "abi-guard"),
        help="Directory to write abidw reports.",
    )

    check_parser = subparsers.add_parser(
        "check", help="ABI Guard using libabigail (abidw/abidiff)"
    )

    _add_head_spec_arg(check_parser)

    check_parser.add_argument(
        "-b",
        "--base-spec",
        help="Path to (base) versioning.yml file.",
        required=True,
    )
    check_parser.add_argument("--abidw-args", default="", help="Extra args for abidw.")
    check_parser.add_argument(
        "--abidiff-args", default="", help="Extra args for abidiff."
    )
    check_parser.add_argument(
        "-d",
        "--output-directory",
        default=os.path.join(os.getcwd(), "abi-guard"),
        help="Directory to write abidw/abidiff reports and logs.",
    )
    check_parser.add_argument(
        "--head-abi-xml",
        help="Path to folder containing the (head) ABI XML files (if generated previously via abi-generate).",
        required=False,
        type=str,
        default=None,
    )
    check_parser.add_argument(
        "--base-abi-xml",
        help="Path to folder containing the (base) ABI XML files (if generated previously via abi-generate).",
        required=False,
        type=str,
        default=None,
    )

    args = parser.parse_args()

    for _attr in ["head_abi_xml", "base_abi_xml"]:
        if has_attr(args, _attr):
            _path = Path(getattr(args, _attr))
            if not _path.is_dir():
                _option = f"--{_attr.replace('_', '-')}"
                raise argparse.ArgumentError(
                    None,
                    message=f"Specified {_option} option is not a directory: '{str(_path)}'",
                )

    logging.basicConfig(
        level=logging_choices.get(args.log_level, logging.WARNING),
        format="[%(levelname)s] %(message)s",
        filename=args.log_file,
    )

    if hasattr(args, "head_spec") and args.head_spec:
        head_spec = VersioningSpec(args.head_spec, args)

    if hasattr(args, "base_spec") and args.base_spec:
        base_spec = VersioningSpec(args.base_spec, args)

    if args.command == "generate":
        generate(args)

    elif args.command == "query":

        def print_dict(obj, prefix=""):
            for key, itr in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                if hasattr(itr, "__dict__"):
                    print_dict(itr, f"{prefix}{key}.")
                else:
                    print(f"- {prefix}{key}")

        if args.field is None or args.field.lower() == "all":
            print_dict(head_spec)

        else:
            val = ""
            obj = head_spec
            for itr in args.field.split("."):
                _ret = obj.get(itr, None)
                if _ret is not None:
                    if isinstance(_ret, (Version, FileSet)):
                        obj = _ret
                    else:
                        val = _ret
                else:
                    break

            if isinstance(val, list):
                val = ", ".join([str(itr) for itr in val])

            print(f"{val}")

    elif args.command == "version":
        if args.echo:
            print(f"{head_spec.version_file}: {head_spec.version}")
        else:
            logging.info(
                f"Updating VERSION file at {head_spec.version_file}: {head_spec.version}"
            )

            if args.bump_major:
                head_spec.version += Version(1, 0, 0)
                head_spec.version.minor = 0
                head_spec.version.patch = 0
            if args.bump_minor:
                head_spec.version += Version(0, 1, 0)
                head_spec.version.patch = 0
            if args.bump_patch:
                head_spec.version += Version(0, 0, 1)

            if args.set:
                m = re.fullmatch(
                    r"(\d+)\.(\d+)\.(\d+)[\.\-]*([a-zA-Z0-9]*)", args.set.strip()
                )
                if not m:
                    raise argparse.ArgumentError(
                        None, message=f"VERSION must be X.Y.Z, got: {args.set}"
                    )
                head_spec.version = Version(*m.groups())
            if args.set_major:
                head_spec.version.major = args.set_major
            if args.set_minor:
                head_spec.version.minor = args.set_minor
            if args.set_patch:
                head_spec.version.patch = args.set_patch
            if args.set_build:
                head_spec.version.build = args.set_build

            digest = compute_hash(head_spec.sources + head_spec.headers)
            head_spec.version.write(head_spec.version_file, digest)
            logging.warning(
                f"Updated VERSION file at {head_spec.version_file}: {head_spec.version}"
            )

    elif args.command == "hash":

        digest_files = head_spec.sources + head_spec.headers
        digest = compute_hash(digest_files)
        print(f"{digest}")
        files = "\n".join(
            sorted([f"    - {itr}" for itr in digest_files.get_paths(absolute=False)])
        )
        logging.info(f"hashed files:\n{files}")

    elif args.command == "abi-generate":
        head_version = head_spec.version

        logging.warning(f"Current  VERSION: {head_version}")

        # Collect libraries
        head_libs = head_spec.abi_check

        logging.info(f"Current ABI libs: {head_libs.get_paths(absolute=False)}")

        head_by_name, dup_head_by_name = map_by_basename(
            head_libs.get_paths(absolute=False)
        )

        abi_libs = sorted(set(head_by_name.keys()))
        if not abi_libs:
            sys.exit("No ABI libraries found")

        logging.warning(f"ABI libraries: {abi_libs}")

        # Prepare dirs
        out_dir = Path(args.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        for ditr in [out_dir]:
            with open(ditr / ".gitignore", "w") as ofs:
                ofs.write("*\n")

        abidw_extras = split_shell_args(head_spec.abidw_args) + GLOBAL_ABIDW_EXTRAS

        for name in abi_libs:
            logging.warning(f"Processing library: {name}")
            _flush_streams()

            head_so = head_by_name[name]
            head_xml = out_dir / f"{head_spec.name}.{name}.abi"

            head_abidw_header_files = split_shell_args(
                " ".join(
                    [
                        f"--header-file {str(p)}"
                        for p in head_spec.headers.get_paths(absolute=True)
                    ]
                )
            )

            # Dump ABI XML with abidw
            head_rc, head_out, head_err = run_cmd(
                ["abidw", *abidw_extras, *head_abidw_header_files, head_so]
            )

            if head_rc == 0:
                logging.warning(f"Writing ABI XML: '{str(head_xml)}'...")
                head_xml.write_text(f"{head_out}\n")
            else:
                Path(out_dir / f"{head_spec.name}.{name}.abidw.stderr.txt").write_text(
                    head_err
                )
                logging.warning(
                    f"::warning title=abidw head::{name}: abidw returned {head_rc}\nstderr:\n{head_err}"
                )
                sys.exit(head_rc)

            if name in dup_head_by_name:
                for itr in dup_head_by_name[name]:
                    copy_xml = out_dir / f"{head_spec.name}.{itr}.abi"
                    logging.warning(
                        f"Copying ABI XML: '{str(head_xml)}' to '{str(copy_xml)}'..."
                    )
                    shutil.copy2(str(head_xml), str(copy_xml))

    elif args.command == "check":
        head_version = head_spec.version
        base_version = base_spec.version

        logging.warning(f"Baseline VERSION: {base_version}")
        logging.warning(f"Current  VERSION: {head_version}")

        # Collect libraries
        base_libs = base_spec.abi_check
        head_libs = head_spec.abi_check

        logging.info(f"Baseline ABI libs: {base_libs.get_paths(absolute=False)}")
        logging.info(f"Current  ABI libs: {head_libs.get_paths(absolute=False)}")

        base_by_name, _ = map_by_basename(base_libs.get_paths(absolute=False))
        head_by_name, _ = map_by_basename(head_libs.get_paths(absolute=False))

        common_libs = sorted(set(base_by_name.keys()) & set(head_by_name.keys()))
        if not common_libs:
            sys.exit(
                f"No common library basenames found between baseline and head artifacts.\nbase: {base_by_name}\nhead: {head_by_name}"
            )

        base_lib_patterns = [
            f".{base_version.major}.{base_version.minor}.{base_version.patch}",
            f".{base_version.major}.{base_version.minor}",
            f".{base_version.major}",
        ]
        head_lib_patterns = [
            f".{head_version.major}.{head_version.minor}.{head_version.patch}",
            f".{head_version.major}.{head_version.minor}",
            f".{head_version.major}",
        ]

        del_base = []
        del_head = []
        for bitr, bpath in base_by_name.items():
            for bpattern, hpattern in zip(base_lib_patterns, head_lib_patterns):
                if bpattern == hpattern:
                    continue
                if bitr.endswith(bpattern):
                    hitr = bitr.replace(bpattern, hpattern)
                    if hitr in head_by_name:
                        logging.warning(f"Removing versioned '{bitr}' and '{hitr}'")
                        del_head.append(hitr)
                        del_base.append(bitr)

        for d in del_base:
            del base_by_name[d]

        for d in del_head:
            del head_by_name[d]

        added_libs = sorted(set(head_by_name.keys()) - set(base_by_name.keys()))
        removed_libs = sorted(set(base_by_name.keys()) - set(head_by_name.keys()))

        logging.warning(f" Common libraries: {common_libs}")
        logging.warning(f"  Added libraries: {added_libs}")
        logging.warning(f"Removed libraries: {removed_libs}")

        # Prepare dirs
        out_dir = Path(args.output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        for ditr in [out_dir]:
            with open(ditr / ".gitignore", "w") as ofs:
                ofs.write("*\n")

        abidw_extras = split_shell_args(args.abidw_args) + GLOBAL_ABIDW_EXTRAS
        abidiff_extras = split_shell_args(args.abidiff_args) + GLOBAL_ABIDIFF_EXTRAS

        incompatible = 0
        added_any = False
        changed_any = False
        deleted_any = False

        for name in common_libs:
            logging.warning(f"Processing library: {name}")
            _flush_streams()

            base_so = base_by_name[name]
            head_so = head_by_name[name]

            def _get_abi_xml(_label, _spec, _abi_xml_dir, _so):
                if _abi_xml_dir:
                    assert os.path.isdir(
                        _abi_xml_dir
                    ), f"Specified {_label} ABI XML path is not a directory: '{_abi_xml_dir}'"
                    _abi_xml = (
                        Path(_abi_xml_dir) / f"{_spec.name}.{name}.abi"
                        if _abi_xml_dir
                        else None
                    )
                    assert os.path.exists(
                        _abi_xml
                    ), f"Specified {_label} ABI XML file does not exist: '{str(_abi_xml)}'"
                    return Path(_abi_xml)
                else:
                    _xml = out_dir / f"{base_spec.name}.{name}.{_label}.abi"
                    _abidw_header_files = split_shell_args(
                        " ".join(
                            [
                                f"--header-file {str(p)}"
                                for p in _spec.headers.get_paths(absolute=True)
                            ]
                        )
                    )
                    # Dump ABI XML with abidw
                    _rc, _out, _err = run_cmd(
                        ["abidw", *abidw_extras, *_abidw_header_files, _so]
                    )
                    _xml.write_text(f"{_out}\n")
                    if _rc != 0:
                        Path(
                            out_dir / f"{_spec.name}.{name}.{_label}.abidw.stderr.txt"
                        ).write_text(_err)
                        logging.warning(
                            f"::warning title=abidw baseline::{name}: abidw returned {_rc}\nstderr:\n{_err}"
                        )
                    if _rc != 0:
                        logging.critical(
                            f"::error title=ABI / Version policy:: abidw ({_label}) failed."
                        )
                        sys.exit(1)

                    return Path(_xml)

            base_xml = _get_abi_xml("baseline", base_spec, args.base_abi_xml, base_so)
            head_xml = _get_abi_xml("head", head_spec, args.head_abi_xml, head_so)

            # Compare with abidiff
            def _run_abidiff(option):
                global incompatible

                cmd = [
                    "abidiff",
                    *abidiff_extras,
                    f"--{option}",
                    str(base_xml),
                    str(head_xml),
                ]
                rc, out, err = run_cmd(cmd)
                report_path = out_dir / f"{name}.{option}.txt"
                report_path.write_text(
                    f"ExitCode={rc}\n\nstderr:\n{err}\n\nstdout:\n{out}\n"
                )
                print(f"[===== abidiff {option} {name} (exit code: {rc}) =====]")
                print(out)

                # Exit code bit 8 => incompatible changes
                if rc & 8:
                    incompatible = 1

                return 1 if rc != 0 else 0

            added_funcs = _run_abidiff("added-fns")
            added_vars = _run_abidiff("added-vars")
            changed_funcs = _run_abidiff("changed-fns")
            changed_vars = _run_abidiff("changed-vars")
            deleted_funcs = _run_abidiff("deleted-fns")
            deleted_vars = _run_abidiff("deleted-vars")

            if added_funcs + added_vars > 0:
                added_any = True
            if changed_funcs + changed_vars > 0:
                changed_any = True
            if deleted_funcs + deleted_vars > 0:
                deleted_any = True

            _flush_streams()

        # Enforce semver policy
        fail_reason = ""
        if incompatible:
            if not (head_version.major > base_version.major):
                fail_reason = (
                    f"ABI break detected, but major VERSION not incremented "
                    f"(prev={base_version}, "
                    f"curr={head_version})."
                )
        else:
            if added_any or changed_any or deleted_any or added_libs or removed_libs:
                if not (
                    head_version.major > base_version.major
                    or (
                        head_version.major == base_version.major
                        and head_version.minor > base_version.minor
                    )
                ):
                    fail_reason = (
                        f"Public API additions/compatible changes detected, but minor VERSION not incremented "
                        f"(prev={base_version}, "
                        f"curr={head_version})."
                    )
            else:
                if head_version != base_version:
                    fail_reason = (
                        f"No public ABI change, but major/minor changed "
                        f"(prev={base_version}, "
                        f"curr={head_version})."
                    )
                elif head_version == base_version:
                    base_digest = compute_hash(base_spec.sources + base_spec.headers)
                    head_digest = compute_hash(head_spec.sources + head_spec.headers)

                    if (
                        base_digest != head_digest
                        and head_version.patch == base_version.patch
                    ):
                        fail_reason = f"Source/header files changed (prev={base_digest}, curr={head_digest}), but patch VERSION not incremented (prev={base_version}, curr={head_version})."

        if fail_reason:
            logging.critical(f"::error title=ABI / Version policy::{fail_reason}")
            sys.exit(1)


if __name__ == "__main__":
    main()
