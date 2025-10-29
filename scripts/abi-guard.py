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
import hashlib
import logging
import argparse
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

    def __init__(self, head_spec, args, **kwargs) -> None:

        def has_attr(obj, name):
            return hasattr(obj, name) and getattr(obj, name) is not None

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

        root_working_dir = Path(head_spec).parent
        with open(head_spec, "r") as ifs:
            self.spec = yaml.safe_load(ifs)

        self.name = self.spec["versioning"].get("name", None)
        self.tree = None
        self.build_directory = self.spec["versioning"].get("build-directory", "build")
        self.install_directory = self.spec["versioning"].get(
            "install-directory", "install"
        )
        self.abidw_args = self.spec["versioning"].get("abidw-args", "")
        self.abidiff_args = self.spec["versioning"].get("abidiff-args", "")

        # command line overrides
        if has_attr(args, "build_directory"):
            self.build_directory = args.build_directory
        if has_attr(args, "install_directory"):
            self.install_directory = args.install_directory
        if has_attr(args, "abidw_args"):
            self.abidw_args = args.abidw_args
        if has_attr(args, "abidiff_args"):
            self.abidiff_args = args.abidiff_args
        if has_attr(args, "tree"):
            self.tree = args.tree

        self.build_directory = kwargs.get("build_directory", self.build_directory)
        self.install_directory = kwargs.get("install_directory", self.install_directory)
        self.abidw_args = kwargs.get("abidw_args", self.abidw_args)
        self.abidiff_args = kwargs.get("abidiff_args", self.abidiff_args)
        self.tree = kwargs.get("tree", self.tree)

        if self.tree is None:
            raise argparse.ArgumentError(None, message="-t / --tree must be specified.")

        if self.tree is None:
            self.tree = "source-tree"

        for itr, attrib in zip(
            ["source-tree", "build-tree", "install-tree"], ["source", "build", "install"]
        ):
            if self.tree == "source-tree" and itr == "install-tree":
                # if the spec is from the source tree, ignore the install-tree
                continue
            elif self.tree == "install-tree" and itr in ["source-tree", "build-tree"]:
                # if the spec is from the install tree, ignore the source and build trees
                continue

            working_directory = self.spec["versioning"][itr].get(
                "working-directory", None
            )

            if working_directory is not None:
                working_directory = (
                    Path(working_directory)
                    if Path(working_directory).is_absolute()
                    else Path(root_working_dir / working_directory).resolve()
                )
            else:
                working_directory = root_working_dir

            if self.tree != "build-tree" and itr == "build-tree":
                # if the spec is not from the build tree, configure build tree working directory
                working_directory = (
                    root_working_dir / self.build_directory
                    if not Path(self.build_directory).is_absolute()
                    else self.build_directory
                )
            elif self.tree != "install-tree" and itr == "install-tree":
                # if the spec is not from the install tree, configure install tree working directory
                working_directory = (
                    root_working_dir / self.install_directory
                    if not Path(self.install_directory).is_absolute()
                    else self.install_directory
                )

            # read the VERSION file from the reference tree
            if itr == self.tree:
                _version_file = self.spec["versioning"][itr]["version-file"]
                self.version_file = (
                    Path(working_directory) / _version_file
                    if not Path(_version_file).is_absolute()
                    else Path(_version_file)
                )
                self.version = parse_version_file(self.version_file)

            if not os.path.exists(working_directory):
                raise RuntimeError(
                    f"Working directory for {itr} does not exist: {working_directory}"
                )

            setattr(self, f"{attrib}_working_directory", working_directory)
            setattr(
                self,
                f"{attrib}_sources",
                _get_file_set(
                    self.spec["versioning"][itr],
                    working_directory,
                    "sources",
                ),
            )
            setattr(
                self,
                f"{attrib}_headers",
                _get_file_set(
                    self.spec["versioning"][itr],
                    working_directory,
                    "headers",
                ),
            )
            setattr(
                self,
                f"{attrib}_abi_check",
                _get_file_set(
                    self.spec["versioning"][itr],
                    working_directory,
                    "abi-check",
                ),
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

    def get(self, name, default=None):
        if not hasattr(self, name) and hasattr(self, name.replace("-", "_")):
            name = name.replace("-", "_")
        return getattr(self, name, default)


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


def compute_hash(data) -> str:
    """
    Compute a hash for the given data.
    """
    if isinstance(data, FileSet):
        combined = {}
        for path in data.get_paths():
            lang = _language_for_path(path)
            text = path.read_text(encoding="utf-8", errors="ignore")
            pattern = COMMENT_PATTERNS.get(lang)
            if pattern:
                text = re.sub(pattern, "", text)
            # Remove all whitespace
            text = re.sub(r"\s+", "", text)
            combined[f"{path}"] = text

        # Compute MD5
        return compute_hash(f"{combined}") if combined else "0"
    else:
        _data = f"{data}" if not isinstance(data, str) else data
        return hashlib.md5(_data.encode()).hexdigest()


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

    _cmd = "\n\t  ".join([f"{itr}" for itr in cmd])
    logging.debug(f"Running command (cwd={str(cwd)}):\n\t'{_cmd}'")

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
    if not path.is_file():
        sys.exit(f"Missing VERSION file at {path} (expected X.Y.Z).")
    ver = path.read_text().strip()
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)[\.\-]*([a-zA-Z0-9]*)\n# hash: (\w+)", ver)
    if not m:
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", ver)
    if not m:
        sys.exit(f"VERSION must be X.Y.Z, got: {ver}")
    return Version(*m.groups())


def map_by_basename(paths: List[str]) -> Dict[str, str]:
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
    for i, itr in enumerate(keys):
        for j in range(i + 1, len(keys)):
            nitr = keys[j]
            if out[itr] == out[nitr]:
                logging.info(
                    f"Duplicate basename '{itr}' found for paths in '{nitr}':\n\t- {out[itr]}\n\t- {out[nitr]}"
                )
                if nitr not in del_keys:
                    del_keys.append(nitr)

    for d in del_keys:
        logging.debug(f"Removing path for '{d}': {out[d]}")
        del out[d]

    return out


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
        "build_directory": "build",
        "install_directory": "install",
        "cmake_build_type": "RelWithDebInfo",
        "cmake_generator": "Ninja",
        "cmake_config_args": "",
        "source_tree": {
            "version_file": "VERSION",
            "working_directory": ".",
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

    output_file = os.path.join(args.output_path, args.output_file)
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

    logging_choices = dict(
        [level.lower(), getattr(logging, level)]
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    parser.add_argument(
        "--log-level",
        default="warning",
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
            "-t",
            "--tree",
            help="Select tree to operate on (source-tree, build-tree, install-tree).",
            choices=["source-tree", "build-tree", "install-tree"],
            default="source-tree",
        )

        _parser.add_argument(
            "-p",
            "--package",
            help="Select package to operate on (from versioning.yml).",
            type=str,
            default=None,
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
        "--output-path",
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
        "--baseline-tag-pattern",
        default="v[0-9]*.[0-9]*.[0-9]*",
        help="Tag pattern to infer baseline version if VERSION missing in baseline artifact.",
    )
    check_parser.add_argument(
        "--report-dir",
        default="abi-guard/diff-reports",
        help="Directory to write abidiff reports.",
    )
    check_parser.add_argument(
        "--abi-base-dir",
        default="abi-guard/abi-base",
        help="Output dir for baseline ABI XML.",
    )
    check_parser.add_argument(
        "--abi-head-dir",
        default="abi-guard/abi-head",
        help="Output dir for head ABI XML.",
    )

    args = parser.parse_args()

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
        print(f"  {args.package} {args.tree} hash: {digest}")
        files = "\n".join(
            sorted([f"    - {itr}" for itr in digest_files.get_paths(absolute=False)])
        )
        logging.info(f"  {args.package} {args.tree} files:\n{files}")

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

        base_by_name = map_by_basename(base_libs.get_paths(absolute=False))
        head_by_name = map_by_basename(head_libs.get_paths(absolute=False))

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
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        abi_base = Path(args.abi_base_dir)
        abi_base.mkdir(parents=True, exist_ok=True)
        abi_head = Path(args.abi_head_dir)
        abi_head.mkdir(parents=True, exist_ok=True)

        for ditr in [report_dir, abi_base, abi_head]:
            with open(ditr / ".gitignore", "w") as ofs:
                ofs.write("*\n")

        abidw_extras = split_shell_args(args.abidw_args)
        abidiff_extras = split_shell_args(args.abidiff_args)

        incompatible = 0
        added_any = False
        changed_any = False
        deleted_any = False

        for name in common_libs:
            base_so = base_by_name[name]
            head_so = head_by_name[name]
            old_xml = abi_base / f"{name}.abi"
            new_xml = abi_head / f"{name}.abi"

            base_abidw_header_files = split_shell_args(
                " ".join(
                    [
                        f"--header-file {str(p)}"
                        for p in base_spec.headers.get_paths(absolute=True)
                    ]
                )
            )
            head_abidw_header_files = split_shell_args(
                " ".join(
                    [
                        f"--header-file {str(p)}"
                        for p in head_spec.headers.get_paths(absolute=True)
                    ]
                )
            )

            # Dump ABI XML with abidw
            base_rc, base_out, base_err = run_cmd(
                ["abidw", *abidw_extras, *base_abidw_header_files, base_so]
            )
            old_xml.write_text(base_out)
            if base_rc != 0:
                Path(report_dir / f"{name}.baseline_abidw.stderr.txt").write_text(
                    base_err
                )
                logging.warning(
                    f"::warning title=abidw baseline::{name}: abidw returned {base_rc}\nstderr:\n{base_err}"
                )

            head_rc, head_out, head_err = run_cmd(
                ["abidw", *abidw_extras, *head_abidw_header_files, head_so]
            )
            new_xml.write_text(head_out)
            if head_rc != 0:
                Path(report_dir / f"{name}.head_abidw.stderr.txt").write_text(head_err)
                logging.warning(
                    f"::warning title=abidw head::{name}: abidw returned {head_rc}\nstderr:\n{head_err}"
                )

            if base_rc != 0 or head_rc != 0:
                logging.critical("::error title=ABI / Version policy:: abidw failed.")
                sys.exit(1)

            incompatible = 0

            # Compare with abidiff
            def run_abidiff(option):
                global incompatible

                cmd = [
                    "abidiff",
                    *abidiff_extras,
                    f"--{option}",
                    str(old_xml),
                    str(new_xml),
                ]
                rc, out, err = run_cmd(cmd)
                report_path = report_dir / f"{name}.{option}.txt"
                report_path.write_text(
                    f"ExitCode={rc}\n\nstderr:\n{err}\n\nstdout:\n{out}\n"
                )
                print(f"[===== abidiff {option} {name} (exit code: {rc}) =====]")
                print(out)

                # Exit code bit 8 => incompatible changes
                if rc & 8:
                    incompatible = 1

                return 1 if rc != 0 else 0

            added_funcs = run_abidiff("added-fns")
            added_vars = run_abidiff("added-vars")
            changed_funcs = run_abidiff("changed-fns")
            changed_vars = run_abidiff("changed-vars")
            deleted_funcs = run_abidiff("deleted-fns")
            deleted_vars = run_abidiff("deleted-vars")

            if added_funcs + added_vars > 0:
                added_any = True
            if changed_funcs + changed_vars > 0:
                changed_any = True
            if deleted_funcs + deleted_vars > 0:
                deleted_any = True

        # Enforce semver policy
        fail_reason = ""
        if incompatible:
            if not (head_version.major > base_version.major):
                fail_reason = (
                    f"ABI break detected, but VERSION major not incremented "
                    f"(prev={base_version}, "
                    f"curr={head_version})."
                )
        else:
            if added_any or changed_any or deleted_any:
                if not (
                    head_version.major > base_version.major
                    or (
                        head_version.major == base_version.major
                        and head_version.minor > base_version.minor
                    )
                ):
                    fail_reason = (
                        f"Public API additions/compatible changes detected, but VERSION minor/major not incremented "
                        f"(prev={base_version}, "
                        f"curr={head_version})."
                    )
            else:
                if not (head_version == base_version):
                    fail_reason = (
                        f"No public ABI change, but major/minor changed "
                        f"(prev={base_version}, "
                        f"curr={head_version})."
                    )

        if fail_reason:
            logging.critical(f"::error title=ABI / Version policy::{fail_reason}")
            sys.exit(1)


if __name__ == "__main__":
    main()
