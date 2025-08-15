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
import shutil
import datetime
import yaml
import argparse
from . import output_config

rocpd_package_version = "1.0"
rocpd_metadata_param_version = "rocpd_package_version"

IDEAL_NUMBER_OF_DATABASE_FILES = 5


def flatten_rocpd_yaml_input_file(input) -> list:
    """
    Processes a YAML file containing rocprofiler-sdk/rocpd metadata and returns a list of database files.
    Also expands wildcards in both YAML 'files' and direct input parameters.
    Supports .rpdb folders containing index.yaml.

    Args:
        input (list of str): List of input file paths (YAML, DB, or .rpdb folder).

    Returns:
        list: List of database file paths.
    """
    import glob

    def parse_yaml_file(yaml_path, base_dir=None):
        """Parse a rocprofiler-sdk YAML file and expand wildcards."""
        with open(yaml_path, "r") as f:
            meta = yaml.safe_load(f)
            rocpd_meta = meta.get("rocprofiler-sdk", {}).get("rocpd", {})
            version = rocpd_meta.get(rocpd_metadata_param_version, "0.1")
            if version < rocpd_package_version:
                print(
                    f"Warning: {yaml_path} is using an outdated version of rocpd package ({version})."
                )
            cwd = rocpd_meta.get("path", os.getcwd())
            # If base_dir is provided (e.g., for .rpdb), override cwd
            if base_dir is not None:
                cwd = base_dir
            dbs = rocpd_meta.get("files", [])
            if isinstance(dbs, str):
                dbs = [dbs]
            files = []
            for db in dbs:
                db_path = os.path.join(cwd, db) if not os.path.isabs(db) else db
                if "*" in db_path or "?" in db_path or "[" in db_path:
                    files.extend(glob.glob(db_path))
                else:
                    files.append(db_path)
            return files

    input_files = []
    for item in input:
        # Handle .rpdb folder: look for index.yaml inside and flatten it
        if item.endswith(".rpdb") and os.path.isdir(item):
            index_yaml = os.path.join(item, "index.yaml")
            if os.path.isfile(index_yaml):
                input_files.extend(
                    parse_yaml_file(index_yaml, base_dir=os.path.abspath(item))
                )
            else:
                # If no index.yaml, treat as a directory and look for *.db files
                input_files.extend(glob.glob(os.path.join(item, "*.db")))
        elif item.endswith((".yaml", ".yml")):
            input_files.extend(parse_yaml_file(item))
        else:
            # Expand wildcards in direct input parameters as well
            if "*" in item or "?" in item or "[" in item:
                input_files.extend(glob.glob(item))
            else:
                input_files.append(item)

    num_dbs = len(input_files)
    print(f"Found {num_dbs} database files.")

    return input_files


def create_metadata_file(
    db_files, output_path=".", metadata_filename="index.yaml", consolidate=False
):
    """
    Creates a metadata file in a custom YAML format for rocprofiler-sdk/rocpd.

    Args:
        db_files (list of str): List of absolute or relative paths to SQL database files.
        output_path (str): Directory to write the metadata file.
        metadata_filename (str): Name of the metadata file to create.

    Returns:
        str: Path to the created metadata file.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Compute relative paths
    rel_paths = [os.path.relpath(db_file, output_path) for db_file in db_files]

    # Compose the YAML structure as requested
    metadata = {
        "rocprofiler-sdk": {
            "rocpd": {
                rocpd_metadata_param_version: rocpd_package_version,
                # "source": "rocprofv3",  # omitting source, not sure why we need this, and how we determine the source as rocprof-sys, for example.
                "path": ".",
                "files": (
                    rel_paths
                    if len(rel_paths) > 1
                    else (rel_paths[0] if rel_paths else "")
                ),
            }
        }
    }

    metadata_path = os.path.join(output_path, metadata_filename)
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False)

    return metadata_path


def add_args(parser):
    """Add arguments for package."""

    package_options = parser.add_argument_group("Package options")

    package_options.add_argument(
        "-c",
        "--consolidate",
        action="store_true",
        help="Consolidate (copy) database files into a new folder and generate metadata file pointing to that folder",
    )

    package_options.add_argument(
        "-d",
        "--output-path",
        help="Sets the name of output folder (default : current directory)",
        # default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    return [
        "consolidate",
        "output_path",
    ]


def process_args(args, valid_args):

    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def execute(input_files, **kwargs):

    output_path = kwargs.get("output_path", ".")
    consolidate = kwargs.get("consolidate", False)

    if output_path != ".":
        output_path = f"{output_path}.rpdb"
    else:
        # Create a new folder with current date and time for unique folder to consolidate files to
        if consolidate:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"rocpd-{date_str}.rpdb"

    if consolidate:
        # Create a new folder with current date and time
        os.makedirs(output_path, exist_ok=True)
        copied_files = []
        for db_file in input_files:
            dest_file = os.path.join(output_path, os.path.basename(db_file))
            # Only copy if source and destination are not the same file
            if os.path.abspath(db_file) != os.path.abspath(dest_file):
                shutil.copy2(db_file, dest_file)
            copied_files.append(dest_file)
        metadata_path = create_metadata_file(copied_files, output_path, consolidate=True)
    else:
        # If not consolidating, just create metadata file with relative paths to current directory
        metadata_path = create_metadata_file(input_files, output_path)

    print(f"rocPD package created at: {metadata_path}")


def main(argv=None):
    """
    Main function to demonstrate the creation of a metadata file.
    Supports copying database files to a new folder if --copy-db is specified.
    """

    parser = argparse.ArgumentParser(
        description="Convert rocPD to Perfetto file", allow_abbrev=False
    )

    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s). Wildcards accepted, as well as .rpdb folders",
    )

    valid_args = add_args(parser)

    args = parser.parse_args(argv)

    package_args = process_args(args, valid_args)

    input_files = flatten_rocpd_yaml_input_file(args.input)

    execute(input_files, **package_args)


# This is the entry point for the script.
if __name__ == "__main__":
    main()
