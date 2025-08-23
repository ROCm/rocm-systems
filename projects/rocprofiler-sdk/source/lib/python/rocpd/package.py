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


def create_output_folder(output_path, consolidate) -> str:
    """
    Creates the output folder if it doesn't exist.

    Args:
        output_path (str): The path to the output folder.
        consolidate (bool): Whether to consolidate output files.

    Returns:
        str: The path to the created output folder.
    """
    if output_path != ".":
        if not output_path.endswith(".rpdb"):
            output_path = f"{output_path}.rpdb"
    else:
        # Create a new folder with current date and time for unique folder to consolidate files to
        if consolidate:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"rocpd-{date_str}.rpdb"
    return output_path


def flatten_rocpd_yaml_input_file(input, skip_auto_merge=False) -> list:
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
            version = rocpd_meta.get(rocpd_metadata_param_version, "0")
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

    return_list = []
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
            # If a directory, check inside for *.db files
            if os.path.isdir(item):
                input_files.extend(glob.glob(os.path.join(item, "*.db")))
            else:
                # Expand wildcards in direct input parameters as well
                if "*" in item or "?" in item or "[" in item:
                    input_files.extend(glob.glob(item))
                else:
                    input_files.append(item)

    num_dbs = len(input_files)
    print(f"Found {num_dbs} database files.")
    return_list = input_files

    if skip_auto_merge:
        print("Skip auto merge and packaging.")
    else:
        if num_dbs > IDEAL_NUMBER_OF_DATABASE_FILES:
            print(
                f"More than {IDEAL_NUMBER_OF_DATABASE_FILES} database files found. It is recommended to merge and package databases"
            )
            fewer_input_files = merge_and_repackage(input_files)
            print(f"Reduced to {len(fewer_input_files)} database files.")
            return_list = fewer_input_files

    return return_list


def merge_and_repackage(input_files, max_limit=IDEAL_NUMBER_OF_DATABASE_FILES) -> list:
    """
    Merges and repackages the input database files.

    Args:
        input_files (list of str): List of database file paths.

    Returns:
        list: List of merged and repackaged database file paths.
    """
    from . import merge

    # Maybe this try is not necessary since uuid is standard python lib
    try:
        import uuid

        unique_str = uuid.uuid4()
    except Exception as e:
        print(
            f"Warning: could not import uuid, falling back to time as unique string. {e}"
        )
        unique_str = datetime.datetime.now().strftime("%H%M%S")

    original_num_dbs = len(input_files)

    # If within the LIMIT, then return the DBs, no automerge required
    if original_num_dbs <= max_limit:
        print(
            f"Number of database files ({original_num_dbs}) is within the limit ({max_limit}). No merging needed."
        )
        return input_files

    # Otherwise, calculate how many DBs to merge
    target_num_dbs_to_merge = (original_num_dbs // max_limit) + (
        original_num_dbs % max_limit > 0
    )
    print(
        f"Original number of DBs: {original_num_dbs}, Target number of DBs to merge during each batch: {target_num_dbs_to_merge}"
    )

    # Create an output folder to store the merged DBs to
    merged_output_folder = create_output_folder(".", consolidate=True)
    os.makedirs(merged_output_folder, exist_ok=True)

    # Beging batch processing the DBs
    reduced_file_list = []
    for i in range(0, original_num_dbs, target_num_dbs_to_merge):
        batch_files = input_files[i : i + target_num_dbs_to_merge]
        merged_filename = f"merged_db_{i // target_num_dbs_to_merge}_{unique_str}.db"
        args = {"output_path": merged_output_folder, "output_file": merged_filename}
        if len(batch_files) > 1:
            reduced_file_list.append(str(merge.execute(batch_files, **args)))
        elif len(batch_files) == 1:
            # optimize, if just 1 db, no need to call merge, just copy it
            dest_file = os.path.join(merged_output_folder, merged_filename)
            shutil.copy2(batch_files[0], dest_file)
            reduced_file_list.append(str(dest_file))

    for item in reduced_file_list:
        print(f"Reduced file list: {item}")

    # Once # of dbs is reduced (merged), then package and create the metadata .yaml file
    create_metadata_file(reduced_file_list, output_path=merged_output_folder)

    print(
        f"\033[1;34mMerge and repackage completed. Output files are located in: {merged_output_folder}\033[0m"
    )

    return reduced_file_list


def create_metadata_file(db_files, output_path=".", metadata_filename="index.yaml"):
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

    output_path_kw = kwargs.get("output_path", ".")
    consolidate = kwargs.get("consolidate", False)

    output_path = create_output_folder(output_path_kw, consolidate)
    db_files = input_files

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
        db_files = copied_files

    metadata_path = create_metadata_file(db_files, output_path)

    print(f"rocPD package created at: {metadata_path}")


def main(argv=None):
    """
    Main function to demonstrate the creation of a metadata file and .rpdb package

    Consolidates to a .rpdb package if --consolidate is specified.
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

    input_files = flatten_rocpd_yaml_input_file(args.input, skip_auto_merge=True)

    # error check for databases before trying to use the data
    if not input_files:
        print("Error, no databases found\n")
        return

    execute(input_files, **package_args)


# This is the entry point for the script.
if __name__ == "__main__":
    main()
