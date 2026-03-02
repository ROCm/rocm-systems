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
import argparse
from . import output_config

rocpd_package_version = "1.0"
rocpd_metadata_param_version = "rocpd_package_version"

IDEAL_NUMBER_OF_DATABASE_FILES = 1
MAX_LIMIT_OF_DATABASE_FILES = 8


def prepare_output_folder(output_path, consolidate) -> str:
    """
    Prepares the output folder path with appropriate .rpdb extension.

    When consolidating to current directory, generates a timestamped folder.
    Otherwise, ensures the provided path has .rpdb extension.

    Args:
        output_path (str): The path to the output folder.
        consolidate (bool): Whether to consolidate output files.

    Returns:
        str: The output folder path with .rpdb extension.
    """
    # Current directory with consolidation - generate timestamped folder
    if output_path == os.getcwd():
        if consolidate:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"rocpd-{date_str}.rpdb"
    else:
        # Custom path provided - ensure it has .rpdb extension
        if not output_path.endswith(".rpdb"):
            output_path = f"{output_path}.rpdb"
    return output_path


def flatten_rocpd_yaml_input_file(input, **kwargs) -> list:
    """
    Processes input files and returns a flattened list of database files.

    Handles multiple input types:
    - YAML files containing rocpd metadata
    - .rpdb folders with index.yaml
    - Direct database files
    - Directories containing .db files
    - Wildcard patterns

    Optionally merges databases if count exceeds threshold.

    Args:
        input (list of str): List of input file paths (YAML, DB, or .rpdb folder).
        skip_auto_merge (bool): If True, skip automatic merging of multiple databases.

    Returns:
        list: List of database file paths.
    """
    import glob

    def parse_yaml_file(yaml_path, base_dir=None):
        """
        Parse a rocprofiler-sdk YAML file and extract database file paths.

        Args:
            yaml_path (str): Path to the YAML file.
            base_dir (str): Base directory for resolving relative paths.

        Returns:
            list: Expanded list of database file paths.
        """
        try:
            import yaml

            with open(yaml_path, "r") as f:
                meta = yaml.safe_load(f)
                rocpd_meta = meta.get("rocprofiler-sdk", {}).get("rocpd", {})

                # Check version compatibility
                version = rocpd_meta.get(rocpd_metadata_param_version, "0")
                if version < rocpd_package_version:
                    print(
                        f"Warning: {yaml_path} is using an outdated version of rocpd package ({version})."
                    )

                # Determine working directory for relative paths
                cwd = (
                    base_dir
                    if base_dir is not None
                    else rocpd_meta.get("path", os.getcwd())
                )

                # Get database file list from YAML
                dbs = rocpd_meta.get("files", [])
                if isinstance(dbs, str):
                    dbs = [dbs]

                # Expand each database path (handle wildcards and relative paths)
                files = []
                for db in dbs:
                    db_path = os.path.join(cwd, db) if not os.path.isabs(db) else db
                    if _contains_wildcard(db_path):
                        files.extend(glob.glob(db_path))
                    else:
                        files.append(db_path)

                return files
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _contains_wildcard(path):
        """Check if path contains wildcard characters."""
        return "*" in path or "?" in path or "[" in path

    def _process_rpdb_folder(item):
        """Process .rpdb folder and extract database files."""
        index_yaml = os.path.join(item, "index.yaml")
        if os.path.isfile(index_yaml):
            return parse_yaml_file(index_yaml, base_dir=os.path.abspath(item))
        else:
            # No index.yaml, search for .db files directly
            return glob.glob(os.path.join(item, "*.db"))

    def _process_yaml_file(item):
        """Process YAML file and extract database files."""
        base_dir = os.path.dirname(item)
        return parse_yaml_file(item, base_dir)

    def _process_directory(item):
        """Process directory and find .db files."""
        return glob.glob(os.path.join(item, "*.db"))

    def _process_file_or_pattern(item):
        """Process individual file or wildcard pattern."""
        if _contains_wildcard(item):
            return glob.glob(item)
        else:
            return [item]

    # Sanitize and categorize input
    sanitized_input = output_config.sanitize_input_list(input)

    # Process each input item based on its type
    input_files = []
    for item in sanitized_input:
        if item.endswith(".rpdb") and os.path.isdir(item):
            input_files.extend(_process_rpdb_folder(item))
        elif item.endswith((".yaml", ".yml")):
            input_files.extend(_process_yaml_file(item))
        elif os.path.isdir(item):
            input_files.extend(_process_directory(item))
        else:
            input_files.extend(_process_file_or_pattern(item))

    # Validate all files exist
    num_dbs = len(input_files)
    print(f"Found {num_dbs} database files.")

    for db in input_files:
        if not os.path.exists(db):
            print(f"Warning: Input database file not found: {db}. Exiting.")
            return []

    # Optionally merge databases if count exceeds threshold
    skip_auto_merge = kwargs.get("skip_auto_merge", False)
    auto_merge_max_limit = kwargs.get("automerge_limit", IDEAL_NUMBER_OF_DATABASE_FILES)

    # Check if user tried to exceed MAX_LIMIT of DBs.  Conservatively set to 8.  SQLite limit is 10.
    if auto_merge_max_limit > MAX_LIMIT_OF_DATABASE_FILES:
        print(
            f"SQLite has a database attach limit of 10.  Max auto-merge limit of {MAX_LIMIT_OF_DATABASE_FILES} set."
        )
        auto_merge_max_limit = MAX_LIMIT_OF_DATABASE_FILES

    if skip_auto_merge:
        print("Skip auto merge and packaging.")
        return input_files

    if num_dbs > auto_merge_max_limit:
        print(
            f"More than {auto_merge_max_limit} database files found. "
            f"It is recommended to merge and package databases"
        )
        merged_files = merge_and_repackage(
            input_files, max_limit=auto_merge_max_limit, **kwargs
        )
        print(f"Reduced to {len(merged_files)} database files.")
        return merged_files

    return input_files


def merge_and_repackage(
    input_files, max_limit=IDEAL_NUMBER_OF_DATABASE_FILES, **kwargs
) -> list:
    """
    Merges and repackages database files into batches to reduce file count.

    If the number of input files is within the limit, returns them unchanged.
    Otherwise, merges files into batches and creates a timestamped .rpdb folder
    with the merged databases and metadata file.

    Args:
        input_files (list of str): List of database file paths to merge.
        max_limit (int): Maximum number of output files desired (default: 1).

    Returns:
        list: List of merged database file paths.
    """
    import uuid

    original_num_dbs = len(input_files)

    # Early return if already within limit
    if original_num_dbs <= max_limit:
        print(
            f"Number of database files ({original_num_dbs}) is within the limit ({max_limit}). "
            f"No merging needed."
        )
        return input_files

    # Calculate batch size for merging
    batch_size = _calculate_batch_size(original_num_dbs, max_limit)
    print(
        f"Original number of DBs: {original_num_dbs}, "
        f"Target number of DBs to merge per batch: {batch_size}"
    )

    # Prepare output folder for merged databases
    unique_str = uuid.uuid4()
    merged_output_folder = prepare_output_folder(os.getcwd(), consolidate=True)
    os.makedirs(merged_output_folder, exist_ok=True)

    # Process databases in batches
    merged_files = _process_batches(
        input_files, batch_size, merged_output_folder, unique_str, **kwargs
    )

    # Display merged file list
    for item in merged_files:
        print(f"Reduced file list: {item}")

    # Create metadata file for the merged databases
    create_metadata_file(merged_files, output_path=merged_output_folder)

    print(
        f"\033[1;34mMerge and repackage completed. "
        f"Output files are located in: {merged_output_folder}\033[0m"
    )

    return merged_files


def _calculate_batch_size(total_files, max_output_files):
    """
    Calculate how many input files should be merged per batch.

    Args:
        total_files (int): Total number of input files.
        max_output_files (int): Desired maximum number of output files.

    Returns:
        int: Number of files to merge per batch.
    """
    # Use ceiling division to ensure we don't exceed max_output_files
    return (total_files // max_output_files) + (total_files % max_output_files > 0)


def _process_batches(input_files, batch_size, output_folder, unique_str, **kwargs):
    """
    Process database files in batches, merging or copying as needed.

    Args:
        input_files (list): List of input database file paths.
        batch_size (int): Number of files per batch.
        output_folder (str): Directory to store merged files.
        unique_str: Unique identifier for merged filenames.

    Returns:
        list: List of merged/copied database file paths.
    """

    merged_files = []
    total_files = len(input_files)

    for batch_index, i in enumerate(range(0, total_files, batch_size)):
        batch_files = input_files[i : i + batch_size]
        merged_filename = f"merged_db_{batch_index}_{unique_str}.db"

        merged_path = _merge_a_batch(
            batch_files, output_folder, merged_filename, **kwargs
        )
        merged_files.append(merged_path)

    return merged_files


def _merge_a_batch(batch_files, output_folder, output_filename, **kwargs):
    """
    Merge multiple files or copy a single file to the output folder.

    Args:
        batch_files (list): List of database files in this batch.
        output_folder (str): Destination folder.
        output_filename (str): Name for the output file.

    Returns:
        str: Path to the merged or copied file.
    """
    from . import merge

    dest_file = os.path.join(output_folder, output_filename)

    if len(batch_files) > 1:
        # Multiple files: merge them
        args = {"output_path": output_folder, "output_file": output_filename}
        return str(merge.execute(batch_files, **args))
    else:
        # Single file: just copy it (optimization)
        # Because this is auto-merge, we want to just copy the file, don't just move it for the user.
        shutil.copy2(batch_files[0], dest_file)
        return str(dest_file)


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
    try:
        import yaml

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Compute relative paths
        rel_paths = [os.path.relpath(db_file, output_path) for db_file in db_files]

        # Compose the YAML structure
        metadata = {
            "rocprofiler-sdk": {
                "rocpd": {
                    rocpd_metadata_param_version: rocpd_package_version,
                    # "source": "rocprofv3",  # omitting source, not sure why we need this, and how we determine the source as rocprof-sys, for example.
                    "path": ".",
                    "files": rel_paths,
                }
            }
        }

        metadata_path = os.path.join(output_path, metadata_filename)
        with open(metadata_path, "w") as f:
            yaml.safe_dump(metadata, f, default_flow_style=False)

    except Exception as e:
        print(f"Error: {e}")
        return None

    return metadata_path


def add_args(parser):
    """Add arguments for package."""

    package_options = parser.add_argument_group("Package options")

    package_options.add_argument(
        "-c",
        "--consolidate",
        action="store_true",
        help="Consolidate (move) database files into a new folder and generate metadata file pointing to that folder",
    )

    package_options.add_argument(
        "--copy",
        action="store_true",
        help="Copy database files instead of moving them",
    )

    package_options.add_argument(
        "-d",
        "--output-path",
        help="Sets the name of output folder (default : current directory)",
        # default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    def process_args(input, args):
        valid_args = [
            "consolidate",
            "copy",
            "output_path",
        ]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        return ret

    return process_args


def execute(input_files, **kwargs):
    import glob

    output_path_kw = kwargs.get("output_path", os.getcwd())
    consolidate = kwargs.get("consolidate", False)
    copy_instead_of_move = kwargs.get("copy", False)

    output_path = prepare_output_folder(output_path_kw, consolidate)
    db_files = output_config.sanitize_input_list(input_files)

    # check if a folder is provided, if so, search for *.db
    expanded_files = []
    for itr in db_files:
        if os.path.isdir(itr):
            expanded_files.extend(glob.glob(os.path.join(itr, "*.db")))
        else:
            expanded_files.append(itr)
    db_files = expanded_files

    if consolidate:
        # Create a new folder with current date and time
        os.makedirs(output_path, exist_ok=True)
        consolidated_files = []
        for db_file in db_files:
            dest_file = os.path.join(output_path, os.path.basename(db_file))
            # Only copy if source and destination are not the same file
            if os.path.abspath(db_file) != os.path.abspath(dest_file):
                if copy_instead_of_move:
                    shutil.copy2(db_file, dest_file)
                else:
                    shutil.move(db_file, dest_file)
            consolidated_files.append(dest_file)
        db_files = consolidated_files

    metadata_path = create_metadata_file(db_files, output_path)

    print(f"rocPD package created at: {metadata_path}")


def main(argv=None):
    """
    Main function to create a metadata file and .rpdb package

    Consolidates to a .rpdb package if --consolidate is specified.
    """

    parser = argparse.ArgumentParser(
        description="Package database files into .rpdb output"
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

    process_args = add_args(parser)

    args = parser.parse_args(argv)

    input_files = flatten_rocpd_yaml_input_file(
        args.input, skip_auto_merge=True, copy=args.copy
    )

    package_args = process_args(None, args)

    # error check for databases before trying to use the data
    if not input_files:
        print("Error, no databases found\n")
        return

    execute(input_files, **package_args)


# This is the entry point for the script.
if __name__ == "__main__":
    main()
