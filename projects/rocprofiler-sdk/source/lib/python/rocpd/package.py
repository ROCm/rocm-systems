import os

# import json
import shutil
import datetime
import yaml
import argparse
from . import output_config

rocpd_package_version = "1.0"

rocpd_metadata_param_version = "rocpd_package_version"
rocpd_metadata_param_current_directory = "rocpd_current_directory"
rocpd_metadata_param_database_files = "rocpd_relative_path_to_database_files"


def flatten_rocpd_yaml_input_file(input) -> list:
    """
    Processes a YAML file containing rocPD metadata and returns a list of database files.

    Args:
        input (str): Path to the YAML file.

    Returns:
        list: List of database file paths.
    """
    # Flatten input list if any YAML file is provided
    input_files = []
    for item in input:
        if item.endswith((".yaml", ".yml")):
            with open(item, "r") as f:
                meta = yaml.safe_load(f)
                cwd = meta.get(rocpd_metadata_param_current_directory, os.getcwd())
                dbs = meta.get(rocpd_metadata_param_database_files, [])
                new_relative_dbs = [
                    os.path.join(cwd, db) if not os.path.isabs(db) else db for db in dbs
                ]
                input_files.extend(new_relative_dbs)
        else:
            input_files.append(item)
    return input_files


def create_metadata_file(
    db_files, output_path=".", metadata_filename="index.yaml", consolidate=False
):
    """
    Creates a metadata file listing the relative paths to the provided SQL database files.

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

    # If consolidating, set current directory to .
    if consolidate:
        current_directory = "."
    else:
        current_directory = os.path.normpath(os.path.join(os.getcwd(), output_path))

    metadata = {
        rocpd_metadata_param_version: rocpd_package_version,
        rocpd_metadata_param_current_directory: current_directory,
        rocpd_metadata_param_database_files: rel_paths,
    }

    metadata_path = os.path.join(output_path, metadata_filename)
    with open(metadata_path, "w") as f:
        # json.dump(metadata, f, indent=4)  # Uncomment for JSON format
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
    consolidate = kwargs.get("consolidate", "False")

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
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    parser.add_argument(
        "-d",
        "--output-path",
        help="Sets the name of output folder (default : current directory)",
        # default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--consolidate",
        action="store_true",
        help="Consolidate (copy) database files into a new folder and generate metadata file pointing to that folder",
    )

    args = parser.parse_args(argv)

    input_files = flatten_rocpd_yaml_input_file(args.input)

    # TODO: fix this complicated logic. Let's make it simpler.
    if args.output_path:
        output_path = args.output_path
    else:
        # Create a new folder with current date and time for unique folder to consolidate files to
        if args.consolidate:
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            output_path = f"rocpd-{date_str}.rpdb"
        else:
            output_path = "."

    if args.consolidate:
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


# This is the entry point for the script.
if __name__ == "__main__":
    main()
