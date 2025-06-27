###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#
# Utility classes to merge rpd files
#
#
import argparse
import os
import sqlite3
import time

from collections import defaultdict
from typing import List, Any, Dict
from pathlib import Path

# from .schema import RocpdSchema

__all__ = ["RocpdMerger", "execute"]


def prepare_output_file(output: str) -> None:
    """Prepare output file by creating directory and removing existing file"""

    output_path = Path(output)

    # Create parent directory if needed
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.is_file():
        output_path.unlink()

class RocpdMerger():

    def __init__(self, input, output):
 
        if isinstance(input, sqlite3.Connection):
            raise ValueError(
                "RocpdMerger does not accept existing sqlite3 connections"
            )
        elif isinstance(input, str):
             raise ValueError(
                "RocpdMerger only accepts a list of filenames to merge, not a single filename"
            )
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            self._filenames = input[:]
            self._output = output
            prepare_output_file(self._output)
            self._connection = sqlite3.connect(self._output)
            
        else:
            raise ValueError(
                f"input is unsupported type. Expected list of strings. type={type(input).__name__}"
            )

    def __enter__(self):
        # support "with RocpdMerge(...) as db:":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._connection.close()
        print('Closing connection to output database')
    
    def _create_union_views(self, views_by_base_name) -> list:
        union_views = []

        for view_name, table_names in views_by_base_name.items():
            if len(table_names) == 1:
                union_views.append(
                    f"""CREATE VIEW IF NOT EXISTS `{view_name}` AS SELECT * FROM `{table_names[0]}`;"""
                )
            else:
                select_statements = [f"SELECT * FROM `{table}`" for table in table_names]
                union_query = "\nUNION ALL\n".join(select_statements)

                union_views.append(
                    f"""CREATE VIEW IF NOT EXISTS `{view_name}` AS {union_query};"""
                )
        return "\n\n".join(union_views)

    def merge(self):
        """
        Merge multiple SQLite databases into a single destination database.
        """
        cur_dest = self._connection.cursor()

        views_by_base_name = defaultdict(list) 
        
        versions = [] # Check that all databases have the same schema version
        
        for orig in self._filenames:
            
            print(f'Adding {orig}')
            
            con_orig = sqlite3.connect(orig)
            cur_orig = con_orig.cursor()

            cur_orig.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cur_orig.fetchall()
            print(f'Tables found: {len(tables) -1}')
            
            uudid_statement = "SELECT value FROM rocpd_metadata WHERE tag='uuid'"
            _uuid = [ itr[0] for itr in cur_orig.execute(uudid_statement).fetchall()][0]
            
            version_statement = "SELECT value FROM rocpd_metadata WHERE tag='schema_version'"
            versions.extend([ itr[0] for itr in cur_orig.execute(version_statement).fetchall()])
            
            for table in tables:
                table_name = table[0]
                if "sqlite_sequence" in table_name:
                    continue

                view_name = table_name.replace(_uuid, "")
                views_by_base_name[view_name].append(table_name)

                cur_orig.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                create_table_stmt = cur_orig.fetchone()[0]

                cur_dest.execute(create_table_stmt)

                cur_orig.execute(f"SELECT * FROM {table_name}")
                rows = cur_orig.fetchall()
                for row in rows:
                    placeholders = ', '.join('?' * len(row))
                    cur_dest.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)
                    
            con_orig.close()
            
        assert len(list(set(versions))) == 1 , f'Multiple versions found : {list(set(versions))}'
        
        # Create rocpd_<> views
        self._connection.executescript(self._create_union_views(views_by_base_name))
        
        # Create data views
        con_orig = sqlite3.connect(self._filenames[0])
        orig_cursor = con_orig.cursor()
        orig_cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='view' AND name NOT LIKE 'rocpd_%';")
        views = orig_cursor.fetchall()
        
        for view  in views:
            _ , sql_view = view
            self._connection.executescript(sql_view)
            
        con_orig.close()
        # self._connection.executescript(RocpdSchema().views)
        
        self._connection.commit()

#
# Command-line interface functions
#
def add_args(parser):
    """Add arguments for merger."""
    
    o_options = parser.add_argument_group("Output options")

    o_options.add_argument(
        "-o",
        "--output-file",
        help="Sets the base output file name",
        default=os.environ.get("ROCPD_OUTPUT_NAME", "merged"),
        type=str,
        required=False,
    )
    o_options.add_argument(
        "-d",
        "--output-path",
        help="Sets the output path where the output files will be saved (default path: `./rocpd-output-data`)",
        default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    return ["output_file", "output_path"]



def process_args(args, valid_args):
    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def execute(inputs: List[str], **kwargs: Dict[str, Any]) -> None:

    start_time = time.time()

    output_path = kwargs.get("output_path")
    output_filename = kwargs.get("output_file") + ".db"
    output = Path(output_path, output_filename)
    
    with RocpdMerger(inputs, output) as merger:
        merger.merge()

    elapsed_time = time.time() - start_time
    
    print(f"Merge completed successfully! Output saved to: {output}")
    print(f"Time: {elapsed_time:.2f} sec")



def main(argv=None) -> int:
    """Main entry point for command line execution."""

    parser = argparse.ArgumentParser(description="Merge ROCpd databases")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input ROCpd database files",
    )

    valid_args = add_args(parser)

    args = parser.parse_args(argv)

    merged_args = process_args(args, valid_args)

    execute(args.input, **merged_args)


if __name__ == "__main__":
    main()
