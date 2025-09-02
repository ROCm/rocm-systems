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

from .importer import RocpdImportData
from .time_window import apply_time_window
from . import output_config


import nbformat
from nbformat.v4 import new_notebook, new_code_cell


def write_jupiter(importData, config):

    db_list = ",\n".join(f'"{path}"' for path in importData.databases)

    # Connecting to the database
    db_connect = f"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

db_pathes = [
{db_list}]

# Connecting to the database
conn = sqlite3.connect(\":memory:\")
for i, db_path in enumerate(db_pathes):
    db_alias = f\"db{{i}}\"
    conn.execute(f\"ATTACH DATABASE '{{db_path}}' AS '{{db_alias}}';\")
"""

    # Making a query
    query = f"""
# Making a query
query = "    UNION ALL ".join(
f\"\"\"
SELECT
    event_id AS Event_Id,
    start AS Start_Timestamp,
    end AS End_Timestamp
FROM db{{i}}.rocpd_memory_allocate
\"\"\" for i in range(len(db_pathes))) + \"\"\"
ORDER BY
    event_id ASC, start ASC, end DESC\"\"\"

df = pd.read_sql_query(query, conn)
df.head()
"""

    # Creating a plot
    plot = """
# Convert timestamps from nanoseconds to milliseconds
start = df['Start_Timestamp'] / 1e6
end = df['End_Timestamp'] / 1e6

# Calculate duration in milliseconds
duration = end - start

# Plotting
plt.figure(figsize=(12, 6))
plt.hist(duration, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Memory Allocation Durations')
plt.xlabel('Duration (ms)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('memory_allocate_durations.png')
plt.show()
"""

    # Closing the database connection
    db_disconnect = f"""
# Closing the database connection
conn.close()
"""

    cells = [
        new_code_cell(db_connect.strip()),
        new_code_cell(query.strip()),
        new_code_cell(plot.strip()),
        new_code_cell(db_disconnect.strip()),
    ]

    nb = new_notebook()
    nb.cells = cells

    # Save the notebook to a file
    with open("jupyter_output.ipynb", "w") as f:
        nbformat.write(nb, f)

    print("Jupyter notebook 'jupyter_output.ipynb' created successfully.")


def execute(input, config=None, window_args=None, **kwargs):

    importData = RocpdImportData(input)

    apply_time_window(importData, **window_args)

    config = (
        output_config.output_config(**kwargs)
        if config is None
        else config.update(**kwargs)
    )

    write_jupiter(importData, config)


def add_args(parser):
    """Add jupiter arguments."""

    return []


def process_args(args, valid_args):
    ret = {}
    return ret


def main(argv=None):
    import argparse
    from .time_window import add_args as add_args_time_window
    from .time_window import process_args as process_args_time_window
    from .output_config import add_args as add_args_output_config
    from .output_config import process_args as process_args_output_config
    from .output_config import add_generic_args, process_generic_args

    parser = argparse.ArgumentParser(
        description="Convert rocPD to jupiter notepad files",
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    required_params = parser.add_argument_group("Required arguments")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    valid_out_config_args = add_args_output_config(parser)
    valid_generic_args = add_generic_args(parser)
    valid_time_window_args = add_args_time_window(parser)
    valid_jupiter_args = add_args(parser)

    args = parser.parse_args(argv)

    out_cfg_args = process_args_output_config(args, valid_out_config_args)
    generic_out_cfg_args = process_generic_args(args, valid_generic_args)
    window_args = process_args_time_window(args, valid_time_window_args)
    jupiter_args = process_args(args, valid_jupiter_args)

    all_args = {
        **out_cfg_args,
        **generic_out_cfg_args,
        **jupiter_args,
    }

    execute(args.input, window_args=window_args, **all_args)


if __name__ == "__main__":
    main()
