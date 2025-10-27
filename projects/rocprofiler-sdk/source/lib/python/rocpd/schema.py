#!/usr/bin/env python3
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
#   Utility Class to create the rocpd schema on an existing sqlite connection
#
#       Requires a current copy of the schema in the 'schema' subdirectory
#       Executes the contained sql 'scripts' to create the schema
#

import argparse
from . import libpyrocpd


class RocpdSchema:

    def __init__(self, uuid="", guid=""):

        variables = libpyrocpd.schema_jinja_variables()
        variables.uuid = f"{uuid}"
        variables.guid = f"{guid}"

        self.tables = RocpdSchema.load_schema(
            libpyrocpd.sql_engine.sqlite3,
            libpyrocpd.sql_schema.rocpd_tables,
            libpyrocpd.sql_option.sqlite3_pragma_foreign_keys,
            variables,
        )

        self.indexes = RocpdSchema.load_schema(
            libpyrocpd.sql_engine.sqlite3,
            libpyrocpd.sql_schema.rocpd_indexes,
            libpyrocpd.sql_option.none,
            variables,
        )

        _views = []
        for itr in ["rocpd", "data", "summary", "marker"]:
            _views += [
                RocpdSchema.load_schema(
                    libpyrocpd.sql_engine.sqlite3,
                    getattr(libpyrocpd.sql_schema, f"{itr}_views"),
                    libpyrocpd.sql_option.none,
                    variables,
                )
            ]
        self.views = "\n".join(_views)

    def write_schema(self, connection):
        connection.executescript(self.tables)
        connection.executescript(self.indexes)
        connection.executescript(self.views)

    @staticmethod
    def load_schema(engine, kind, options, variables=None, **kwargs):

        if variables is None:
            variables = libpyrocpd.schema_jinja_variables()

        for itr in ["uuid", "guid"]:
            _variable = kwargs.get(itr, None)
            if _variable is not None:
                setattr(variables, itr, f"{_variable}")

        return libpyrocpd.load_schema(engine, kind, options, variables)


def main(create=None):
    schema = RocpdSchema()

    if create:
        from . import connect

        print(f"Creating empty rpd: {args.create}")
        connection = connect(args.create)
        schema.write_schema(connection)
        connection.commit()
    else:
        print(schema.tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert rocprofiler output to an RPD database"
    )
    parser.add_argument(
        "--create", type=str, help="filename in create empty db", default=None
    )
    args = parser.parse_args()

    main(args.create)
