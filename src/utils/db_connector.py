##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

import getpass
import os
from abc import abstractmethod
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils import is_workload_empty

MAX_SERVER_SEL_DELAY = 5000  # 5 sec connection timeout


class DatabaseConnector:
    def __init__(self, args):
        self.args = args
        self.cache = dict()
        self.connection_info = {
            "username": self.args.username,
            "password": self.args.password,
            "host": self.args.host,
            "port": str(self.args.port),
            "team": self.args.team,
            "workload": self.args.workload,
            "db": None,
        }
        self.interaction_type: str = (
            None  # set to 'import' or 'remove' based on user arguments
        )
        self.client: MongoClient = None

    @demarcate
    def prep_import(self):
        # Extract SoC and workload name from sysinfo.csv
        sys_info = str(Path(self.connection_info["workload"]).joinpath("sysinfo.csv"))
        if Path(sys_info).is_file():
            sys_info = pd.read_csv(sys_info)
            try:
                soc = sys_info["gpu_model"][0].strip()
                name = sys_info["workload_name"][0].strip()
            except KeyError as e:
                console_error(
                    f"Outdated workload. "
                    f"Cannot find {e} field. Please reprofile to update."
                )
        else:
            console_error(
                "database", "Unable to parse SoC and/or workload name from sysinfo.csv"
            )

        self.connection_info["db"] = (
            "rocprofiler-compute_"
            + str(self.args.team)
            + "_"
            + str(name)
            + "_"
            + str(soc)
        )

    @demarcate
    def db_import(self):
        self.prep_import()
        i = 0
        file = "blank"
        for file in tqdm(os.listdir(self.connection_info["workload"])):
            if file.endswith(".csv"):
                console_log(
                    "database",
                    "Uploading: %s" % self.connection_info["workload"] + "/" + file,
                )
                try:
                    fileName = file[0 : file.find(".")]
                    data = pd.read_csv(self.connection_info["workload"] + "/" + file)

                    # Demangle original KernelNames
                    kernel_name_shortener(data, self.args.kernel_verbose)
                    data.reset_index(inplace=True)
                    data_dict = data.to_dict("records")

                    client = MongoClient(
                        "mongodb://{}:{}@{}:{}/{}?authSource=admin".format(
                            self.connection_info["username"],
                            self.connection_info["password"],
                            self.connection_info["host"],
                            self.connection_info["port"],
                            self.connection_info["db"],
                        )
                    )
                    db = client[self.connection_info["db"]]
                    collection = db[fileName]
                    collection.insert_many(data_dict)
                    i += 1
                except pd.errors.EmptyDataError:
                    console_warning("database", "Skipping empty file: %s" % file)

        console_log("database", "%s collections successfully added." % i)
        mydb = self.client["workload_names"]
        mycol = mydb["names"]
        value = {"name": self.connection_info["db"]}
        newValue = {"name": self.connection_info["db"]}
        mycol.replace_one(value, newValue, upsert=True)
        console_log("database", "Workload name uploaded.")

    @demarcate
    def db_remove(self):
        db_to_remove = self.client[self.connection_info["workload"]]

        # check the collection names on the database
        self.client.drop_database(db_to_remove)
        db = self.client["workload_names"]
        col = db["names"]
        col.delete_many({"name": self.connection_info["workload"]})

        console_log(
            "database", "Successfully removed %s" % self.connection_info["workload"]
        )

    @abstractmethod
    def pre_processing(self):
        """Perform any pre-processing steps prior to database conncetion."""
        console_debug("database", "pre-processing database connection")
        if not self.args.remove and not self.args.upload:
            console_error(
                "Either -i/--import or -r/--remove is required in database mode"
            )
        self.interaction_type = "import" if self.args.upload else "remove"

        # Detect interaction type
        if self.interaction_type == "remove":
            console_debug("database", "validating arguments for --remove workflow")
            is_full_workload_name = self.args.workload.count("_") >= 3
            if not is_full_workload_name:
                console_error(
                    "-w/--workload is not valid. Please use full workload name "
                    "as seen in GUI when removing (i.e. "
                    "rocprofiler-compute_asw_vcopy_mi200)"
                )
            if (
                self.connection_info["host"] == None
                or self.connection_info["username"] == None
            ):
                console_error(
                    "-H/--host and -u/--username are required when "
                    "interaction type is set to %s" % self.interaction_type
                )
            if (
                self.connection_info["workload"] == "admin"
                or self.connection_info["workload"] == "local"
            ):
                console_error(
                    "Cannot remove %s. Try again." % self.connection_info["workload"]
                )
        else:
            console_debug("database", "validating arguments for --import workflow")
            if (
                self.connection_info["host"] == None
                or self.connection_info["team"] == None
                or self.connection_info["username"] == None
                or self.connection_info["workload"] == None
            ):
                console_error(
                    "-H/--host, -w/--workload, -u/--username, and -t/--team are all "
                    "required when interaction type is set to %s"
                    % self.interaction_type
                )

            if Path(self.connection_info["workload"]).absolute().is_dir():
                is_workload_empty(self.connection_info["workload"])
            else:
                console_error(
                    "--workload is invalid. Please pass path to a valid directory."
                )

            if len(self.args.team) > 13:
                console_error("--team exceeds 13 character limit. Try again.")

            # format path properly
            self.connection_info["workload"] = str(
                Path(self.connection_info["workload"]).absolute().resolve()
            )

        # Detect password
        if self.connection_info["password"] == "":
            try:
                self.connection_info["password"] = getpass.getpass()
            except Exception as e:
                console_error("database", "PASSWORD ERROR %s" % e)
            else:
                console_log("database", "Password received")
        else:
            pass

        # Establish client connection
        connection_str = (
            "mongodb://"
            + self.connection_info["username"]
            + ":"
            + self.connection_info["password"]
            + "@"
            + self.connection_info["host"]
            + ":"
            + self.connection_info["port"]
            + "/?authSource=admin"
        )
        self.client = MongoClient(
            connection_str, serverSelectionTimeoutMS=MAX_SERVER_SEL_DELAY
        )
        try:
            self.client.server_info()
        except Exception:
            console_error("database", "Unable to connect to the DB server.")
