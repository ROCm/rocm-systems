##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

from abc import ABC, abstractmethod
from utils.utils import error, is_workload_empty, demarcate
from pymongo import MongoClient
from tqdm import tqdm

import os
import logging
import getpass
import pandas as pd

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
            "db": None
        }
        self.interaction_type: str = None #set to 'import' or 'remove' based on user arguments
        self.client: MongoClient = None
    @demarcate
    def prep_import(self, profile_and_export=False):
        if profile_and_export:
            self.connection_info['workload'] = os.path.join(self.connection_info['workload'], self.args.target)

        # Extract SoC and workload name from sysinfo.csv
        sys_info = os.path.join(self.connection_info['workload'], "sysinfo.csv")
        if os.path.isfile(sys_info):
            sys_info = pd.read_csv(sys_info)
            soc = sys_info["name"][0]
            name = sys_info["workload_name"][0]
        else:
            error("[database] Unable to parse SoC and/or workload name from sysinfo.csv")

        self.connection_info["db"] = "omniperf_" + str(self.args.team) + "_" + str(name) + "_" + str(soc)

    @demarcate
    def db_import(self):
        self.prep_import()
        i = 0
        file = "blank"
        for file in tqdm(os.listdir(self.connection_info["workload"])):
            if file.endswith(".csv"):
                logging.info("[database] Uploading: %s" % self.connection_info["workload"] + "/" + file)
                try:
                    fileName = file[0 : file.find(".")]
                    cmd = (
                        "mongoimport --quiet --uri mongodb://{}:{}@{}:{}/{}?authSource=admin --file {} -c {} --drop --type csv --headerline"
                    ).format(
                        self.connection_info["username"],
                        self.connection_info["password"],
                        self.connection_info["host"],
                        self.connection_info["port"],
                        self.connection_info["db"],
                        self.connection_info["workload"] + "/" + file,
                        fileName,
                    )
                    os.system(cmd)
                    i += 1
                except pd.errors.EmptyDataError:
                    logging.info("[database] Skipping empty file: %s" % file)

        logging.info("[database] %s collections successfully added." % i)
        mydb = self.client["workload_names"]
        mycol = mydb["names"]
        value = {"name": self.connection_info["db"]}
        newValue = {"name": self.connection_info["db"]}
        mycol.replace_one(value, newValue, upsert=True)
        logging.info("[database] Workload name uploaded.")

    @demarcate
    def db_remove(self):
        db_to_remove = self.client[self.connection_info['workload']]

        # check the collection names on the database
        col_list = db_to_remove.list_collection_names()
        self.client.drop_database(db_to_remove)
        db = self.client["workload_names"]
        col = db["names"]
        col.delete_many({"name": self.connection_info['workload']})

        logging.info("[database] Successfully removed %s" % self.connection_info['workload'])


    @abstractmethod
    def pre_processing(self):
        """Perform any pre-processing steps prior to database conncetion.
        """
        logging.debug("[database] pre-processing database connection")
        if not self.args.remove and not self.args.upload:
            error("Either -i/--import or -r/--remove is required in database mode")
        self.interaction_type = 'import' if self.args.upload else 'remove'

        # Detect interaction type
        if self.interaction_type == 'remove':
            logging.debug("[database] validating arguments for --remove workflow")
            is_full_workload_name = self.args.workload.count("_") >= 3
            if not is_full_workload_name:
                error("-w/--workload is not valid. Please use full workload name as seen in GUI when removing (i.e. omniperf_asw_vcopy_mi200)")

            if self.connection_info['host'] == None or self.connection_info['username'] == None:
                error("-H/--host and -u/--username are required when interaction type is set to %s" % self.interaction_type)
            if self.connection_info['workload'] == "admin" or self.connection_info['workload'] == "local":
                error("Cannot remove %s. Try again." % self.connection_info['workload'])
        else:
            logging.debug("[database] validating arguments for --import workflow")
            if (
                self.connection_info['host'] == None
                or self.connection_info['team'] == None
                or self.connection_info['username'] == None
                or self.connection_info['workload'] == None
            ):
                error("-H/--host, -w/--workload, -u/--username, and -t/--team are all required when interaction type is set to %s" % self.interaction_type)

            if os.path.isdir(os.path.abspath(self.connection_info['workload'])):
                is_workload_empty(self.connection_info['workload'])
            else:
                error("--workload is invalid. Please pass path to a valid directory.")

            if len(self.args.team) > 13:
                error("--team exceeds 13 character limit. Try again.")
            
            # format path properly
            self.connection_info['workload'] = os.path.abspath(self.connection_info['workload'])

        # Detect password
        if self.connection_info['password'] == "":
            try:
                self.connection_info['password'] = getpass.getpass()
            except Exception as e:
                error("[database] PASSWORD ERROR %s" % e)
            else:
                logging.info("[database] Password recieved")
        else:
            password = self.connection_info['password']

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
        self.client = MongoClient(connection_str, serverSelectionTimeoutMS=MAX_SERVER_SEL_DELAY)
        try:
            self.client.server_info()
        except:
            error("[database] Unable to connect to the DB server.")
        

        