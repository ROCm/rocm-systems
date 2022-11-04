#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import argparse
import getpass
from pymongo import MongoClient

# Verify target directory and setup connection
def remove_workload(args):
    # parser = argparse.ArgumentParser(description='Remove a workload from an Omniperf Instance')

    # parser.add_argument('-H', '--host',required=True, help="Name or IP address of the server host")
    # parser.add_argument('-P', '--port', required=False, help="TCP/IP Port (defaults to 27018)", default=27018)
    # parser.add_argument('-u', '--username', required=True, help="Name of the user to connect with")
    # parser.add_argument('-p', '--password', required=False, help="The user's password (will be requested later if it's not set)", default='')
    # parser.add_argument('-w', '--workload', required=True, help="Specify the full workload name to delete")

    host = args.host
    port = str(args.port)
    username = args.username

    if args.password == "":
        try:
            password = getpass.getpass()
        except Exception as error:
            print("PASSWORD ERROR", error)
        else:
            print("Password recieved")
    else:
        password = args.password
    workload = args.workload

    # Verify workload  is valid
    if workload == "admin" or workload == "local":
        print("You cannot remove this database. Try again.")
    else:
        print("Attempting to remove " + workload)

    connection_info = {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "workload": workload,
    }

    client = MongoClient(
        "mongodb://"
        + connection_info["username"]
        + ":"
        + connection_info["password"]
        + "@"
        + connection_info["host"]
        + ":"
        + connection_info["port"]
        + "/?authSource=admin"
    )

    dbToRemove = client[workload]

    # check the collection names on the database
    col_list = dbToRemove.list_collection_names()
    # if not col_list:
    #     print("This workload appears to be empty... Try again.")
    # else:
    client.drop_database(dbToRemove)
    db = client["workload_names"]
    col = db["names"]
    col.delete_many({"name": workload})

    print("Done!")
