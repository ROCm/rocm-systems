#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# This script gets the name of the default NIC and writes it to standard output.
# NOTE: if command "ip r" finds multiple default NICs, this script will output
#       all of them.
ip r | awk '/default/{print $5}'
