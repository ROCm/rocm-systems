#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# This script gets the name of the default NIC and writes it to standard output.
# NOTE: if command "ip r" finds multiple default NICs, this script will output
#       all of them.
nics=$(ip r | awk '/^default /{print $5}' | sort -u)

# nics="ens50f1 ens50f2 ens50f3 ens50f4"  # For testing purposes only
# nics= # For testing purposes only

if [ -z "$nics" ]; then
  echo "Error: no default route found" >&2
  exit 1
fi

echo "$nics"
