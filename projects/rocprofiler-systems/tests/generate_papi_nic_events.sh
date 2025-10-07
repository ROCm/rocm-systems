#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# This script gets a list of default NICs from ip command
# and generates a list of PAPI events, 4 for each NIC.
# and generates a list of PAPI events; 4 for each NIC.
# For example, if the NIC is enp7s0, the PAPI events are:
# net:::enp7s0:tx:byte net:::enp7s0:rx:byte net:::enp7s0:tx:packet net:::enp7s0:rx:packet

# Get the directory where this script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$script_dir/get_default_nic.sh" ]; then
  echo "Error: helper script get_default_nic.sh not found or not executable in $script_dir" >&2
  exit 1
fi

# Call the `get_default_nic.sh`` script to get the list of default NICs
# and store it in the nic_list variable
nic_list="$("$script_dir/get_default_nic.sh")"
if [ $? -ne 0 ]; then
  echo "Error: failed to get default NICs" >&2
  exit 1
fi

events=() 

for nic in $nic_list; do  
  events+=("net:::${nic}:tx:byte" "net:::${nic}:rx:byte" "net:::${nic}:tx:packet" "net:::${nic}:rx:packet")  
done  

event_list="${events[*]}" 
echo $event_list
