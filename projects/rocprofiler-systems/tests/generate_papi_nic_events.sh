#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# This script gets a list of default NICs from ip command
# and generates a list of PAPI events, 4 for each NIC.
# and generates a list of PAPI events; 4 for each NIC.
# For example, if the NIC is enp7s0, the PAPI events are:
# net:::enp7s0:tx:byte net:::enp7s0:rx:byte net:::enp7s0:tx:packet net:::enp7s0:rx:packet

nic_list=$(ip r | awk '/default/{print $5}')

event_list=
for nic in $nic_list
do
  event_list="$event_list net:::$nic:tx:byte net:::$nic:rx:byte net:::$nic:tx:packet net:::$nic:rx:packet"
done
echo $event_list
