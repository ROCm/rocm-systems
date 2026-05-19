#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Used by CI to test hipFile IO using aiscp.
#
# NOTE: CI must ensure this script is executed within a directory on an AIS
# supported filesystem backed by an AIS supported device.

# Usage:
# ci-aiscp-test.sh AISCP
#   AISCP: Path to the aiscp binary

set -e

trap 'if [ -e "$TMPDIR" ]; then rm -fr "$TMPDIR"; fi' EXIT

AISCP=$1

echo "pwd: $(pwd)"
echo "aiscp: $AISCP"

TMPDIR=$(mktemp --directory --tmpdir=.)
SRC="$TMPDIR/rand.dat"
DST="$TMPDIR/rand.dat.cp"
echo "Source: $SRC"
echo "Dest:   $DST"

head --bytes 1M /dev/urandom > "$SRC"

"$AISCP" "$SRC" "$DST"

SRC_MD5=$(md5sum "$SRC" | awk '{ print $1 }')
DST_MD5=$(md5sum "$DST" | awk '{ print $1 }')

echo "$SRC_MD5 $SRC"
echo "$DST_MD5 $DST"

if [ "$SRC_MD5" != "$DST_MD5" ]; then
    echo "Files differ! Test failed. MD5 checksums do not match."
    exit 1
fi
