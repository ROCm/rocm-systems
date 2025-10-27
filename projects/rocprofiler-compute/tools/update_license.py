##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Support script for license header management.
# -------------------------------------------------------------------------------

import argparse
import filecmp
import glob
import logging
import os
import re
import shutil
import sys
from pathlib import Path

begDelim = "######bl$"
endDelim = "######el$"
maxHeaderLines = 200


def cacheLicenseFile(infile, comment="#"):
    if not Path(infile).is_file():
        logging.error(f"Unable to access license file - >{infile}")
        sys.exit(1)

    license = ""
    with open(infile) as file_in:
        for line in file_in:
            license += comment
            if line.strip() != "":
                license += " "
            license += line
    return license


parser = argparse.ArgumentParser()
parser.add_argument("--license", required=True, help="License File")
parser.add_argument("--source", required=True, help="Source directory")
parser.add_argument("--dryrun", help="enable dryrun mode", action="store_true")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--extension", help="file extension to parse")
group.add_argument("--files", help="specific file(s) to parse")

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

args = parser.parse_args()

srcDir = args.source
fileExtension = None
specificFiles = None
if args.extension:
    fileExtension = args.extension
if args.files:
    specificFiles = args.files.split(",")

print("")
logging.info(f"Source directory = {srcDir}")
if fileExtension:
    logging.info(f"File extension   = {fileExtension}")
if specificFiles:
    logging.info(f"Specific files   = {specificFiles}")

# cache license file
license = cacheLicenseFile(args.license)

# Scan files in provided source directory...
for filename in glob.iglob(f"{srcDir}/**", recursive=True):
    # skip directories
    if Path(filename).is_dir():
        continue

    # File matching options:

    # (1) filter non-matching extensions
    if fileExtension:
        if not filename.endswith(fileExtension):
            continue

    # or, (2) filter for specific filename
    if specificFiles:
        found = False
        for file in specificFiles:
            fullPath = str(Path(srcDir) / file)
            if fullPath == filename:
                found = True
                break
        if not found:
            continue

    logging.debug(f"Examining {filename} for license...")

    # Update license header contents if delimiters are found
    with open(filename) as file_in:
        base_name = Path(filename).name
        dir_name = Path(filename).parent
        tmp_file = dir_name / f".{base_name}.tmp"

        file_out = open(tmp_file, "w")
        for line in file_in:
            if re.search(begDelim, line):
                logging.debug("Found beginning delimiter")
                file_out.write(line)
                file_out.write(license)

                foundEnd = False

                for i in range(maxHeaderLines):
                    line = file_in.readline()
                    if re.search(endDelim, line):
                        logging.debug("Found ending delimiter")
                        file_out.write(line)
                        foundEnd = True
                        break
                if not foundEnd:
                    logging.error("Unable to find end of delimited header")
                    sys.exit(1)

            else:
                file_out.write(line)

    file_out.close()

    # Check if file changed and update
    if not filecmp.cmp(filename, tmp_file, shallow=False):
        logging.info(f"{filename} changed")
        shutil.copystat(filename, tmp_file)
        if not args.dryrun:
            os.rename(tmp_file, filename)
    else:
        os.unlink(tmp_file)
