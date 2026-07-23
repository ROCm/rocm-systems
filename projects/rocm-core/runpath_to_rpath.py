#!/usr/bin/env python3

##Copyright © Advanced Micro Devices, Inc., or its affiliates.
##SPDX-License-Identifier: MIT

import os
import sys
import subprocess
import argparse
import pathlib
import re


def _get_rpath(filepath: str):
    """Read DT_RPATH or DT_RUNPATH from an ELF binary via readelf.

    patchelf --print-rpath only reads DT_RPATH, not DT_RUNPATH, so using it
    directly on a binary that has only DT_RUNPATH returns an empty string and
    would cause --set-rpath to wipe the rpath. readelf reads both tags.
    """
    try:
        out = subprocess.check_output(
            ["readelf", "-d", filepath],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None
    m = re.search(r"\(R(?:UN)?PATH\)\s+Library r(?:un)?path: \[(.+)\]", out)
    return m.group(1) if m else None


def update_rpath(search_path, excludes):
    """Change DT_RUNPATH to DT_RPATH in all ELF files under search_path.

    Uses patchelf --force-rpath --set-rpath, which is the same mechanism used
    by the Python wheel packaging path. readelf is used to read the existing
    rpath value because patchelf --print-rpath only reads DT_RPATH, not
    DT_RUNPATH, and would return empty for binaries that only have DT_RUNPATH.
    """
    for path, dirs, files in os.walk(search_path, topdown=True, followlinks=True):
        dirs[:] = [d for d in dirs if d not in excludes]
        for filename in files:
            filepath = os.path.join(path, filename)
            if os.path.islink(filepath):
                continue
            # Quick ELF magic check before invoking readelf/patchelf
            try:
                with open(filepath, "rb") as f:
                    if f.read(4) != b"\x7fELF":
                        continue
            except OSError:
                continue
            rpath = _get_rpath(filepath)
            if not rpath:
                continue
            # Write the rpath back, forcing DT_RPATH tag instead of DT_RUNPATH
            try:
                subprocess.check_call(
                    [
                        "patchelf",
                        "--force-rpath",
                        "--set-rpath", rpath,
                        filepath,
                    ]
                )
                print(f"DT_RUNPATH changed to DT_RPATH: {filepath}")
            except subprocess.CalledProcessError as ex:
                print(f"patchelf failed for {filepath}: {ex}")

def update_config_file(cfg_path):
    ''' Function helps to update rocm llvm config file to default to DT_RPATH. '''
    print("Updating cfg file in", cfg_path)
    config_file_exist = os.path.exists(cfg_path)
    if config_file_exist:
        print("cfg file exist in path, going ahead with update ")
        search_str = "enable-new-dtags"
        replace_str = "disable-new-dtags"
        try:
            # Read contents from file as a single string
            file_string = ''
            with open(cfg_path, 'r') as f:
                file_string = f.read()

            # Use RE package for string replacement
            file_string = (re.sub(search_str, replace_str, file_string))

            # Write contents back to file. Using mode 'w' truncates the file.
            with open(cfg_path, 'w') as f:
                f.write(file_string)
        except Exception as ex:
            print("Couldnt update rocm.cfg file. ",  ex)
    else:
        print("Config path doesnt exist", cfg_path)

def update_compiler_config(search_path):
    ''' Function search for rocm llvm config(rocm.cfg) file in the search_path folder.
        If the config file is not foung search in ROCM_PATH. Once the config file is found,
        update llvm config to default to DT_RPATH '''
    cfg_file_name = "rocm.cfg"
    found_cfg = False
    print("Searching for ", cfg_file_name)
    for path, dirs, files in os.walk(search_path):
        # Search for rocm.cfg in the search path and default to DT_RPATH
        if cfg_file_name in files:
            cfg_path = os.path.join(path, cfg_file_name)
            print(" Found cfg file cfg_path")
            found_cfg = True
            update_config_file(cfg_path)
            # Continue with the search as there could be cfg files in llvm and llvm/alt
            continue;
    if found_cfg:
        return
    # rocm.cfg config file not found in search path. Search in the ROCM_PATH.
    print(cfg_file_name, " not found in search_path. Trying to search in ROCM_PATH")
    try :
        rocm_path = os.environ["ROCM_PATH"]
        print(" Found ROCM_PATH trying for rocm.cfg")
        # There are multiple possible paths for cfg file.
        # ROCM_PATH/llvm/bin and ROCM_PATH/lib/llvm/bin. Also alt location
        update_config_file(rocm_path + "/llvm/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/llvm/alt/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/lib/llvm/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/lib/llvm/alt/bin/" + cfg_file_name)
            # Found config file. Change default DT_RUNPATH setting to DT_RPATH
    except Exception as ex:
        print("ROCM_PATH not found ", ex)

def main():
    # The script expect a search folder as parameter. It finds all ELF files and updates RPATH
    argparser = argparse.ArgumentParser(
            usage='usage: %(prog)s  <folder-to-search>',
            description='Find the ELF files in the specified folder and convert the RUNPATH to RPATH. \n',
            add_help=False,
            prog='runpath_to_rpath.py')

    argparser.add_argument('searchdir',
            nargs='?', type=pathlib.Path, default=None,
            help='Folder to search for ELF file. \nPlease note: Any folder with name llvm in that path will be discarded')
    argparser.add_argument('-h', '--help',
            action='store_true', dest='help',
            help='Display this information')

    args = argparser.parse_args()
    if args.help or not args.searchdir:
        argparser.print_help()
        sys.exit(0)

    # Find the elf files in the search path and update DT_RUNPATH to DT_RPATH
    # SWDEV-467155 : remove the exclusion of llvm folder
    excludes = []
    update_rpath(args.searchdir, excludes)
    # Update rocm clang configs to default to DT_RPATH
    update_compiler_config(args.searchdir)
    print("Done with rpath update")

if __name__ == "__main__":
    main()
