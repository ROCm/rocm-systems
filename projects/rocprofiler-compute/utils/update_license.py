#!/usr/bin/env python3
# -------------------------------------------------------------------------------
# Support script for license header management.
# -------------------------------------------------------------------------------

import argparse
import logging
import glob
import os
import sys
import re
import filecmp
import shutil

begDelim='######bl$'
endDelim='######el$'
maxHeaderLines=200


def cacheLicenseFile(infile,comment='#'):
    if not os.path.isfile(infile):
        logging.error("Unable to access license file - >%s" % infile)
        sys.exit(1)

    license = ""
    with open(infile, 'r') as file_in:
        for line in file_in:
            license += comment + ' ' + line
    return(license)


parser = argparse.ArgumentParser()
parser.add_argument('--license',  required=True,help='License File')
parser.add_argument('--source',   required=True,help='Source directory')
parser.add_argument("--dryrun", help="enable dryrun mode", action="store_true")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--extension',help='file extension to parse')
group.add_argument('--files',    help='specific file(s) to parse')

logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.INFO)

args = parser.parse_args()

srcDir = args.source
fileExtension = None
specificFiles = None
if args.extension:
    fileExtension = args.extension
if args.files:
    specificFiles = args.files.split(",")

print("")
logging.info("Source directory = %s" % srcDir)
if fileExtension:
    logging.info("File extension   = %s" % fileExtension)
if specificFiles:
    logging.info("Specific files   = %s" % specificFiles)

# cache license file
license = cacheLicenseFile(args.license)

# Scan files in provided source directory...
for filename in glob.iglob(srcDir + '/**', recursive=True):
     # skip directories
    if os.path.isdir(filename):
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
            fullPath = os.path.join(srcDir,file)
            if fullPath == filename:
                found = True
                break
        if not found:
            continue
        
    logging.debug("Examining %s for license..." % filename)

    # Update license header contents if delimiters are found
    with open(filename, 'r') as file_in:
        baseName = os.path.basename(filename)
        dirName  = os.path.dirname(filename)
        tmpFile  = dirName + "/." + baseName + ".tmp"

        file_out = open(tmpFile,'w')
        
        for line in file_in:

            if re.search(begDelim,line):
                logging.debug("Found beginning delimiter")
                file_out.write(line)
                file_out.write(license)

                foundEnd = False

                for i in range(maxHeaderLines):
                    line = file_in.readline()
                    if re.search(endDelim,line):
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
    if not filecmp.cmp(filename,tmpFile,shallow=False):
        logging.info("%s changed" % filename)
        shutil.copystat(filename,tmpFile)
        if not args.dryrun:
            os.rename(tmpFile,filename)
    else:
        os.unlink(tmpFile)

        

