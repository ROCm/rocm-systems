#!/usr/bin/env bash

pyinstaller src/omniperf.py \
    --name "omniperf" \
    --add-data "src/perfmon_pub/*:perfmon_pub" \
    --add-data "src/utils/*:utils" \
    --add-data "src/soc_params/*.csv:soc_params" \
    --add-data "src/omniperf_cli/*:omniperf_cli" \
    --hidden-import matplotlib.backends.backend_pdf \
    ${@}

while [ $# -gt 0 ]; do
    case "$1" in
        -y*)
            if [[ "$1" != *=* ]]; then shift; fi
            y="${1#*=}"
            ;;
        --workpath*)
            if [[ "$1" != *=* ]]; then shift; fi
            workpath="${1#*=}"
            ;;
        --distpath*)
            if [[ "$1" != *=* ]]; then shift; fi
            distpath="${1#*=}"
            ;;
        *)
            echo "Invalid argument"
            exit 1
            ;;
    esac
    shift
done

echo "distpath=$distpath"

echo "(build.sh) Checking for submodules"
# Check to se if submodules are availible 
if [ -d "src/waveparser/" ] && [ -d "src/multevent/" ]
then
    echo "Found submodules"
    if [ "$(ls -A src/waveparser/)" ] && [ "$(ls -A src/multevent/)" ]; then
        echo "waveparser and multevents submodules loaded. Packaging..."
        cp -r src/waveparser/ "$distpath"/waveparser/
        cp -r src/multevent/ "$distpath"/multevent/
    else
        echo "One of your submodules isn't loaded. Skipping submodule packaging"
    fi
else
    echo "ERROR: Couldn't find directory for submodules"
fi

echo "(build.sh) Loading dash_svg"
# Take care of dash-svg module that isn't detected by PyInstaller
dash_info=$(pip3 show dash_svg)
dash_loc=$(sed -n '8p' <<<"$dash_info")
cp -r ${dash_loc:10}/dash_svg "$distpath"/omniperf/

echo "(build.sh) Fixing flattened directories"
#TODO: Copy orig file structure from over to flattened packaged version
rm -rf "$distpath"/omniperf/omniperf_cli/
cp -r src/omniperf_cli/ "$distpath"/omniperf/

rm -rf "$distpath"/omniperf/perfmon_pub/
cp -r src/perfmon_pub/ "$distpath"/omniperf/

rm -rf "$distpath"/omniperf/perfmon/
