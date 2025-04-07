#!/usr/bin/env bash

pyinstaller src/rocprofiler-compute.py \
    --name "rocprofiler-compute" \
    --add-data "src/perfmon_pub/*:perfmon_pub" \
    --add-data "src/utils/*:utils" \
    --add-data "src/soc_params/*.csv:soc_params" \
    --add-data "src/rocprof_compute_analyze/*:rocprof_compute_analyze" \
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
# Check to se if submodules are available
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
cp -r ${dash_loc:10}/dash_svg "$distpath"/rocprofiler-compute/

echo "(build.sh) Fixing flattened directories"
#TODO: Copy orig file structure from over to flattened packaged version
rm -rf "$distpath"/rocprofiler-compute/rocprof_compute_analyze/
cp -r src/rocprof_compute_analyze/ "$distpath"/rocprofiler-compute/

rm -rf "$distpath"/rocprofiler-compute/perfmon_pub/
cp -r src/perfmon_pub/ "$distpath"/rocprofiler-compute/

rm -rf "$distpath"/rocprofiler-compute/perfmon/
