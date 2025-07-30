#!/usr/bin/env bash

# Script creates and/or configures the Unified Header submodule
# to work with the Unified Header repository.
#
# User must have access to the Unified Header repository,
# ssh credentials, which resides in the repository.
#
# Should only be run once

# Exit script immediately on any error
set -eu

# Prerequisite: Cloning github amdsmi repository:
# git clone git@github.com:AMD-ROCm-Internal/amdsmi.git

# Defaults:
UH_DIR="include"
UH_NAME="unified_amdsmi"
show_help="False"
remove_module="False"
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
        show_help="True"
        shift
        ;;
    -r|--remove)
        remove_module="True"
        shift
        ;;
    -d|--dir)
        UH_DIR=$2
        shift
        shift
        ;;
    -n|--name)
        UH_NAME=$2
        shift
        shift
        ;;
    *|-*|--*)
        echo "Unknown option $1"
        show_help="True"
        shift
        ;;
    esac
done

if [ "${show_help}" == "True" ]
then
    echo ""
    echo "Usage: $0 --option --opt <value>"
    echo "    option"
    echo "        --help   : Prints usage and exit"
    echo "        --remove : Removes existing Unified Header submodule"
    echo ""
    echo "    opt <value>"
    echo "        --dir <value>  : Directory that contains the Unified Header dir"
    echo "                         Default is \"${UH_DIR}\""
    echo "        --name <value> : Directory Name of the Unified Header"
    echo "                         Default is \"${UH_NAME}\""
    echo "    Note:"
    echo "        Must run script from repository root directory"
    echo ""
    exit 0
fi

if [ "${remove_module}" == "True" ]
then
    git submodule deinit -f "${UH_DIR}/${UH_NAME}/"
    exit 0
fi

# Find repository root dir
REPO_ROOT_DIR=$PWD
while [[ ! -d "${REPO_ROOT_DIR}/.git" ]]
do
    REPO_ROOT_DIR=$(dirname ${REPO_ROOT_DIR})
    if [[ "${REPO_ROOT_DIR}" == "/" ]]
    then
        echo "Repository root dir not found"
        exit 0
    fi
done

# Determine whether to create or configure the submodule
action="Config"
if [ ! -d "${REPO_ROOT_DIR}/${UH_DIR}/${UH_NAME}" ]
then
    action="Create"
else
    if [ -f "${REPO_ROOT_DIR}/${UH_DIR}/${UH_NAME}/.git" ]
    then
        echo "Unified Header submodule is already configured"
        exit 1
    fi

    if [ ! -f ".git/modules/${UH_DIR}/${UH_NAME}/config" ]
    then
        rmdir"${UH_DIR}/${UH_NAME}"
        action="Create"
    fi
fi
echo "$action Unified Header submodule"

# Directory where the Unified Header is created
if [ "$action" == "Create" ]
then
    if [ ! -d "${REPO_ROOT_DIR}/${UH_DIR}" ]
    then
        mkdir -p ${REPO_ROOT_DIR}/${UH_DIR}
    fi
    cd ${REPO_ROOT_DIR}/${UH_DIR}
    git submodule add -f -b amd-mainline git@github.com:AMD-ROCm-Internal/amdsmi_unified.git ${UH_NAME}
fi

# Changes "username" in 3 git files to your Unified Header username
#     amdsmi/.gitmodules
#     amdsmi/.git/config
#     amdsmi/.git/modules/${UH_DIR}/${UH_NAME}/config
cd ${REPO_ROOT_DIR}

if [ "$action" == "Config" ]
then
    # Initializes the submodule
    git submodule init
    git submodule update
fi

cd ${REPO_ROOT_DIR}/${UH_DIR}/${UH_NAME}
SUBMODULE_BRANCH=$(git config -f ${REPO_ROOT_DIR}/.gitmodules submodule.${UH_DIR}/${UH_NAME}.branch || echo "amd-mainline")
git checkout ${SUBMODULE_BRANCH}
cd ${REPO_ROOT_DIR}

if [ "$action" == "Create" ]
then
    echo "Unified Header submodule has been created"
    echo "Commit and push changes to github repository"
    echo "    git commit -s -m \"[ticket-id] Awesome Unified Header submodule addition\""
    echo "    git push"
else
    echo "Unified Header submodule has been configured"
fi

exit 0
