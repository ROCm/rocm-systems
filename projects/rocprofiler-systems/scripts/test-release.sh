#!/bin/bash -e

SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
cd $(dirname ${SCRIPT_DIR})
echo -e "Working directory: $(pwd)"

umask 0000

verbose-run()
{
    echo -e "\n##### Executing \"${@}\"... #####\n"
    eval $@
}

: ${RPM_FILE:=""}
: ${DEB_FILE:=""}
: ${STGZ_FILE:=""}
: ${SUDO_CMD:=""}

usage()
{
    print_option() { printf "    --%-10s %-24s     %s\n" "${1}" "${2}" "${3}"; }
    echo "Options:"
    print_option stgz "<FILE>" "Run installation test on STGZ package"
    print_option deb "<FILE>" "Run installation test on DEB package"
    print_option rpm "<FILE>" "Run installation test on RPM package"
    print_option sudo "" "Execute commands with sudo"
}

while [[ $# -gt 0 ]]
do
    ARG=${1}
    shift

    case "${ARG}" in
        --clean)
            CLEAN=1
            continue
            ;;
        --fresh)
            FRESH=1
            continue
            ;;
    esac

    VAL=""
    while [[ $# -gt 0 ]]
    do
        VAL=${1}
        shift
        break
    done

    if [ -z "${VAL}" ]; then
        echo "Error! Missing value for argument \"${ARG}\""
        usage
        exit -1
    fi

    case "${ARG}" in
        ? | -h | --help)
            usage
            exit 0
            ;;
        --stgz)
            STGZ_FILE=${VAL}
            ;;
        --deb)
            DEB_FILE=${VAL}
            ;;
        --rpm)
            RPM_FILE=${VAL}
            ;;
        --sudo)
            SUDO_CMD=sudo
            ;;
        *)
            echo -e "Error! Unknown option : ${ARG}"
            usage
            exit -1
            ;;
    esac
done

remove-pycache()
{
    rm -rf ${1}/lib/python/site-packages/rocprofsys/__pycache__
}

setup-env()
{
    rm -rf ${1}/*
    export PATH=${1}/bin:${PATH}
    export LD_LIBRARY_PATH=${1}/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}
}

test-install()
{
    verbose-run rocprof-sys-instrument --help
    verbose-run rocprof-sys-avail --help
    verbose-run rocprof-sys-avail --all
    if [ -d "${1}/lib/python/site-packages/rocprofsys" ]; then
        verbose-run rocprof-sys-python --help
    fi
}

change-directory()
{
    if [ ! -f "${1}" ]; then
        if [ -f "/home/omnitrace/${1}" ]; then
            cd /home/omnitrace
        elif [ -f "/home/omnitrace/docker/${1}" ]; then
            cd /home/omnitrace/docker
        fi
    fi
    realpath ${1}
}

test-stgz()
{
    if [ -z "${1}" ]; then return; fi

    local INSTALLER=$(change-directory ${1})
    mkdir /opt/rocprof-sys-stgz
    setup-env /opt/rocprof-sys-stgz

    verbose-run ${INSTALLER} --prefix=/opt/rocprof-sys-stgz --skip-license --exclude-dir

    test-install /opt/rocprof-sys-stgz
}

test-deb()
{
    if [ -z "${1}" ]; then return; fi

    local INSTALLER=$(change-directory ${1})
    setup-env /opt/rocprof-sys

    verbose-run ${SUDO_CMD} dpkg --contents ${INSTALLER}
    verbose-run ${SUDO_CMD} dpkg -i ${INSTALLER}

    test-install /opt/rocprof-sys
    remove-pycache /opt/rocprof-sys
    verbose-run apt-get remove -y rocprof-sys
    if [ -d /opt/rocprof-sys ]; then
        find /opt/rocprof-sys -type f
    fi
}

test-rpm()
{
    if [ -z "${1}" ]; then return; fi

    local INSTALLER=$(change-directory ${1})
    setup-env /opt/rocprof-sys

    verbose-run ${SUDO_CMD} rpm -ql -p ${INSTALLER}
    verbose-run ${SUDO_CMD} rpm -v -i -p ${INSTALLER} --nodeps

    test-install /opt/rocprof-sys
    remove-pycache /opt/rocprof-sys
    verbose-run rpm -e rocprof-sys
    if [ -d /opt/rocprof-sys ]; then
        find /opt/rocprof-sys -type f
    fi
}

test-stgz ${STGZ_FILE}
test-deb  ${DEB_FILE}
test-rpm  ${RPM_FILE}
