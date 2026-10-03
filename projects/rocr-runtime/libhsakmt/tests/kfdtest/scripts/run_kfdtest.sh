#!/bin/bash
#
# Copyright (C) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#

# See if we can find the SHARE/BIN dirs in their expected locations
CWD="${BASH_SOURCE%/*}"
while read candidate; do
    if [ -e "$candidate/kfdtest.exclude" ]; then
        source "$candidate/kfdtest.exclude"
        break
    fi
done <<EOF
$KFDTEST_SHARE_DIR
$CWD
$CWD/../share/kfdtest
/opt/rocm/share/kfdtest
EOF

# Keep these checks until automation starts using the package install
if [ -z "${FILTER[core]}" ]; then
    if [ -e "$CWD/../bin/kfdtest/kfdtest.exclude" ]; then
        source "$CWD/../bin/kfdtest/kfdtest.exclude"
    elif [ -e "$CWD/../../share/kfdtest.exclude" ]; then
        source "$CWD/../../share/kfdtest.exclude"
    fi
fi

# This filter will always exist if we sourced a valid kfdtest.exclude
if [ -z "${FILTER[core]}" ]; then
    echo "Unable to locate kfdtest.exclude."
    echo "Please set KFDTEST_SHARE_DIR or ensure that kfdtest.exclude is present inside $CWD, $CWD/../share/kfdtest or /opt/rocm/share/kfdtest"
    exit 1
fi

# Using "which" produces different results in different
# OSes so use command -v instead. It returns "" if the
# command isn't in the PATH
if [ -z "$(command -v kfdtest)" ]; then
    if [ -z "$BIN_DIR" ]; then
        if [ -e "${0%/*}/kfdtest" ]; then
            BIN_DIR="${0%/*}"
        else
            # The default location
            BIN_DIR="/opt/rocm/bin"
        fi
    fi
    if [ -e "$BIN_DIR/kfdtest" ]; then
        KFDTEST="$BIN_DIR/kfdtest"
    else
        echo "Unable to locate kfdtest."
        echo "Please set BIN_DIR, ensure that kfdtest is in $PATH, or ensure that kfdtest is present inside ${0%/*} or /opt/rocm/bin"
        exit 1
    fi
else
    KFDTEST="kfdtest"
fi

PLATFORM=""
GDB=""
NODE=""
CONCURRENTNODES=""
TESTNODENUM=""
RUN_IN_DOCKER=""
ADDITIONAL_EXCLUDE=""
COMPARTMENTALIZE=""
LOG_DIR=""
TEST_TIMEOUT=300
KMSG_MARKER="true"

printUsage() {
    echo
    echo "Usage: $(basename $0) [options ...] [gtest arguments]"
    echo
    echo "Options:"
    echo "  -p <platform> , --platform <platform>    Only run tests that"\
                               "pass on the specified platform. Usually you"\
                               "don't need this option"
    echo "  -g            , --gdb                    Run in debugger"
    echo "  -n <node(s)>  , --node <node(s)>         NodeId(s) to test. Takes a single integer, or a"\
                               "quoted, space-separated string as an argument"\
                               "(e.g. -n 1 OR -n \"1 2 3\")"\
                               "NOTE: Node numbers come from /sys/class/kfd/kfd/topology/nodes/#"
    echo "  -c            , --concurrentnodes        Concurrent nodes string for multi-gpu testing."\
                               "Takes a string comma-separated as an argument"\
                               "(e.g. -c \"1,2,3\" or --concurrentnodes \"1,2,3\")"\
                               "use -c \"all\" or --concurrentnodes \"all\" to test on all available nodes"\
                               "NOTE: Node numbers come from /sys/class/kfd/kfd/topology/nodes/#"
    echo "  -t            , --testnodenum            Number of concurrent nodes for multi-gpu testing."\
                               "Takes an integer as argument"\
                               "(e.g. -t 2 or --testnodenum 2)"
    echo "  -l            , --list                   List available nodes"
    echo "  --high                                   Force clocks to high for test execution (non-functional)"
    echo "  -d            , --docker                 Run in docker container"
    echo "  -e <list>     , --exclude <list>         Additional tests to exclude, in addition to kfdtest.exclude."\
                               "Takes a colon-separated string as an argument"\
                               "(e.g. -e KFDEvictTest.*:KFDSVMEvictTest.*)"
    echo "  -C            , --compartmentalize       Run every test in its own kfdtest process and store"\
                               "the kernel messages it produced in a log of its own"
    echo "  -o <dir>      , --logdir <dir>           Log directory for --compartmentalize"\
                               "(default: ./kfdtest_logs/<timestamp>)"
    echo "  --timeout <s>                            Kill a --compartmentalize test that runs longer"\
                               "than this, so the run continues (0 disables, default: 300)"
    echo "  --no-kmsg-marker                         Do not name each test in the kernel log."\
                               "Naming is on by default, needs passwordless sudo, and lets"\
                               "a panic be traced back to the test that caused it"
    echo "  -h            , --help                   Prints this help"
    echo
    echo "Gtest arguments will be forwarded to the app"
    echo
    echo "Valid platform options: core_sws, core, polaris10, vega10, vega20, pm, all, and so on"
    echo "'all' option runs all tests"

    return 0
}
# Print gtest_filter for the given Platform
#    param - Platform.
getFilter() {
# For regular platforms such as vega10, this will automatically generate
# the valid variable BLACKLIST based on the variable platform.
    local platform=$1;

    case "$platform" in
        all ) gtestFilter="" ;;
        * )
            if [ -z "${FILTER[$platform]}" ]; then
                echo "Unsupported platform $platform. Exiting"
                exit 1
            fi

            gtestFilter="--gtest_filter=${FILTER[$platform]}"
            ;;
    esac

    # Check if the loaded driver is upstream (in-box) or DKMS
    rdma_get_pages_func=$(cat /proc/kallsyms | grep rdma_get_pages || true)
    if [ -z "$rdma_get_pages_func" ]; then
        # If the filter is a blacklist (test list starts with -), we want to add to the list
        # If the filter is a whitelist (test list starts with the test name), we don't want to add
        # known-unsupported tests to the list, so don't add anything
        if [[ "$gtestFilter" == --gtest_filter=-* ]]; then
            gtestFilter="$gtestFilter:${FILTER[upstream]}"
        fi
    fi

    if [ -n "$ADDITIONAL_EXCLUDE" ]; then
        # If the filter is a blacklist (test list starts with -), we want to add to the list
	# If the filter is a whitelist (test list starts with the test name), we don't want to add
	# excluded tests to the list, so don't add anything
	# TODO: Add parsing so we can use --gtest_filter and -e together.
        if [[ "$gtestFilter" == --gtest_filter=-* ]]; then
            gtestFilter="$gtestFilter:$ADDITIONAL_EXCLUDE"
        fi
    fi
}

TOPOLOGY_SYSFS_DIR=/sys/devices/virtual/kfd/kfd/topology/nodes

# Prints list of HSA Nodes. HSA Nodes are identified from sysfs KFD topology. The nodes
# should have valid SIMD count
getHsaNodes() {
    for i in $(find $TOPOLOGY_SYSFS_DIR  -maxdepth 1 -mindepth 1 -type d); do
        simdcount=$(cat $i/properties | grep simd_count | awk '{print $2}')
        if [ $simdcount != 0 ]; then
            hsaNodeList+="$(basename $i) "
        fi
    done
    echo "$hsaNodeList"
}


# Prints GPU Name for the given Node ID. If transitioned to IP discovery,
# use target gfx version
#   param - Node ID
getNodeName() {
    local nodeId=$1; shift;
    local gpuName=$(cat $TOPOLOGY_SYSFS_DIR/$nodeId/name)
    if [ "$gpuName" == "raven" ]; then
      local CpuCoresCount=$(cat $TOPOLOGY_SYSFS_DIR/$nodeId/properties | grep cpu_cores_count | awk '{print $2}')
      local SimdCount=$(cat $TOPOLOGY_SYSFS_DIR/$nodeId/properties | grep simd_count | awk '{print $2}')
      if [ "$CpuCoresCount" -eq 0 ] && [ "$SimdCount" -gt 0 ]; then
        gpuName="raven_dgpuFallback"
      fi
    elif [ "$gpuName" == "ip discovery" ]; then
      if [ -n "$HSA_OVERRIDE_GFX_VERSION" ]; then
          gpuName="gfx$(echo "$HSA_OVERRIDE_GFX_VERSION" | awk 'BEGIN {FS="."; RS=""} {printf "%d%x%x", $1, $2, $3 }')"
      else
          local GfxVersionDec=$(cat $TOPOLOGY_SYSFS_DIR/$nodeId/properties | grep gfx_target_version | awk '{print $2}')
          if [[ ${#GfxVersionDec} = 5 ]]; then
              GfxVersionDec="0${GfxVersionDec}"
          fi
          gpuName="gfx$(printf "$GfxVersionDec" | fold -w2 | awk 'BEGIN {FS="\n"; RS=""} {printf "%d%x%x", $1, $2, $3}')"
      fi
    fi
    echo "$gpuName"
}

printGpuNodelist() {
    local hsaNodes=$(getHsaNodes)
    for node in $hsaNodes; do
        local name=$(getNodeName $node)
        echo "Node $node: $name"
    done
}

###############################################################################
# Compartmentalized execution (--compartmentalize)
#
# Every test runs in its own kfdtest process so the kernel messages it produced
# get a file of their own, next to a full log of the whole run.
#
# Readers in order: journalctl --kernel, dmesg, sudo dmesg. The journal is first
# because kernel.dmesg_restrict hides dmesg from unprivileged users on Ubuntu,
# Fedora and RHEL. One follower runs for the whole session; each test's slice is
# cut from its output by a cat sharing descriptor 3's offset.
###############################################################################

KLOG_MODE=""              # journalctl | dmesg | sudo dmesg | none
KLOG_FOLLOW_CMD=()        # argv of the follower detectKlogMode picked
KLOG_STREAM=""            # file the follower appends to, kernel-log.full
KLOG_PID=""               # follower pid, empty when there is none
KLOG_LAST_SIZE=0          # stream size at the previous collection
KLOG_STEP=0.05            # poll interval while waiting for the log to go quiet
KLOG_MAX_POLLS=30         # 1.5s cap, so a message storm cannot stall the run
TEST_LIST=()              # test names matching the current filter
KMSG_OK=""                # set when markers can actually be written
TIMEOUT_CMD=()            # timeout(1) prefix, empty when disabled
COMPART_TOTAL=0
COMPART_FAILED=0
COMPART_FAILED_LIST=()
COMPART_RUN_FAILED=0      # failures of the most recent runCompartmentalized call

#   param - message
compartFatal() {
    echo "$1"
    exit 1
}

# A hung test would otherwise stall the run. --kill-after handles one that
# ignores SIGTERM; --foreground leaves Ctrl-C working.
initTimeout() {
    [ "$TEST_TIMEOUT" -gt 0 ] || return 0
    command -v timeout >/dev/null 2>&1 ||
        { echo "WARNING: timeout(1) not found, tests will run unbounded."; return 0; }
    TIMEOUT_CMD=(timeout --foreground --kill-after=10 "$TEST_TIMEOUT")
}

# Probe by writing the session marker: if that works, per-test markers will too.
initKmsgMarker() {
    [ -n "$KMSG_MARKER" ] || return 0
    if kmsgMark "compartmentalized run starting"; then
        KMSG_OK="true"
    else
        echo "NOTE: no passwordless sudo, so tests are not named in the kernel log and a"
        echo "      panic cannot be traced to one. --no-kmsg-marker silences this."
    fi
}

# Distinctive token: klogCollect uses it to spot a marker-only window.
#   param - text to place in the kernel log
kmsgMark() {
    echo "kfdtest-marker: $1" | sudo -n tee /dev/kmsg > /dev/null 2>&1
}

# Pick a kernel log reader and build the command that follows it.
detectKlogMode() {
    local -a cmd=()
    local -a sudoPrefix=()

    if command -v journalctl >/dev/null 2>&1 &&
       [ -n "$(journalctl --kernel --lines 1 --no-pager 2>/dev/null)" ]; then
        KLOG_MODE="journalctl"
        cmd=(journalctl --kernel --follow --lines 0 --no-pager --output short-precise)
    elif [ -n "$(dmesg --time-format iso 2>/dev/null | tail -n 1)" ]; then
        KLOG_MODE="dmesg"
        cmd=(dmesg --follow-new --time-format iso)
    elif sudo -n true 2>/dev/null; then
        KLOG_MODE="sudo dmesg"
        sudoPrefix=(sudo -n)
        cmd=(dmesg --follow-new --time-format iso)
    else
        KLOG_MODE="none"
        return 0
    fi

    # Block buffering would land messages in the wrong test's log.
    command -v stdbuf >/dev/null 2>&1 && cmd=(stdbuf --output=L "${cmd[@]}")

    KLOG_FOLLOW_CMD=("${sudoPrefix[@]}" "${cmd[@]}")
}

startKlogFollower() {
    # Occupied even with no follower, so closing it for children is always valid.
    [ "$KLOG_MODE" == "none" ] && { exec 3< /dev/null; return 0; }

    KLOG_STREAM="$LOG_DIR/kernel-log.full"
    : > "$KLOG_STREAM"
    "${KLOG_FOLLOW_CMD[@]}" >> "$KLOG_STREAM" 2>> "$LOG_DIR/kernel-log.stderr" &
    KLOG_PID=$!
    exec 3< "$KLOG_STREAM"

    # Let the follower attach so the first test's messages are not missed.
    sleep 0.5
    if ! kill -0 "$KLOG_PID" 2>/dev/null; then
        echo "WARNING: kernel log follower ($KLOG_MODE) exited immediately."
        echo "         See $LOG_DIR/kernel-log.stderr. Continuing without kernel logs."
        KLOG_MODE="none"
        KLOG_PID=""
        return 0
    fi

    # Drop the attach-time backlog a dmesg without --follow-new would replay.
    cat <&3 > /dev/null
    KLOG_LAST_SIZE=$(stat --format=%s "$KLOG_STREAM" 2>/dev/null)
}

stopKlogFollower() {
    [ -n "$KLOG_PID" ] || return 0
    kill "$KLOG_PID" 2>/dev/null
    wait "$KLOG_PID" 2>/dev/null
    KLOG_PID=""
}

# Move everything the follower appended since the previous call into $1.
#   param - destination file
klogCollect() {
    [ "$KLOG_MODE" == "none" ] && return 0

    local cur prev i
    cur=$(stat --format=%s "$KLOG_STREAM" 2>/dev/null)

    # Most tests log nothing, so leaving here keeps them at one stat and no file.
    [ "$cur" == "$KLOG_LAST_SIZE" ] && return 0

    # Messages arrive asynchronously, so one logged during the test can land
    # after it exits. Wait for quiet, or it is billed to the next test.
    for ((i = 0; i < KLOG_MAX_POLLS; i++)); do
        sleep "$KLOG_STEP"
        prev="$cur"
        cur=$(stat --format=%s "$KLOG_STREAM" 2>/dev/null)
        [ "$cur" == "$prev" ] && break
    done

    # Descriptor 3 shares its offset with cat, resuming where the last call ended.
    cat <&3 > "$1"
    KLOG_LAST_SIZE=$cur

    # A window holding only our marker means the test itself was silent.
    grep -qv 'kfdtest-marker:' "$1" || rm -f "$1"
}

# Fill TEST_LIST with the tests kfdtest would run. Parameterized suites use an
# empty prefix, so those names start with a slash: /KFDSVMRangeTest.Basic/0.
#   param - node id
#   param - gtest filter argument, may be empty
listTests() {
    local node=$1 filter=$2 list

    # A suite header is an unindented "Suite." line; a test is an indented line
    # after one. $1 drops the "# GetParam() = ..." trailer. Patterns are strings
    # because mawk and gawk differ on an escaped slash in a bracket expression.
    list=$($KFDTEST "--node=$node" $filter $GTEST_ARGS --gtest_list_tests 2>/dev/null | awk '
        BEGIN {
            header = "^[A-Za-z_/][A-Za-z0-9_/]*[.]$"
            tname  = "^[A-Za-z0-9_/]+$"
        }
        /^[^ \t]/ { if ($1 ~ header) suite = $1; next }
        /^[ \t]/  { if (suite != "" && $1 ~ tname) print suite $1 }')

    TEST_LIST=()
    [ -n "$list" ] && mapfile -t TEST_LIST <<< "$list"
    return 0
}

resolveLogDir() {
    local stamp
    printf -v stamp '%(%Y%m%d-%H%M%S)T' -1
    [ -n "$LOG_DIR" ] || LOG_DIR="$PWD/kfdtest_logs/$stamp"
    mkdir -p "$LOG_DIR" 2>/dev/null ||
        compartFatal "Unable to create log directory $LOG_DIR, pick another with -o"
}

printCompartmentalizeNotes() {
    echo
    echo "NOTE: Tests run one per process, so a failure that depends on state left"
    echo "      by an earlier test will not reproduce here, and collecting kernel"
    echo "      logs between tests shifts timing. Use a normal run for sign-off."

    if [ "$KLOG_MODE" == "none" ]; then
        echo
        echo "WARNING: No readable kernel log source; kernel messages will not be"
        echo "         captured. Any one of these fixes it:"
        echo "           sudo usermod -aG systemd-journal $USER  (adm or wheel also work)"
        echo "           sudo sysctl -w kernel.dmesg_restrict=0"
        echo "           passwordless sudo for this user"
    fi
    echo
}

printCompartmentalizeSummary() {
    echo "tests run:      $COMPART_TOTAL"
    echo "passed:         $((COMPART_TOTAL - COMPART_FAILED))"
    echo "failed:         $COMPART_FAILED"
    if [ "$COMPART_FAILED" -ne 0 ]; then
        echo
        echo "failed tests:"
        printf '  %s\n' "${COMPART_FAILED_LIST[@]}"
    fi
    echo
    echo "Logs in $LOG_DIR"
}

compartmentalizeCleanup() {
    stopKlogFollower
    # kernel-log.stderr, and kernel-log.full on a quiet run, are usually empty.
    [ -n "$LOG_DIR" ] && find "$LOG_DIR" -type f -empty -delete 2>/dev/null
    return 0
}

checkCompartmentalizeArgs() {
    # Rejected rather than silently leaving the run unbounded.
    [[ "$TEST_TIMEOUT" =~ ^[0-9]+$ ]] ||
        compartFatal "--timeout expects a non-negative integer number of seconds."
    [ -z "$GDB" ] ||
        compartFatal "--compartmentalize cannot be combined with --gdb."
    [ "$RUN_IN_DOCKER" != "true" ] ||
        compartFatal "--compartmentalize cannot be combined with --docker."
    [[ "$GTEST_ARGS" != *--gtest_filter* || -z "$ADDITIONAL_EXCLUDE" ]] ||
        compartFatal "Cannot use -e and --gtest_filter flags together"
}

# Set nodeName from a node id, honouring -p, and fill TEST_LIST from its filter.
#   param - node id
prepareTestList() {
    nodeName=$(getNodeName $1)
    if [ "$PLATFORM" != "" ] && [ "$PLATFORM" != "$nodeName" ]; then
        echo "WARNING: Actual ASIC $nodeName treated as $PLATFORM"
        nodeName="$PLATFORM"
    fi
    getFilter $nodeName
    listTests "$1" "$gtestFilter"
}

# Run every test in TEST_LIST on its own. Failures land in COMPART_RUN_FAILED.
#   param - node selection argument for kfdtest
#   param - directory for this pass's logs
#   param - how to name this pass in messages
runCompartmentalized() {
    local nodeArg=$1 dir=$2 label=$3
    local total=${#TEST_LIST[@]}
    local width=${#total} index=0
    local testName safeName klogFile rc klines status

    COMPART_RUN_FAILED=0

    echo ""
    echo "++++ Starting compartmentalized testing $label: $total test(s) ++++"
    [ "$total" -eq 0 ] && { echo "No test matched the filter for $label"; return 0; }

    mkdir -p "$dir"

    for testName in "${TEST_LIST[@]}"; do
        index=$((index + 1))
        safeName=${testName#/}
        safeName=${safeName//\//_}
        # Numbered by execution order, since silent tests leave no file.
        printf -v klogFile '%s/%0*d.%s.klog' "$dir" "$width" "$index" "$safeName"

        printf '\n[%*d/%d] %s\n' "$width" "$index" "$total" "$testName"

        # In the ring buffer before the test, so a panic points back here.
        [ -n "$KMSG_OK" ] && kmsgMark "[$index/$total] $label $testName"

        # Filter last so it beats any --gtest_filter in GTEST_ARGS; 3 is ours.
        "${TIMEOUT_CMD[@]}" $KFDTEST "$nodeArg" $GTEST_ARGS "--gtest_filter=$testName" 3<&-
        rc=$?

        klogCollect "$klogFile"

        # Excludes our marker; 0 when the file was pruned or never written.
        klines=$(grep -cv 'kfdtest-marker:' "$klogFile" 2>/dev/null) || klines=0

        status="PASS"
        if [ "$rc" -ne 0 ]; then
            status="FAIL"
            # 124 is how timeout(1) reports that it had to intervene.
            [ "$rc" -eq 124 ] && status="TIMEOUT"
            COMPART_RUN_FAILED=$((COMPART_RUN_FAILED + 1))
            COMPART_FAILED=$((COMPART_FAILED + 1))
            COMPART_FAILED_LIST+=("$label $testName ($status, exit $rc)")
        fi
        COMPART_TOTAL=$((COMPART_TOTAL + 1))

        printf '[%*d/%d] %s %s, %s kernel message(s)\n' \
            "$width" "$index" "$total" "$status" "$testName" "$klines"
    done

    echo "---- Finished $label: $((total - COMPART_RUN_FAILED)) passed," \
         "$COMPART_RUN_FAILED failed ----"
}

# Run each test of the filtered suite on its own, one kernel log per test.
runKfdTestCompartmentalized() {
    checkCompartmentalizeArgs

    local hsaNodes
    if [ "$NODE" == "" ]; then
        hsaNodes=$(getHsaNodes)
        [ -n "$hsaNodes" ] || compartFatal "No GPU found in the system."
    else
        hsaNodes=$NODE
    fi

    resolveLogDir
    initTimeout
    detectKlogMode
    # Before the follower, so its session marker is not billed to test one.
    initKmsgMarker
    startKlogFollower
    printCompartmentalizeNotes

    trap 'echo; echo "Interrupted."; printCompartmentalizeSummary; compartmentalizeCleanup; exit 130' INT TERM

    echo "Logging to $LOG_DIR (kernel log: $KLOG_MODE)"

    local aggregate_fail=0 hsaNode nodeName

    # The multi-GPU selectors hand every node to one invocation, so there is a
    # single pass. As in a normal run, the filter comes from the first node.
    if [ -n "$CONCURRENTNODES" ]; then
        prepareTestList "${hsaNodes%% *}"
        runCompartmentalized "--concurrentnodes=$CONCURRENTNODES" \
            "$LOG_DIR/concurrentnodes-${CONCURRENTNODES//,/_}" \
            "node(s) $CONCURRENTNODES"
        aggregate_fail=$COMPART_RUN_FAILED
    elif [ -n "$TESTNODENUM" ]; then
        prepareTestList "${hsaNodes%% *}"
        runCompartmentalized "--testnodenum=$TESTNODENUM" \
            "$LOG_DIR/testnodenum-$TESTNODENUM" "$TESTNODENUM node(s)"
        aggregate_fail=$COMPART_RUN_FAILED
    else
        for hsaNode in $hsaNodes; do
            prepareTestList "$hsaNode"
            runCompartmentalized "--node=$hsaNode" \
                "$LOG_DIR/node$hsaNode-$nodeName" "node $hsaNode ($nodeName)"
            aggregate_fail=$((aggregate_fail + COMPART_RUN_FAILED))
        done
    fi

    trap - INT TERM
    echo ""
    printCompartmentalizeSummary
    compartmentalizeCleanup

    # An exit status is one byte; 256 failures must not look like a clean run.
    [ "$aggregate_fail" -gt 255 ] && aggregate_fail=255
    exit $aggregate_fail
}

# Run KfdTest independently. Two global variables set by command-line
# will influence the tests as indicated below
#   PLATFORM - If set all tests will run with this platform filter
#   NODE - If set tests will be run only on this NODE, else it will be
#           run on all available HSA Nodes
runKfdTest() {
    if [ "$RUN_IN_DOCKER" == "true" ]; then
        if [ `sudo systemctl is-active docker` != "active" ]; then
            echo "docker isn't active, install and setup docker first!!!!"
            exit 0
        fi
        PKG_ROOT="$(getPackageRoot)"
    fi

    if [[ "$GTEST_ARGS" =~ "--gtest_filter" && -n "$ADDITIONAL_EXCLUDE" ]]; then
        echo "Cannot use -e and --gtest_filter flags together"
        exit 0
    fi

    if [ "$NODE" == "" ]; then
        hsaNodes=$(getHsaNodes)

        if [ "$hsaNodes" == "" ]; then
            echo "No GPU found in the system."
            exit 1
        fi
    else
        hsaNodes=$NODE
    fi

    local aggregate_fail=0
    for hsaNode in $hsaNodes; do
        nodeName=$(getNodeName $hsaNode)
        if [ "$PLATFORM" != "" ] && [ "$PLATFORM" != "$nodeName" ]; then
            echo "WARNING: Actual ASIC $nodeName treated as $PLATFORM"
            nodeName="$PLATFORM"
        fi

        getFilter $nodeName

        if [ "$RUN_IN_DOCKER" == "true" ]; then
            if [ "$NODE" == "" ]; then
                DEVICE_NODE="/dev/dri"
            else
                RENDER_NODE=$(($hsaNode + 127))
                DEVICE_NODE="/dev/dri/renderD${RENDER_NODE}"
            fi

            echo "Starting testing node $hsaNode ($nodeName) in docker container"
            sudo docker run -it --name kfdtest_docker --user="jenkins" --network=host \
            --device=/dev/kfd --device=${DEVICE_NODE} --group-add video --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined -v $PKG_ROOT:/home/jenkins/rocm \
            compute-artifactory.amd.com:5000/yuho/tianli-ubuntu1604-kfdtest:01 \
            /home/jenkins/rocm/utils/run_kfdtest.sh -n $hsaNode $gtestFilter $GTEST_ARGS
            if [ "$?" = "0" ]; then
                echo "Finished node $hsaNode ($nodeName) successfully in docker container"
            else
                echo "Testing failed for node $hsaNode ($nodeName) in docker container"
                ((aggregate_fail+=1))
            fi
            sudo docker rm kfdtest_docker
        else
            if [ -n "$CONCURRENTNODES" ]; then
                echo "++++ Starting parallel testing on node(s) $CONCURRENTNODES  ++++"
                $GDB $KFDTEST "--concurrentnodes=$CONCURRENTNODES" $gtestFilter $GTEST_ARGS
                if [ "$?" != "0" ]; then
                    ((aggregate_fail+=1))
                fi
                echo "++++ Finished parallel testing on node(s) $CONCURRENTNODES  ++++"
                exit $aggregate_fail;
            elif [ -n "$TESTNODENUM" ]; then
                echo "++++ Starting parallel testing on $TESTNODENUM node(s) ++++"
                $GDB $KFDTEST "--testnodenum=$TESTNODENUM" $gtestFilter $GTEST_ARGS
                if [ "$?" != "0" ]; then
                    ((aggregate_fail+=1))
                fi
                echo "++++ Finished parallel testing on $TESTNODENUM node(s) ++++"
                exit $aggregate_fail;
            else
                echo ""
                echo "++++ Starting testing node $hsaNode ($nodeName) ++++"
                $GDB $KFDTEST "--node=$hsaNode" $gtestFilter $GTEST_ARGS
                if [ "$?" != "0" ]; then
                    ((aggregate_fail+=1))
                fi
                echo "---- Finished testing node $hsaNode ($nodeName) ----"
            fi
        fi


    done
    if [ $aggregate_fail -ne 0 ]; then
        echo "NOTE: $aggregate_fail nodes failed at least one test"
    fi
    exit $aggregate_fail
}

# Prints number of GPUs present in the system
getGPUCount() {
    gNodes=$(getHsaNodes)
    gNodes=( $gNodes )
    gpuCount=${#gNodes[@]}
    echo "$gpuCount"
}

while [ "$1" != "" ]; do
    case "$1" in
        -p  | --platform )
            shift 1; PLATFORM=$1 ;;
        -g  | --gdb )
            GDB="gdb --args" ;;
        -l  | --list )
            printGpuNodelist; exit 0 ;;
        -n  | --node )
            shift 1; NODE=$1 ;;
        -c  | --concurrentnodes )
            shift 1; CONCURRENTNODES="$1" ;;
        -t  | --testnodenum )
            shift 1; TESTNODENUM="$1" ;;
        --high)
            echo "--high flag is no longer functional. Flag kept for backwards-compatibility" ;;
        -d  | --docker )
            RUN_IN_DOCKER="true" ;;
        -e  | --exclude )
            shift 1; ADDITIONAL_EXCLUDE="$1" ;;
        -C  | --compartmentalize )
            COMPARTMENTALIZE="true" ;;
        -o  | --logdir )
            shift 1; LOG_DIR="$1" ;;
        --timeout )
            shift 1; TEST_TIMEOUT="$1" ;;
        --no-kmsg-marker )
            KMSG_MARKER="" ;;
    	-h  | --help )
            printUsage; exit 0 ;;
        *)
            GTEST_ARGS=$@; break;;
    esac
    shift 1
done

if [ "$CONCURRENTNODES" == "all" ]; then
    validNodes=$(getHsaNodes)
    CONCURRENTNODES=$(echo $validNodes | tr ' ' ',') 
else
    validNodes=$(getHsaNodes)
    validNodesArray=($validNodes)
    IFS=',' read -ra concurrentNodesArray <<< "$CONCURRENTNODES"

    for concurrentNode in "${concurrentNodesArray[@]}"; do
        if [[ ! " ${validNodesArray[@]} " =~ " $concurrentNode " ]]; then
            echo "Error: Invalid node $concurrentNode specified in --concurrentnodes."
            echo "Valid nodes are: $validNodes"
            exit 1
        fi
    done
fi

# Set HSA_DEBUG env to run KFDMemoryTest.PtraceAccessInvisibleVram
export HSA_DEBUG=1
if [ "$COMPARTMENTALIZE" == "true" ]; then
    runKfdTestCompartmentalized
else
    runKfdTest
fi
