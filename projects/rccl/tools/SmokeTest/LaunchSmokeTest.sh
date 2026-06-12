#!/bin/bash

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#
# LaunchSmokeTest - Multi-rank RcclSmokeTest Socket Execution Script
#
# This script simplifies the execution of socket-based multi-rank RcclSmokeTest
# by automatically setting up SSH connections to specified hosts and setting
# the appropriate environment variables.
#
# Usage:
#   ./LaunchSmokeTest.sh <hosts> [env_vars...] [-- <smoketest_args>]
#
# Arguments:
#   hosts: Comma-separated list of hostnames/IPs to run on
#   env_vars: Optional environment variables (e.g., NCCL_DEBUG=INFO)
#   smoketest_args: Arguments to pass to RcclSmokeTest (after --)
#
# Special Environment Variables (intercepted by this script):
#   GPUS_PER_NODE      Number of GPUs (ranks) per node (default: auto-detected via amd-smi)
#   NCCL_COMM_ID_PORT  Override the bootstrap port (default: random free port)
#
# Examples:
#   ./LaunchSmokeTest.sh node0,node1
#   ./LaunchSmokeTest.sh node0,node1 GPUS_PER_NODE=8
#   ./LaunchSmokeTest.sh node0,node1 GPUS_PER_NODE=4 NCCL_DEBUG=INFO
#
# Notes:
#   - The first host in the list becomes rank 0 (master/bootstrap root)
#   - NCCL_COMM_ID is set to <first_host>:<port>, where port is random or
#     overridden via NCCL_COMM_ID_PORT
#   - RcclSmokeTest must be built in the same directory as this script on all hosts
#   - SSH access must be configured for all hosts
#   - Each host launches GPUS_PER_NODE processes (one per GPU, device 0..N-1)
#

set -e

show_usage() {
    cat << EOF
LaunchSmokeTest.sh — helper script for launching RcclSmokeTest across multiple
nodes without MPI. It sets up SSH connections to all specified hosts, auto-detects
GPU counts, collects per-node stack info, and launches one RcclSmokeTest process
per GPU per node.

NCCL_COMM_ID is managed automatically: the script selects a random free port on
he first host (or uses NCCL_COMM_ID_PORT if specified) and sets
NCCL_COMM_ID=<first_host>:<port> for all ranks, bootstrapping the communicator
without any additional setup required from the user.

Usage: $0 <hosts> [env_vars...] [-- <smoketest_args>]

Arguments:
  hosts              Comma-separated list of hostnames/IPs to run on
  env_vars           Optional environment variables (KEY=VALUE format)
  smoketest_args     Arguments to pass to RcclSmokeTest (after --)

Special Environment Variables (intercepted by this script):
  GPUS_PER_NODE      Number of GPUs (ranks) per node (default: auto-detected via amd-smi)
  NCCL_COMM_ID_PORT  Override the bootstrap port (default: random free port)
  NCCL_DEBUG         Blocked — the binary manages RCCL debug logging internally

Logging (handled by RcclSmokeTest, passed through):
  NCCL_LOGGING=0     Disable RCCL debug logging (default: enabled)
  NCCL_LOG_DIR=<dir> Directory for log files (default: ./logs)

Examples:
  $0 node0,node1
  $0 node0,node1 GPUS_PER_NODE=8
  $0 node0,node1 GPUS_PER_NODE=4 NCCL_LOG_DIR=/tmp/rccl-logs
  $0 node0,node1 NCCL_LOGGING=0

EOF
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    show_usage
    exit 1
fi

# Parse hosts
hosts_input="$1"
shift

if [[ -z "$hosts_input" ]]; then
    echo "ERROR: No hosts specified" >&2
    show_usage
    exit 1
fi

IFS=',' read -ra hosts_raw <<< "$hosts_input"
hosts=()
for host in "${hosts_raw[@]}"; do
    host=$(echo "$host" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [[ -z "$host" ]]; then
        echo "ERROR: Empty hostname found in host list" >&2
        exit 1
    fi
    if [[ "$host" =~ [[:space:]] ]]; then
        echo "ERROR: Hostname '$host' contains whitespace" >&2
        exit 1
    fi
    hosts+=("$host")
done
num_hosts=${#hosts[@]}

# Parse environment variables and smoketest arguments
env_vars=()
st_args=()
parsing_st_args=false

while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
        parsing_st_args=true
        shift
        continue
    fi

    if [[ $parsing_st_args == true ]]; then
        st_args+=("$1")
    elif [[ "$1" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
        env_vars+=("$1")
    else
        echo "ERROR: Invalid environment variable format: $1" >&2
        echo "Environment variables should be in KEY=VALUE format" >&2
        exit 1
    fi
    shift
done

# Extract special env vars (GPUS_PER_NODE, NCCL_COMM_ID_PORT, NCCL_DEBUG)
gpus_per_node_explicit=false
gpus_per_node=""
port=""
for env_var in "${env_vars[@]}"; do
    key="${env_var%%=*}"
    value="${env_var#*=}"
    if [[ "$key" == "GPUS_PER_NODE" ]]; then
        gpus_per_node="$value"
        gpus_per_node_explicit=true
    elif [[ "$key" == "NCCL_COMM_ID_PORT" ]]; then
        port="$value"
    elif [[ "$key" == "NCCL_DEBUG" ]]; then
        echo "WARNING: NCCL_DEBUG is managed internally by RcclSmokeTest and will be ignored." >&2
        echo "  Use NCCL_LOGGING=0 to disable logging, or NCCL_LOG_DIR=<dir> to set the log directory." >&2
    fi
done

# Filter env vars passed through (for display later)
pass_env=()
for env_var in "${env_vars[@]}"; do
    key="${env_var%%=*}"
    if [[ "$key" != "GPUS_PER_NODE" && "$key" != "NCCL_COMM_ID_PORT" && "$key" != "NCCL_DEBUG" ]]; then
        pass_env+=("$env_var")
    fi
done

# Resolve binary path relative to this script's directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
smoketest_path="$script_dir/build/RcclSmokeTest"

if [[ ! -x "$smoketest_path" ]]; then
    echo "ERROR: Binary not found or not executable: $smoketest_path" >&2
    echo "Build first: cd $script_dir && cmake -B build && cmake --build build -j\$(nproc)" >&2
    exit 1
fi

# Pick a random free port on the master host if not overridden
pick_free_port() {
    while true; do
        local p
        p=$(shuf -i 10000-65535 -n 1)
        if ! ss -tlnp 2>/dev/null | grep -q ":${p} "; then
            echo "$p"
            return
        fi
    done
}

if [[ -z "$port" ]]; then
    port=$(pick_free_port)
fi

master_host="${hosts[0]}"
nccl_comm_id="${master_host}:${port}"

"$smoketest_path" --version
echo "Running multi-node smoke test"
echo "Hosts     : ${hosts[*]}"
echo "EnvVars   : ${pass_env[*]:-none}"
if [[ ${#st_args[@]} -eq 0 ]]; then
    echo "Args      : none"
else
    echo "Args      : ${st_args[*]}"
fi
echo "CommID    : $nccl_comm_id"

# Determine and display NCCL log info (mirrors binary logic)
_nccl_logging="${NCCL_LOGGING:-1}"
# Check if user passed NCCL_LOGGING via env_vars
for _ev in "${env_vars[@]}"; do
    if [[ "${_ev%%=*}" == "NCCL_LOGGING" ]]; then _nccl_logging="${_ev#*=}"; fi
done
if [[ "$_nccl_logging" == "0" || "$_nccl_logging" == "off" || "$_nccl_logging" == "OFF" || "$_nccl_logging" == "false" ]]; then
    echo "NCCL log  : disabled"
else
    # Resolve log dir: user override takes precedence, then default to script dir
    _nccl_log_dir="${NCCL_LOG_DIR:-}"
    for _ev in "${env_vars[@]}"; do
        if [[ "${_ev%%=*}" == "NCCL_LOG_DIR" ]]; then _nccl_log_dir="${_ev#*=}"; fi
    done
    if [[ -z "$_nccl_log_dir" ]]; then
        _nccl_log_dir="$script_dir/logs"
    fi
    # Resolve to absolute path
    _nccl_log_dir="$(cd "$script_dir" && realpath -m "$_nccl_log_dir" 2>/dev/null || echo "$_nccl_log_dir")"
    echo "NCCL log  : $_nccl_log_dir/rccl_smoketest_<hostname>_rank<N>_<timestamp>.log"
fi
echo

# Collect stack info from a node via SSH
collect_node_info() {
    local host=$1
    ssh -q -o LogLevel=ERROR "$host" '
        rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
        gpu_count=$(amd-smi list 2>/dev/null | grep -c "^GPU:" || echo "0")
        gfx_ver=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | awk "{print \$NF}" || echo "unknown")
        gpu_model=$(rocm-smi --showproductname 2>/dev/null | grep "Card Series" | head -1 | sed "s/.*Card Series: *//" | xargs || echo "unknown")
        rccl_lib=$(basename "$(readlink -f /opt/rocm/lib/librccl.so 2>/dev/null)" 2>/dev/null | sed "s/librccl\.so\.//" || echo "unknown")
        cpu_count=$(lscpu 2>/dev/null | grep "^CPU(s):" | awk "{print \$2}" || echo "unknown")
        cpu_model=$(lscpu 2>/dev/null | grep "Model name" | sed "s/.*Model name[[:space:]]*:[[:space:]]*//" | xargs || echo "unknown")
        cpu_arch=$(lscpu 2>/dev/null | grep "^Architecture" | awk "{print \$2}" || echo "unknown")
        echo "$rocm_ver|$gpu_count|$gfx_ver|$gpu_model|$rccl_lib|$cpu_count|$cpu_model|$cpu_arch"
    '
}

# Warn if a field differs across nodes
warn_mismatch() {
    local field=$1; shift
    local first="$1"
    for val in "$@"; do
        if [[ "$val" != "$first" ]]; then
            echo "  WARNING: $field mismatch detected across nodes!" >&2
            return
        fi
    done
}

# Collect and display per-node stack info
declare -A _rocm _gpus _gfx _model _rccl _cpucount _cpumodel _cpuarch

echo "Node Stack Info:"
printf "  %-22s  %-8s  %-6s  %-8s  %-28s  %-14s  %-6s  %s\n" \
    "Host" "ROCm" "GPUs" "GFX" "GPU Model" "RCCL Lib" "CPUs" "CPU Model"
printf "  %s\n" "$(printf '%.0s-' {1..110})"

for host in "${hosts[@]}"; do
    info=$(collect_node_info "$host")
    IFS='|' read -r rocm gpus gfx model rccl cpucount cpumodel cpuarch <<< "$info"
    _rocm[$host]=$rocm
    _gpus[$host]=$gpus
    _gfx[$host]=$gfx
    _model[$host]=$model
    _rccl[$host]=$rccl
    _cpucount[$host]=$cpucount
    _cpumodel[$host]=$cpumodel
    _cpuarch[$host]=$cpuarch
    printf "  %-22s  %-8s  %-6s  %-8s  %-28s  %-14s  %-6s  %s\n" \
        "$host" "$rocm" "$gpus" "$gfx" "$model" "$rccl" "$cpucount" "$cpumodel ($cpuarch)"
done
echo

warn_mismatch "ROCm version" "${_rocm[@]}"
warn_mismatch "GPU count"    "${_gpus[@]}"
warn_mismatch "GFX arch"     "${_gfx[@]}"
warn_mismatch "GPU model"    "${_model[@]}"
warn_mismatch "RCCL library" "${_rccl[@]}"
warn_mismatch "CPU count"    "${_cpucount[@]}"
warn_mismatch "CPU model"    "${_cpumodel[@]}"
warn_mismatch "CPU arch"     "${_cpuarch[@]}"
echo

# Compute per-host GPU counts and total ranks
declare -A host_gpus
num_ranks=0
for host in "${hosts[@]}"; do
    if [[ "$gpus_per_node_explicit" == "true" ]]; then
        host_gpus[$host]=$gpus_per_node
    else
        local_gpus=${_gpus[$host]:-1}
        if [[ "$local_gpus" -le 0 ]]; then local_gpus=1; fi
        host_gpus[$host]=$local_gpus
    fi
    num_ranks=$(( num_ranks + host_gpus[$host] ))
done

# Build properly escaped environment variable string (excluding special vars)
env_string=""
for env_var in "${env_vars[@]}"; do
    key="${env_var%%=*}"
    value="${env_var#*=}"
    if [[ "$key" == "GPUS_PER_NODE" || "$key" == "NCCL_COMM_ID_PORT" || "$key" == "NCCL_DEBUG" ]]; then
        continue
    fi
    escaped_value=$(printf '%q' "$value")
    env_string="$env_string $key=$escaped_value"
done

# Inject absolute NCCL_LOG_DIR (so SSH processes don't use a relative path
# resolved from the SSH landing directory) and suppress the binary's log
# announcement since the script already printed it above.
if [[ "$_nccl_logging" != "0" && "$_nccl_logging" != "off" && "$_nccl_logging" != "OFF" && "$_nccl_logging" != "false" ]]; then
    # Only inject if user didn't already pass NCCL_LOG_DIR explicitly
    _user_set_log_dir=false
    for _ev in "${env_vars[@]}"; do
        if [[ "${_ev%%=*}" == "NCCL_LOG_DIR" ]]; then _user_set_log_dir=true; fi
    done
    if [[ "$_user_set_log_dir" == "false" ]]; then
        env_string="$env_string NCCL_LOG_DIR=$(printf '%q' "$_nccl_log_dir")"
    fi
fi
env_string="$env_string _RCCL_ST_LOG_QUIET=1"

# Cleanup function for interruption
cleanup() {
    echo >&2
    echo "Interrupted! Cleaning up processes..." >&2

    if [[ ${#worker_pids[@]} -gt 0 ]]; then
        for pid in "${worker_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done

        sleep 2

        for pid in "${worker_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing SSH PID $pid..." >&2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done

        for pid in "${worker_pids[@]}"; do
            timeout 1 bash -c "wait $pid" 2>/dev/null || true
        done
    fi

    # Kill remote RcclSmokeTest processes on all hosts
    for host in "${all_hosts[@]}"; do
        ssh -q -o LogLevel=ERROR -o ConnectTimeout=1 "$host" \
            "pkill -u \$(whoami) -f RcclSmokeTest 2>/dev/null || true" 2>/dev/null || true &
    done

    echo "Cleanup complete" >&2
    exit 130
}

trap cleanup INT TERM

# Build properly escaped smoketest arguments string
st_args_escaped=""
for arg in "${st_args[@]}"; do
    st_args_escaped+=" $(printf '%q' "$arg")"
done

# Build rank-to-host and rank-to-device mappings
worker_pids=()
worker_hosts=()
all_hosts=("${hosts[@]}")

rank=0
for ((n=0; n<num_hosts; n++)); do
    host="${hosts[$n]}"
    node_gpu_count=${host_gpus[$host]}
    for ((d=0; d<node_gpu_count; d++)); do
        if [[ $rank -eq 0 ]]; then
            # Rank 0 is launched last, in foreground
            master_device=$d
            ((++rank))
            continue
        fi
        worker_cmd="NCCL_COMM_ID=$nccl_comm_id $env_string '$smoketest_path' --nranks $num_ranks --rank $rank --device $d$st_args_escaped"
        ssh -q -o LogLevel=ERROR "$host" "$worker_cmd" >/dev/null 2>&1 &
        worker_pids+=($!)
        worker_hosts+=("$host")
        ((++rank))
    done
done

# Start master rank (rank 0) in foreground so its output is visible
master_cmd="NCCL_COMM_ID=$nccl_comm_id $env_string '$smoketest_path' --nranks $num_ranks --rank 0 --device ${master_device:-0}$st_args_escaped"
if ! ssh -q -o LogLevel=ERROR "$master_host" "$master_cmd"; then
    echo "ERROR: Master rank failed on $master_host" >&2
    cleanup
    exit 1
fi

# Check worker exit codes
any_worker_failed=false
for ((i=0; i<${#worker_pids[@]}; i++)); do
    if ! wait "${worker_pids[$i]}"; then
        echo "ERROR: Worker rank failed on ${worker_hosts[$i]}" >&2
        any_worker_failed=true
    fi
done

if [[ "$any_worker_failed" == "true" ]]; then
    exit 1
fi
