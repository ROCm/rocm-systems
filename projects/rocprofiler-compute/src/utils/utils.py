##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import logging
import sys
import os
import io
import re
import selectors
import subprocess
import shutil
import pandas as pd
import glob
from utils import specs
from datetime import datetime
from pathlib import Path as path
import config

rocprof_cmd = ""

def demarcate(function):
    def wrap_function(*args, **kwargs):
        logging.trace("----- [entering function] -> %s()" % (function.__qualname__))
        result = function(*args, **kwargs)
        logging.trace("----- [exiting  function] -> %s()" % function.__qualname__)
        return result
    return wrap_function

def error(message):
    logging.error("")
    logging.error("[ERROR]: " + message)
    logging.error("")
    sys.exit(1)

def trace_logger(message, *args, **kwargs):
    logging.log(logging.TRACE, message, *args, **kwargs)

def get_version(omniperf_home) -> dict:
    """Return Omniperf versioning info
    """
    # symantic version info
    version = os.path.join(omniperf_home.parent, "VERSION")
    try:
        with open(version, "r") as file:
            VER = file.read().replace("\n", "")
    except EnvironmentError:
        logging.critical("ERROR: Cannot find VERSION file at {}".format(version))
        sys.exit(1)

    # git version info
    gitDir = os.path.join(omniperf_home.parent, ".git")
    if (shutil.which("git") is not None) and os.path.exists(gitDir):
        gitQuery = subprocess.run(
            ["git", "log", "--pretty=format:%h", "-n", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if gitQuery.returncode != 0:
            SHA = "unknown"
            MODE = "unknown"
        else:
            SHA = gitQuery.stdout.decode("utf-8")
            MODE = "dev"
    else:
        shaFile = os.path.join(omniperf_home.parent, "VERSION.sha")
        try:
            with open(shaFile, "r") as file:
                SHA = file.read().replace("\n", "")
        except EnvironmentError:
            error("Cannot find VERSION.sha file at {}".format(shaFile))
            sys.exit(1)

        MODE = "release"

    versionData = {"version": VER, "sha": SHA, "mode": MODE}
    return versionData

def get_version_display(version, sha, mode):
    """Pretty print versioning info
    """
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print("Omniperf version: %s (%s)" % (version, mode), file=buf)
    print("Git revision:     %s" % sha, file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()

def detect_rocprof():
    """Detect loaded rocprof version. Resolve path and set cmd globally.
    """
    global rocprof_cmd
    # detect rocprof
    if not "ROCPROF" in os.environ.keys():
        rocprof_cmd = "rocprof"
    else:
        rocprof_cmd = os.environ["ROCPROF"]
    
    # resolve rocprof path
    rocprof_path = shutil.which(rocprof_cmd)

    if not rocprof_path:
        rocprof_cmd = "rocprof"
        logging.warning("Warning: Unable to resolve path to %s binary. Reverting to default." % rocprof_cmd)
        rocprof_path = shutil.which(rocprof_cmd)
        if not rocprof_path:
            error("Please verify installation or set ROCPROF environment variable with full path.")
    else:
        # Resolve any sym links in file path
        rocprof_path = os.path.realpath(rocprof_path.rstrip("\n"))
        logging.info("ROC Profiler: " + str(rocprof_path))
        return rocprof_cmd #TODO: Do we still need to return this? It's not being used in the function call

def capture_subprocess_output(subprocess_args, new_env=None):
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if new_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=new_env,
        )
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream, mask):
        # Because the process' output is line buffered, there's only ever one
        # line to read when this function is called
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = (return_code == 0)

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)

def run_prof(fname, profiler_options, target, workload_dir):

    fbase = os.path.splitext(os.path.basename(fname))[0]
    m_specs = specs.get_machine_specs(0)
    
    logging.debug("pmc file:", os.path.basename(fname))

    # standard rocprof options
    default_options = [
        "-i", fname
    ]
    options = default_options + profiler_options

    # set required env var for mi300
    new_env = None
    if  (target.lower() == "mi300x_a0" or target.lower() == "mi300x_a1" or target.lower() == "mi300a_a0" or target.lower() == "mi300a_a1") and (
        os.path.basename(fname) == "pmc_perf_13.txt"
        or os.path.basename(fname) == "pmc_perf_14.txt"
        or os.path.basename(fname) == "pmc_perf_15.txt"
        or os.path.basename(fname) == "pmc_perf_16.txt"
    ):
        new_env = os.environ.copy()
        new_env["ROCPROFILER_INDIVIDUAL_XCC_MODE"] = "1"

    # profile the app
    if new_env:
         success, output = capture_subprocess_output(
            [ rocprof_cmd ] + options, new_env
        )
    else:
        success, output = capture_subprocess_output(
            [ rocprof_cmd ] + options
        )

    if not success:
        error(output)

    if new_env:
        # flatten tcc for applicable mi300 input
        f = path(workload_dir + "/out/pmc_1/results_" + fbase + ".csv")
        hbm_stack_num = get_hbm_stack_num(target, m_specs.memory_partition)
        df = flatten_tcc_info_across_hbm_stacks(
            f, hbm_stack_num, int(m_specs.L2Banks)
        )
        df.to_csv(f, index=False)

    if os.path.exists(workload_dir + "/out"):
        # copy and remove out directory if needed
        shutil.copyfile(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv",
            workload_dir + "/" + fbase + ".csv",
        )
        # Remove temp directory
        shutil.rmtree(workload_dir + "/" + "out")

    # Standardize rocprof headers via overwrite
    # {<key to remove>: <key to replace>}
    output_headers = {
        # ROCm-6.1.0 specific csv headers
        "KernelName": "Kernel_Name",
        "Index": "Dispatch_ID",
        "grd": "Grid_Size",
        "gpu-id": "GPU_ID",
        "wgr": "Workgroup_Size",
        "lds": "LDS_Per_Workgroup",
        "scr": "Scratch_Per_Workitem",
        "sgpr": "SGPR",
        "arch_vgpr": "Arch_VGPR",
        "accum_vgpr": "Accum_VGPR",
        "BeginNs": "Start_Timestamp",
        "EndNs": "End_Timestamp",
        # ROCm-6.0.0 specific csv headers
        "GRD": "Grid_Size",
        "WGR": "Workgroup_Size",
        "LDS": "LDS_Per_Workgroup",
        "SCR": "Scratch_Per_Workitem",
        "ACCUM_VGPR": "Accum_VGPR",
    }
    df = pd.read_csv(workload_dir + "/" + fbase + ".csv")
    df.rename(columns=output_headers, inplace=True)
    df.to_csv(workload_dir + "/" + fbase + ".csv", index=False)
    
    # write rocprof output to logging
    logging.info(output)

def replace_timestamps(workload_dir):
    df_stamps = pd.read_csv(workload_dir + "/timestamps.csv")
    if "Start_Timestamp" in df_stamps.columns and "End_Timestamp" in df_stamps.columns:
        # Update timestamps for all *.csv output files
        for fname in glob.glob(workload_dir + "/" + "*.csv"):
            if path(fname).name != "sysinfo.csv":
                df_pmc_perf = pd.read_csv(fname)

                df_pmc_perf["Start_Timestamp"] = df_stamps["Start_Timestamp"]
                df_pmc_perf["End_Timestamp"] = df_stamps["End_Timestamp"]
                df_pmc_perf.to_csv(fname, index=False)
    else:
        warning = "WARNING: Incomplete profiling data detected. Unable to update timestamps."
        logging.warning(warning + "\n")

def gen_sysinfo(workload_name, workload_dir, ip_blocks, app_cmd, skip_roof, roof_only):
    # Record system information
    mspec = specs.get_machine_specs(0)
    sysinfo = open(workload_dir + "/" + "sysinfo.csv", "w")

    # write header
    header = "workload_name,"
    header += "command,"
    header += "host_name,host_cpu,sbios,host_distro,host_kernel,host_rocmver,date,"
    header += "gpu_soc,vbios,numSE,numCU,numSIMD,waveSize,maxWavesPerCU,maxWorkgroupSize,"
    header += "L1,L2,sclk,mclk,cur_sclk,cur_mclk,L2Banks,LDSBanks,name,numSQC,hbmBW,compute_partition,memory_partition,"
    header += "ip_blocks\n"
    sysinfo.write(header)

    # timestamp
    now = datetime.now()
    local_now = now.astimezone()
    local_tz = local_now.tzinfo
    local_tzname = local_tz.tzname(local_now)
    timestamp = now.strftime("%c") + " (" + local_tzname + ")"
    # host info
    param = [workload_name]
    param += ['"' + app_cmd + '"']
    param += [
        mspec.hostname,
        mspec.CPU,
        mspec.sbios,
        mspec.distro,
        mspec.kernel_version,
        mspec.rocm_version,
        timestamp,
    ]

    # GPU info
    param += [
        mspec.arch,
        mspec.vbios,
        mspec.SE,
        mspec.CU,
        mspec.SIMD,
        mspec.wave_size,
        mspec.max_waves_per_cu,
        mspec.workgroup_max_size,
    ]
    param += [
        mspec.L1,
        mspec.L2,
        mspec.cur_mclk,
        mspec.cur_mclk,
        mspec.cur_sclk,
        mspec.cur_mclk,
        mspec.L2Banks,
        mspec.LDSBanks,
        mspec.GPU,
        mspec.numSQC,
        mspec.hbmBW,
        mspec.compute_partition,
        mspec.memory_partition,
    ]

    blocks = []
    if mspec.GPU == "gfx90a" and (not skip_roof):
        blocks.append("roofline")

    # ip block info
    if ip_blocks == None:
        t = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
        blocks += t
    else:
        blocks += ip_blocks
    param.append("|".join(blocks))

    sysinfo.write(",".join(param))
    sysinfo.close()

def detect_roofline():
    mspec = specs.get_machine_specs(0)
    rocm_ver = mspec.rocm_version[:1]

    os_release = path("/etc/os-release").read_text()
    ubuntu_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)
    rhel_distro = specs.search(r'PLATFORM_ID="(.*?)"', os_release)
    sles_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)

    if "ROOFLINE_BIN" in os.environ.keys():
        rooflineBinary = os.environ["ROOFLINE_BIN"]
        if os.path.exists(rooflineBinary):
            logging._SysExcInfoType("Detected user-supplied binary")
            return {"rocm_ver": "override", "distro": "override", "path": rooflineBinary}
        else:
            logging.error("ROOFLINE ERROR: user-supplied path to binary not accessible")
            logging.error("--> ROOFLINE_BIN = %s\n" % target_binary)
            sys.exit(1)
    elif rhel_distro == "platform:el8" or rhel_distro == "platform:el9":
        # Must be a valid RHEL machine
        distro = "platform:el8"
    elif (
        (type(sles_distro) == str and len(sles_distro) >= 3) and # confirm string and len
        sles_distro[:2] == "15" and int(sles_distro[3]) >= 3 # SLES15 and SP >= 3
    ):
        # Must be a valid SLES machine
        # Use SP3 binary for all forward compatible service pack versions
        distro = "15.3"
    elif ubuntu_distro == "20.04":
        # Must be a valid Ubuntu machine
        distro = ubuntu_distro
    else:
        logging.error("ROOFLINE ERROR: Cannot find a valid binary for your operating system")
        sys.exit(1)

    target_binary = {"rocm_ver": rocm_ver, "distro": distro}
    return target_binary

def run_rocscope(args, fname):
    # profile the app
    if args.use_rocscope == True:
        result = shutil.which("rocscope")
        if result:
            rs_cmd = [
                result.stdout.decode("ascii").strip(),
                "metrics",
                "-p",
                args.path,
                "-n",
                args.name,
                "-t",   
                fname,
                "--",
            ]
            for i in args.remaining.split():
                rs_cmd.append(i)
            logging.info(rs_cmd)
            success, output = capture_subprocess_output(
                rs_cmd
            )
            if not success:
                logging.error(result.stderr.decode("ascii"))
                sys.exit(1)

def mibench(args):
    """Run roofline microbenchmark to generate peak BW and FLOP measurements.
    """
    logging.info("[roofline] No roofline data found. Generating...")
    distro_map = {"platform:el8": "rhel8", "15.3": "sle15sp3", "20.04": "ubuntu20_04"}

    target_binary = detect_roofline()
    if target_binary["rocm_ver"] == "override":
        path_to_binary = target_binary["path"]
    else:
        path_to_binary = (
            str(config.omniperf_home)
            + "/utils/rooflines/roofline"
            + "-"
            + distro_map[target_binary["distro"]]
            + "-"
            + args.target.lower()
            + "-rocm"
            + target_binary["rocm_ver"]
        )

    # Distro is valid but cant find rocm ver
    if not os.path.exists(path_to_binary):
        logging.error("ROOFLINE ERROR: Unable to locate expected binary (%s)." % path_to_binary)
        sys.exit(1)

    subprocess.run(
        [
            path_to_binary,
            "-o",
            args.path + "/" + "roofline.csv",
            "-d",
            str(args.device),
        ],
        check=True
    )

def flatten_tcc_info_across_hbm_stacks(file, stack_num, tcc_channel_per_stack):
    """
    Flatten TCC per channel counters across all HBM stacks in used.
    NB: This func highly depends on the default behavior of rocprofv2 on MI300,
        which might be broken anytime in the future!
    """
    df_orig = pd.read_csv(file)
    # display(df_orig.info)

    ### prepare column headers
    tcc_cols_orig = []
    non_tcc_cols_orig = []
    for c in df_orig.columns.to_list():
        if "TCC" in c:
            tcc_cols_orig.append(c)
        else:
            non_tcc_cols_orig.append(c)
    # print(tcc_cols_orig)

    cols = non_tcc_cols_orig
    tcc_cols_in_group = {}
    for i in range(0, stack_num):
        tcc_cols_in_group[i] = []

    for col in tcc_cols_orig:
        for i in range(0, stack_num):
            # filter the channel index only
            p = re.compile(r"(\d+)")
            # pick up the 1st element only
            r = lambda match: str(int(float(match.group(0))) + i * tcc_channel_per_stack)
            tcc_cols_in_group[i].append(re.sub(pattern=p, repl=r, string=col))

    for i in range(0, stack_num):
        # print(tcc_cols_in_group[i])
        cols += tcc_cols_in_group[i]
    # print(cols)
    df = pd.DataFrame(columns=cols)

    ### Rearrange data with extended column names

    # print(len(df_orig.index))
    for idx in range(0, len(df_orig.index), stack_num):
        # assume the front none TCC columns are the same for all XCCs
        df_non_tcc = df_orig.iloc[idx].filter(regex=r"^(?!.*TCC).*$")
        # display(df_non_tcc)
        flatten_list = df_non_tcc.tolist()

        # extract all tcc from one dispatch
        # NB: assuming default contiguous order might not be safe!
        df_tcc_all = df_orig.iloc[idx : (idx + stack_num)].filter(regex="TCC")
        # display(df_tcc_all)

        for idx, row in df_tcc_all.iterrows():
            flatten_list += row.tolist()
        # print(len(df.index), len(flatten_list), len(df.columns), flatten_list)
        # NB: It is not the best perf to append a row once a time
        df.loc[len(df.index)] = flatten_list

    return df

def get_hbm_stack_num(gpu_name, memory_partition):
    """
    Get total HBM stack numbers based on  memory partition for MI300.
    """

    # TODO:
    # - better err log
    if gpu_name.lower() == "mi300a_a0" or gpu_name.lower() == "mi300a_a1":
        if memory_partition.lower() == "nps1":
            return 6
        elif memory_partition.lower() == "nps4":
            return 2
        elif memory_partition.lower() == "nps8":
            return 1
        else:
            print("Invalid MI300A memory partition mode!")
            sys.exit()
    elif gpu_name.lower() == "mi300x_a0" or gpu_name.lower() == "mi300x_a1":
        if memory_partition.lower() == "nps1":
            return 8
        elif memory_partition.lower() == "nps2":
            return 4
        elif memory_partition.lower() == "nps4":
            return 2
        elif memory_partition.lower() == "nps8":
            return 1
        else:
            print("Invalid MI300X memory partition mode!")
            sys.exit()
    else:
        # Fixme: add proper numbers for other archs
        return -1
    
def get_submodules(package_name):
    """List all submodules for a target package
    """
    import importlib
    import pkgutil

    submodules = []
    
    # walk all submodules in target package
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__):
        pretty_name = name.split("_", 1)[1].replace("_", "")
        # ignore base submodule, add all other
        if pretty_name != "base":
            submodules.append(pretty_name)

    return submodules
    