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
import subprocess
import shutil
import os
import io
import selectors
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
            logging.error("ERROR: Cannot find VERSION.sha file at {}".format(shaFile))
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
    # rocprof info
    if not "ROCPROF" in os.environ.keys():
        rocprof_cmd = "rocprof"
    else:
        rocprof_cmd = os.environ["ROCPROF"]
    rocprof_path = shutil.which(rocprof_cmd)

    # TODO: this could be more elegant, clean code later
    if not rocprof_path:
        rocprof_cmd = "rocprof"
        rocprof_path = shutil.which(rocprof_cmd)

    if not rocprof_path:
        logging.error("\nError: Unable to resolve path to %s binary" % rocprof_cmd)
        logging.error(
            "Please verify installation or set ROCPROF environment variable with full path."
        )
        sys.exit(1)
    else:
        # Resolve any sym links in file path
        rocprof_path = os.path.realpath(rocprof_path.rstrip("\n"))
        logging.info("ROC Profiler: " + str(rocprof_path))
        return rocprof_cmd #TODO: Do we still need to return this? It's not being used in the function call

def capture_subprocess_output(subprocess_args):
    """Run specified subprocess and concurrently capture output
    """
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = subprocess.Popen(subprocess_args,
                            bufsize=1,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True)

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

def run_prof(fname, workload_dir, perfmon_dir, cmd, target, verbose):

    fbase = os.path.splitext(os.path.basename(fname))[0]

    logging.debug("pmc file:", os.path.basename(fname))

    # profile the app (run w/ custom config files for mi100)
    if target == "mi100":
        logging.info("RUNNING WITH CUSTOM METRICS")
        success, output = capture_subprocess_output(
            [
                rocprof_cmd,
                "-i",
                fname,
                "-m",
                perfmon_dir + "/" + "metrics.xml",
                "--timestamp",
                "on",
                "-o",
                workload_dir + "/" + fbase + ".csv",
                '"' + cmd + '"',
            ]
        )
    else:
        success, output = capture_subprocess_output(
            [
                rocprof_cmd,
                "-i",
                fname,
                "--timestamp",
                "on",
                "-o",
                workload_dir + "/" + fbase + ".csv",
                '"' + cmd + '"',
            ]
        )
    # write rocprof output to logging
    logging.info(output)

def replace_timestamps(workload_dir):
    df_stamps = pd.read_csv(workload_dir + "/timestamps.csv")
    if "BeginNs" in df_stamps.columns and "EndNs" in df_stamps.columns:
        # Update timestamps for all *.csv output files
        for fname in glob.glob(workload_dir + "/" + "*.csv"):
            df_pmc_perf = pd.read_csv(fname)

            df_pmc_perf["BeginNs"] = df_stamps["BeginNs"]
            df_pmc_perf["EndNs"] = df_stamps["EndNs"]
            df_pmc_perf.to_csv(fname, index=False)
    else:
        warning = "WARNING: Incomplete profiling data detected. Unable to update timestamps."
        logging.warning(warning + "\n")

def gen_sysinfo(workload_name, workload_dir, ip_blocks, app_cmd, skip_roof):
    # Record system information
    mspec = specs.get_machine_specs(0)
    sysinfo = open(workload_dir + "/" + "sysinfo.csv", "w")

    # write header
    header = "workload_name,"
    header += "command,"
    header += "host_name,host_cpu,host_distro,host_kernel,host_rocmver,date,"
    header += "gpu_soc,numSE,numCU,numSIMD,waveSize,maxWavesPerCU,maxWorkgroupSize,"
    header += "L1,L2,sclk,mclk,cur_sclk,cur_mclk,L2Banks,LDSBanks,name,numSQC,hbmBW,"
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
        mspec.cpu,
        mspec.distro,
        mspec.kernel,
        mspec.rocmversion,
        timestamp,
    ]

    # GPU info
    param += [
        mspec.GPU,
        mspec.SE,
        mspec.CU,
        mspec.SIMD,
        mspec.wave_size,
        mspec.wave_occu,
        mspec.workgroup_size,
    ]
    param += [
        mspec.L1,
        mspec.L2,
        mspec.SCLK,
        mspec.cur_MCLK,
        mspec.cur_SCLK,
        mspec.cur_MCLK,
    ]

    blocks = []
    hbmBW = int(mspec.cur_MCLK) / 1000 * 4096 / 8 * 2
    if mspec.GPU == "gfx906":
        param += ["16", "32", "mi50", str(int(mspec.CU) // 4), str(hbmBW)]
    elif mspec.GPU == "gfx908":
        param += ["32", "32", "mi100", "48", str(hbmBW)]
    elif mspec.GPU == "gfx90a":
        param += ["32", "32", "mi200", "56", str(hbmBW)]
        if not skip_roof:
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
    rocm_ver = mspec.rocmversion[:1]

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
    