##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

import locale
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


def console_error(*argv, exit=True):
    if len(argv) > 1:
        logging.error(f"[{argv[0]}] {argv[1]}")
    else:
        logging.error(f"{argv[0]}")
    if exit:
        sys.exit(1)


def console_log(*argv, indent_level=0):
    indent = ""
    if indent_level >= 1:
        indent = " " * 3 * indent_level + "|-> "  # spaces per indent level

    if len(argv) > 1:
        logging.info(indent + f"[{argv[0]}] {argv[1]}")
    else:
        logging.info(indent + f"{argv[0]}")


def console_debug(*argv):
    if len(argv) > 1:
        logging.debug(f"[{argv[0]}] {argv[1]}")
    else:
        logging.debug(f"{argv[0]}")


def console_warning(*argv):
    if len(argv) > 1:
        logging.warning(f"[{argv[0]}] {argv[1]}")
    else:
        logging.warning(f"{argv[0]}")


def trace_logger(message, *args, **kwargs):
    logging.log(logging.TRACE, message, *args, **kwargs)


def get_version(rocprof_compute_home) -> dict:
    """Return ROCm Compute Profiler versioning info"""

    # symantic version info - note that version file(s) can reside in
    # two locations depending on development vs formal install
    searchDirs = [rocprof_compute_home, rocprof_compute_home.parent]
    found = False
    versionDir = None

    for dir in searchDirs:
        version = os.path.join(dir, "VERSION")
        try:
            with open(version, "r") as file:
                VER = file.read().replace("\n", "")
                found = True
                versionDir = dir
                break
        except:
            pass
    if not found:
        console_error("Cannot find VERSION file at {}".format(searchDirs))

    # git version info
    gitDir = os.path.join(rocprof_compute_home.parent, ".git")
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
        shaFile = os.path.join(versionDir, "VERSION.sha")
        try:
            with open(shaFile, "r") as file:
                SHA = file.read().replace("\n", "")
        except EnvironmentError:
            console_error("Cannot find VERSION.sha file at {}".format(shaFile))
            sys.exit(1)

        MODE = "release"

    versionData = {"version": VER, "sha": SHA, "mode": MODE}
    return versionData


def get_version_display(version, sha, mode):
    """Pretty print versioning info"""
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print("rocprofiler-compute version: %s (%s)" % (version, mode), file=buf)
    print("Git revision:     %s" % sha, file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()


def detect_rocprof():
    """Detect loaded rocprof version. Resolve path and set cmd globally."""
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
        console_warning(
            "Unable to resolve path to %s binary. Reverting to default." % rocprof_cmd
        )
        rocprof_path = shutil.which(rocprof_cmd)
        if not rocprof_path:
            console_error(
                "Please verify installation or set ROCPROF environment variable with full path."
            )
    else:
        # Resolve any sym links in file path
        rocprof_path = os.path.realpath(rocprof_path.rstrip("\n"))
        console_debug("ROC Profiler: " + str(rocprof_path))
        return rocprof_cmd  # TODO: Do we still need to return this? It's not being used in the function call


def capture_subprocess_output(subprocess_args, new_env=None, profileMode=False):
    console_debug("subprocess", subprocess_args)
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
        try:
            # Because the process' output is line buffered, there's only ever one
            # line to read when this function is called
            line = stream.readline()
            buf.write(line)
            if profileMode:
                console_log(rocprof_cmd, line.strip(), indent_level=1)
            else:
                console_log(line.strip())
        except UnicodeDecodeError:
            # Skip this line
            pass

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

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)


def run_prof(fname, profiler_options, workload_dir, mspec, loglevel):

    fbase = os.path.splitext(os.path.basename(fname))[0]

    console_debug("pmc file: %s" % str(os.path.basename(fname)))

    # standard rocprof options
    default_options = ["-i", fname]
    options = default_options + profiler_options

    # set required env var for mi300
    new_env = None
    if (
        mspec.gpu_model.lower() == "mi300x_a0"
        or mspec.gpu_model.lower() == "mi300x_a1"
        or mspec.gpu_model.lower() == "mi300a_a0"
        or mspec.gpu_model.lower() == "mi300a_a1"
    ) and (
        os.path.basename(fname) == "pmc_perf_13.txt"
        or os.path.basename(fname) == "pmc_perf_14.txt"
        or os.path.basename(fname) == "pmc_perf_15.txt"
        or os.path.basename(fname) == "pmc_perf_16.txt"
        or os.path.basename(fname) == "pmc_perf_17.txt"
    ):
        new_env = os.environ.copy()
        new_env["ROCPROFILER_INDIVIDUAL_XCC_MODE"] = "1"

    # profile the app
    if new_env:
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=new_env, profileMode=True
        )
    else:
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, profileMode=True
        )

    if not success:
        if loglevel > logging.INFO:
            for line in output.splitlines():
                console_error(output, exit=False)
        console_error("Profiling execution failed.")

    if rocprof_cmd.endswith("v2"):
        # rocprofv2 has separate csv files for each process
        results_files = glob.glob(workload_dir + "/out/pmc_1/results_*.csv")

        # Combine results into single CSV file
        combined_results = pd.concat(
            [pd.read_csv(f) for f in results_files], ignore_index=True
        )

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )

    if new_env:
        # flatten tcc for applicable mi300 input
        f = path(workload_dir + "/out/pmc_1/results_" + fbase + ".csv")
        xcds = total_xcds(mspec.gpu_model, mspec.compute_partition)
        df = flatten_tcc_info_across_xcds(f, xcds, int(mspec._l2_banks))
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
        console_warning(
            "Incomplete profiling data detected. Unable to update timestamps.\n"
        )


def gen_sysinfo(
    workload_name, workload_dir, ip_blocks, app_cmd, skip_roof, roof_only, mspec, soc
):
    df = mspec.get_class_members()

    # Append workload information to machine specs
    df["command"] = app_cmd
    df["workload_name"] = workload_name

    blocks = []
    if ip_blocks == None:
        t = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
        blocks += t
    else:
        blocks += ip_blocks
    if hasattr(soc, "roofline_obj") and (not skip_roof):
        blocks.append("roofline")
    df["ip_blocks"] = "|".join(blocks)

    # Save csv
    df.to_csv(workload_dir + "/" + "sysinfo.csv", index=False)


def detect_roofline(mspec):
    from utils import specs

    rocm_ver = mspec.rocm_version[:1]

    os_release = path("/etc/os-release").read_text()
    ubuntu_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)
    rhel_distro = specs.search(r'PLATFORM_ID="(.*?)"', os_release)
    sles_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)

    if "ROOFLINE_BIN" in os.environ.keys():
        rooflineBinary = os.environ["ROOFLINE_BIN"]
        if os.path.exists(rooflineBinary):
            console_warning("roofline", "Detected user-supplied binary")
            return {
                "rocm_ver": "override",
                "distro": "override",
                "path": rooflineBinary,
            }
        else:
            msg = "user-supplied path to binary not accessible"
            msg += "--> ROOFLINE_BIN = %s\n" % target_binary
            console_error("roofline", msg)
    elif rhel_distro == "platform:el8" or rhel_distro == "platform:el9":
        # Must be a valid RHEL machine
        distro = "platform:el8"
    elif (
        (type(sles_distro) == str and len(sles_distro) >= 3)
        and sles_distro[:2] == "15"  # confirm string and len
        and int(sles_distro[3]) >= 3  # SLES15 and SP >= 3
    ):
        # Must be a valid SLES machine
        # Use SP3 binary for all forward compatible service pack versions
        distro = "15.3"
    elif ubuntu_distro == "20.04" or ubuntu_distro == "22.04":
        # Must be a valid Ubuntu machine
        distro = ubuntu_distro
    else:
        console_error("roofline", "Cannot find a valid binary for your operating system")

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
            console_log(rs_cmd)
            success, output = capture_subprocess_output(rs_cmd)
            if not success:
                console_error(result.stderr.decode("ascii"))


def mibench(args, mspec):
    """Run roofline microbenchmark to generate peek BW and FLOP measurements."""
    console_log("roofline", "No roofline data found. Generating...")

    distro_map = {
        "platform:el8": "rhel8",
        "15.3": "sles15sp5",
        "20.04": "ubuntu20_04",
        "22.04": "ubuntu20_04",
    }

    binary_paths = []

    target_binary = detect_roofline(mspec)
    if target_binary["rocm_ver"] == "override":
        binary_paths.append(target_binary["path"])
    else:
        # check two potential locations for roofline binaries due to differences in
        # development usage vs formal install
        potential_paths = [
            "%s/utils/rooflines/roofline" % config.rocprof_compute_home,
            "%s/bin/roofline" % config.rocprof_compute_home.parent.parent,
        ]

        for dir in potential_paths:
            path_to_binary = (
                dir
                + "-"
                + distro_map[target_binary["distro"]]
                + "-"
                + mspec.gpu_model.lower()
                + "-rocm"
                + target_binary["rocm_ver"]
            )
            binary_paths.append(path_to_binary)

    # Distro is valid but cant find rocm ver
    found = False
    for path in binary_paths:
        if os.path.exists(path):
            found = True
            path_to_binary = path
            break

    if not found:
        console_error("roofline", "Unable to locate expected binary (%s)." % binary_paths)

    my_args = [
        path_to_binary,
        "-o",
        args.path + "/" + "roofline.csv",
        "-d",
        str(args.device),
    ]
    if args.quiet:
        my_args += "--quiet"
    subprocess.run(
        my_args,
        check=True,
    )


def flatten_tcc_info_across_xcds(file, xcds, tcc_channel_per_xcd):
    """
    Flatten TCC per channel counters across all XCDs in partition.
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
    for i in range(0, xcds):
        tcc_cols_in_group[i] = []

    for col in tcc_cols_orig:
        for i in range(0, xcds):
            # filter the channel index only
            p = re.compile(r"\[(\d+)\]")
            # pick up the 1st element only
            r = (
                lambda match: "["
                + str(int(match.group(1)) + i * tcc_channel_per_xcd)
                + "]"
            )
            tcc_cols_in_group[i].append(re.sub(pattern=p, repl=r, string=col))

    for i in range(0, xcds):
        # print(tcc_cols_in_group[i])
        cols += tcc_cols_in_group[i]
    # print(cols)
    df = pd.DataFrame(columns=cols)

    ### Rearrange data with extended column names

    # print(len(df_orig.index))
    for idx in range(0, len(df_orig.index), xcds):
        # assume the front none TCC columns are the same for all XCCs
        df_non_tcc = df_orig.iloc[idx].filter(regex=r"^(?!.*TCC).*$")
        # display(df_non_tcc)
        flatten_list = df_non_tcc.tolist()

        # extract all tcc from one dispatch
        # NB: assuming default contiguous order might not be safe!
        df_tcc_all = df_orig.iloc[idx : (idx + xcds)].filter(regex="TCC")
        # display(df_tcc_all)

        for idx, row in df_tcc_all.iterrows():
            flatten_list += row.tolist()
        # print(len(df.index), len(flatten_list), len(df.columns), flatten_list)
        # NB: It is not the best perf to append a row once a time
        df.loc[len(df.index)] = flatten_list

    return df


def total_xcds(archname, compute_partition):
    # check MI300 has a valid compute partition
    mi300a_archs = ["mi300a_a0", "mi300a_a1"]
    mi300x_archs = ["mi300x_a0", "mi300x_a1"]
    mi308x_archs = ["mi308x"]
    if (
        archname.lower() in mi300a_archs + mi300x_archs + mi308x_archs
        and compute_partition == "NA"
    ):
        console_error("Invalid compute partition found for {}".format(archname))
    if archname.lower() not in mi300a_archs + mi300x_archs + mi308x_archs:
        return 1
    # from the whitepaper
    # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
    if compute_partition.lower() == "spx":
        if archname.lower() in mi300a_archs:
            return 6
        if archname.lower() in mi300x_archs:
            return 8
        if archname.lower() in mi308x_archs:
            return 4
    if compute_partition.lower() == "tpx":
        if archname.lower() in mi300a_archs:
            return 2
    if compute_partition.lower() == "dpx":
        if archname.lower() in mi300x_archs:
            return 4
        if archname.lower() in mi308x_archs:
            return 2
    if compute_partition.lower() == "qpx":
        if archname.lower() in mi300x_archs:
            return 2
    if compute_partition.lower() == "cpx":
        if archname.lower() in mi300x_archs:
            return 2
        if archname.lower() in mi308x_archs:
            return 1
    # TODO implement other archs here as needed
    console_error(
        "Unknown compute partition / arch found for {} / {}".format(
            compute_partition, archname
        )
    )


def get_submodules(package_name):
    """List all submodules for a target package"""
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


def is_workload_empty(path):
    """Peek workload directory to verify valid profiling output"""
    pmc_perf_path = path + "/pmc_perf.csv"
    if os.path.isfile(pmc_perf_path):
        temp_df = pd.read_csv(pmc_perf_path)
        if temp_df.dropna().empty:
            console_error(
                "profiling"
                "Found empty cells in %s.\nProfiling data could be corrupt."
                % pmc_perf_path
            )

    else:
        console_error("profiling", "Cannot find pmc_perf.csv in %s" % path)


def print_status(msg):
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def set_locale_encoding():
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error as error:
        console_error(
            "Please ensure that the 'en_US.UTF-8' locale is available on your system.",
            exit=False,
        )
        console_error(error)
