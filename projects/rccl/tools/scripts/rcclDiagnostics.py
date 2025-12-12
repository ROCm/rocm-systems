# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information

## README
# This script gathers configuration information from your system to help identify and debug any issues when running the ROCm Communication Collectives Library (RCCL). Please ensure that python3 is installed on your system and added to your system's PATH environment variable.

# Prerequisites
# - python3 (make sure it's added to the PATH)
# - Sudo access on the system if you want ACS info
# - RCCL_PATH environment variable must be set to the directory containing the librccl.so shared object (e.g., /opt/rocm/lib/)

# Usage
# To run the script and gather the configuration information, execute the following command:
# default
# python3 rcclDiagnostics.py

# when you require acs output
# sudo python3 rcclDiagnostics.py
# Note: Running the script without sudo will not check if ACS is disabled or not, sudo is needed to complete system configuration information but the script will skip what it can't get.

# The script will gather essential system configuration information, such as OS information, network information, driver versions, etc., to help with debugging RCCL issues. It will generate a report in a readable format, which you can share with the support team or use for troubleshooting.


import subprocess
import time
import os
import re
import argparse
import textwrap


class CommandResult:
	def __init__(self, stdout, stderr):
		self.stdout = stdout
		self.stderr = stderr


# Function to Parse arguements


def parse_arguments():
	readme = """\
This script gathers configuration information from your system to help identify and debug any issues when running the ROCm Communication Collectives Library (RCCL). Please ensure that python3 is installed on your system and added to your system's PATH environment variable.\n
Prerequisites\n
- python3 (make sure it's added to the PATH)\n
- Sudo access on the system if you want ACS info\n
- RCCL_PATH environment variable must be set to the directory containing the librccl.so shared object (e.g., /opt/rocm/lib/)\n
Usage\n
To run the script and gather the configuration information, execute the following command:\n
- default\n
  python3 rcclDiagnostics.py\n
- when you require acs output\n
  sudo python3 rcclDiagnostics.py\n
Note: Running the script without sudo will not check if ACS is disabled or not, sudo is needed to complete system configuration information but the script will skip what it can't get.\n
The script will gather essential system configuration information, such as OS information, network information, driver versions, etc., to help with debugging RCCL issues. It will generate a report in a readable format, which you can share with the support team or use for troubleshooting.\n
	"""
	parser = argparse.ArgumentParser(
		description=textwrap.dedent(readme),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)

	# Add option flags

	return parser.parse_args()


# Function to center the titles in the detailed section
def centered_title(title, width, fill_char=" "):
	padding_width = (width - len(title)) // 2
	return f"{fill_char*padding_width}{title}{fill_char*padding_width}\n"


# Function to run a CLI command and return its output
def run_cli_command(command):
	try:
		result = subprocess.run(
			command,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			shell=True,
		)
		return result
	except Exception as e:
		return f"Error: {str(e)}"


def get_rccl_version(rccl_path):
	try:
		# Construct the full path to the shared object file
		librccl_path = rccl_path + "librccl.so"
		# Extract RCCL version
		cmd = f'strings {librccl_path} | grep "RCCL version"'
		rccl_version_result = subprocess.run(
			cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
		)
		rccl_version = rccl_version_result.stdout.replace("RCCL version : ", "").strip()

		# Extract commit hash
		cmd2 = f'strings {librccl_path} | grep -E "[A-Za-z]:[0-9a-z]{{7}}$"'
		commit_hash_result = subprocess.run(
			cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
		)
		commit_hash = commit_hash_result.stdout.strip()

	except Exception as e:
		result = CommandResult(
			stdout="",
			stderr=f"Error: {str(e)},\n Please set the RCCL_PATH environment to the directory containing the librccl.so shared object.",
		)
	result = CommandResult(stdout=commit_hash + "+" + rccl_version, stderr="")

	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get the status of a particular command
def status_check(summary, result):
	# List of errors to check
	error_list = [
		r"No such file or directory",
		r"Command not found",
		r"Permission denied",
		r"cannot access",
		r"error",
	]
	status = "OK"
	if summary == "Unable to detect":
		status = "WARN"
	for error in error_list:
		match = re.search(error, result.stderr, re.IGNORECASE)
		if match:
			status = "WARN"
			break
	return status


# Check if a directory is on path or LD_LIBRARY_PATH
def PATH_and_LD_LIBRARY_PATH(dir):
	try:
		path = os.environ.get("PATH")
		LD_path = os.environ.get("LD_LIBRARY_PATH")
		pattern = re.escape(dir)
		match_path = re.search(pattern, path)
		match_LD_path = re.search(pattern, LD_path)
	except Exception as e:
		return False
	pattern = re.escape(dir)
	match_path = re.search(pattern, path)
	match_LD_path = re.search(pattern, LD_path)
	if match_LD_path and match_path:
		return True
	return False


# Get hostname
def get_hostname():
	result = run_cli_command("hostname")
	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get OS version
def get_os_version():
	result = run_cli_command("cat /etc/os-release")
	match = re.search(r'PRETTY_NAME="(.+)"', result.stdout)
	if match:
		summary = match.group(1)
	else:
		summary = "Unable to detect"
	return summary, result


# Get ROCm Version
def get_ROCm_version():
	result = run_cli_command("cat /opt/rocm/.info/version")
	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get HIP Version
def get_HIP_version():
	result = run_cli_command("hipconfig --version")
	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get Vram Information
def get_Vram_info():
	result = run_cli_command("rocm-smi --showmeminfo vram")
	if result.stdout:
		summary = "Memory Usage is detailed in the Vram Information section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get UCX version
def ucx_version():
	path_check = PATH_and_LD_LIBRARY_PATH(dir="ucx")
	if path_check:
		result = run_cli_command("ucx_info -v")
		match = re.search(r"Library version: (\d+\.\d+\.\d+)", result.stdout)
		if match:
			summary = match.group(1)
		else:
			summary = "Unable to detect"
		return summary, result
	else:
		stdout = ""
		stderr = "Error: UCX not on PATH or LD_LIBRARY_PATH"
		result = CommandResult(stdout=stdout, stderr=stderr)
		summary = "UCX not on PATH or LD_LIBRARY_PATH"
		return summary, result


# Get MPI version
def mpi_version():
	path_check = PATH_and_LD_LIBRARY_PATH(dir="ompi")
	if path_check:
		result = run_cli_command("mpirun --version")
		match = re.search(r"mpirun \(Open MPI\) \d+\.\d+\.\d+", result.stdout)
		if match:
			summary = match.group()
		else:
			summary = "Unable to detect"
		return summary, result
	else:
		stdout = ""
		stderr = (
			"Error: ompi4 or ompi5 (only 1 is required) not on PATH or LD_LIBRARY_PATH"
		)
		result = CommandResult(stdout=stdout, stderr=stderr)
		summary = "ompi4 or ompi5 (only 1 is required) not on PATH or LD_LIBRARY_PATH"
		return summary, result


# Get Linux kernel version
def get_Linux_kernel_version():
	result = run_cli_command("uname -r")
	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get Resource limits
def get_resource_limits_info():
	result = run_cli_command("ulimit -a")
	if result.stdout:
		summary = "Output is detailed in the Resource limits section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get Environment config
def get_Environment_config_info():
	result = run_cli_command("env")
	if result.stdout:
		summary = "Output is detailed in the Environment Config section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get Rdma link info
def get_rdma_link_info():
	result = run_cli_command("rdma link")
	if result.stdout:
		summary = "Output is detailed in the rdma link section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get NUMA Balancing
def get_NUMA_balancing_info():
	result = run_cli_command("cat /proc/sys/kernel/numa_balancing")
	if result.stdout:
		summary = result.stdout.strip()
	else:
		summary = "Unable to detect"
	return summary, result


# Get IB status
def get_ib_status():
	result = run_cli_command("ibstatus")
	if result.stdout:
		pattern = r"Infiniband device '[^']+' port \d+ status:\s+default gid:\s+[^ ]+\s+base lid:\s+[^ ]+\s+sm lid:\s+[^ ]+\s+state:\s+\d+: ACTIVE\s+phys state:\s+\d+: LinkUp\s+rate:\s+(\d+) Gb/sec \([^)]+\)\s+link_layer:\s+"
		matches = re.findall(pattern, result.stdout)
		num_ib_devices = len(matches)
		if num_ib_devices == 0:
			summary = f"Detected {num_ib_devices} active IB devices running"
			return summary, result
		rate_same = all(x == matches[0] for x in matches)
		if rate_same:
			summary = f"Detected {num_ib_devices} active IB devices running at {matches[0]} Gb/sec"
		else:
			summary = f"Detected {num_ib_devices} active IB devices running at various rates the peak being {max(matches)} Gb/sec"

	else:
		summary = "Unable to detect"
	return summary, result


# Get Device GUIDs
def get_device_GUIDs():
	result = run_cli_command("ibv_devices")
	if result.stdout:
		summary = "Output is detailed in the IBdevices section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get IB device info
def get_ib_devinfo():
	result = run_cli_command("ibv_devinfo")
	if result.stdout:
		summary = "Output is detailed in the IBdevinfo section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get IBstat info
def get_ibstat():
	result = run_cli_command("ibstat")
	if result.stdout:
		summary = "Output is detailed in the IBstat section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get AMDKFD (GPU Driver version)
def get_gpu_driver():
	result = run_cli_command('dkms status | grep "amdgpu"')
	if result.stdout:
		pattern = r"^.*amdgpu.*$"
		matching_lines = re.findall(pattern, result.stdout, flags=re.MULTILINE)
		if len(matching_lines) == 0:
			summary = "No gpu driver detected"
			return summary, result
		summary = matching_lines[0] + ", WARN = maybe >1 driver check below"
	else:
		summary = "Unable to detect"
	return summary, result


# Get DKMS module info
def get_dkms_status():
	result = run_cli_command("dkms status")
	if result.stdout:
		summary = "DKMS information is detailed in the DKMS Status section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get IP A
def get_IP_addr():
	result = run_cli_command("ip a")
	if result.stdout:
		summary = "IP address information is detailed in the IP Addr section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get IP Link
def get_IP_link():
	result = run_cli_command("ip link")
	if result.stdout:
		summary = "IP link information is detailed in the IP Link section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get IP route
def get_IP_route():
	result = run_cli_command("ip route")
	if result.stdout:
		summary = "IP Route information is detailed in the IP Route section"
	else:
		summary = "Unable to detect"
	return summary, result


# Get ACS info
def get_acs_info():
	result = run_cli_command("lspci -vvv | grep ACSCtl")
	if result.stdout:
		pattern = r"SrcValid\+"
		matches = re.findall(pattern, result.stdout)
		if len(matches) != 0:
			summary = "ACS has not been disabled"
		else:
			summary = "ACS has been disabled"
	else:
		summary = "Unable to detect"
	return summary, result


# Get rocminfo
def get_rocminfo():
	result = run_cli_command("rocminfo")
	if result.stdout:
		gpu_pattern = re.compile(
			r"Name:\s+(gfx\d+)(?:.*?Marketing Name:\s+([^\n]+))?.*?Compute Unit:\s+(\d+)",
			re.DOTALL,
		)
		matches = gpu_pattern.findall(result.stdout)
		num_gpus = len(matches)
		valid_marketing_names = ["MI300X", "MI300A", "MI300", "MI250X/MI250", "MI200"]
		gpu_name = ""
		for name in valid_marketing_names:
			if name in matches[0][1]:
				gpu_name = name
				break
		if gpu_name == "":
			if "gfx942" == matches[0][0]:
				if 304 == int(matches[0][2]):
					gpu_name = "MI300X"
				elif 228 == int(matches[0][2]):
					gpu_name = "MI300A"
				else:
					gpu_name = f"MI300 with {int(matches[0][2])} CUs"
			elif "gfx90a" == matches[0][0]:
				if 104 <= int(matches[0][2]):
					gpu_name = "MI250X/MI250"
				else:
					gpu_name = f"MI200 with {int(matches[0][2])} CUs"
		summary = f"Found {num_gpus} {gpu_name} GPUs"
	else:
		summary = "Unable to detect"
	return summary, result


def checklimits_from_file():
	summary = ""
	try:
		with open("/etc/security/limits.conf", "r") as file:
			lines = file.read().splitlines()

		# Reverse lines list to check for the last occurrence (to avoid overwriting)
		lines.reverse()

		limit_soft_nofile_line = "* soft nofile 1048576"
		limit_hard_nofile_line = "* hard nofile 1048576"
		limit_soft_memlock_line = "* soft memlock unlimited"
		limit_hard_memlock_line = "* hard memlock unlimited"

		lines_to_check = [
			limit_soft_nofile_line,
			limit_hard_nofile_line,
			limit_soft_memlock_line,
			limit_hard_memlock_line,
		]

		missing_lines = []

		for line in lines_to_check:
			if line not in lines:
				missing_lines.append(line)

		if missing_lines:
			summary = "Limits not set"
			error = ""
			for missing_line in missing_lines:
				error += missing_line + "\n"
			results = CommandResult(
				stdout="",
				stderr="Error: The following lines are missing in /etc/security/limits.conf:"
				+ error,
			)
			return summary, results
		else:
			print("All required lines are present in /etc/security/limits.conf.")
			summary = "Limits set correctly"
			results = CommandResult(
				stdout="All required lines are present in /etc/security/limits.conf.",
				stderr="",
			)
			return summary, results

	except FileNotFoundError:
		summary = "Unable to detect"
		results = CommandResult(
			stdout="",
			stderr="Error: File /etc/security/limits.conf not found on this system.",
		)
		return summary, results
	except Exception as e:
		summary = "Unable to detect"
		results = CommandResult(
			stdout="",
			stderr=f"Error opening or reading /etc/security/limits.conf: {str(e)}",
		)
		return summary, results


# Check max file descriptors and max lock memory
def checklimits():
	result = run_cli_command("ulimit -n")
	result2 = run_cli_command("ulimit -l")
	if result.stdout and result2.stdout:
		file_descriptors = int(result.stdout)
		locked_mem = str(result2.stdout).strip()
		if file_descriptors >= 1048576 and locked_mem == "unlimited":
			summary = "Limits set correctly"
			stdout = (
				"ulimit -n output:\n"
				+ result.stdout
				+ "\n"
				+ "ulimit -l output:\n"
				+ result2.stdout
			)
			results = CommandResult(stdout=stdout, stderr="")
			return summary, results
		else:
			summary, results = checklimits_from_file()
			return summary, results

	else:
		summary, results = checklimits_from_file()
		return summary, results


# Gather all data and build summary table and detailed output format
def get_config(root_enabled):
	# Run the commands and store the command outputs

	# Hostname
	hostname_summary, hostname_result = get_hostname()
	hostname_status = status_check(hostname_summary, hostname_result)

	# OS version
	os_summary, os_result = get_os_version()
	os_status = status_check(os_summary, os_result)

	# ROCm Version
	ROCm_summary, ROCm_result = get_ROCm_version()
	ROCm_status = status_check(ROCm_summary, ROCm_result)

	# HIP Version
	HIP_summary, HIP_result = get_HIP_version()
	HIP_status = status_check(HIP_summary, HIP_result)

	# Vram info
	Vram_summary, Vram_result = get_Vram_info()
	Vram_status = status_check(Vram_summary, Vram_result)

	# UCX Version
	ucx_summary, ucx_result = ucx_version()
	ucx_status = status_check(ucx_summary, ucx_result)

	# MPI Version
	mpi_summary, mpi_result = mpi_version()
	mpi_status = status_check(mpi_summary, mpi_result)

	# Linux kernel version
	Lkv_summary, Lkv_result = get_Linux_kernel_version()
	Lkv_status = status_check(Lkv_summary, Lkv_result)

	# Resource limits
	rlv_summary, rlv_result = get_resource_limits_info()
	rlv_status = status_check(rlv_summary, rlv_result)

	# Environment config
	env_summary, env_result = get_Environment_config_info()
	env_status = status_check(env_summary, env_result)

	# Rdma link info
	rdl_summary, rdl_result = get_rdma_link_info()
	rdl_status = status_check(rdl_summary, rdl_result)

	# NUMA Balancing info
	nb_summary, nb_result = get_NUMA_balancing_info()
	nb_status = status_check(nb_summary, nb_result)

	# IB status info
	ibs_summary, ibs_result = get_ib_status()
	ibs_status = status_check(ibs_summary, ibs_result)

	# Device GUIDs
	GUIDs_summary, GUIDs_result = get_device_GUIDs()
	GUIDs_status = status_check(GUIDs_summary, GUIDs_result)

	# IB device info
	ib_dev_summary, ib_dev_result = get_ib_devinfo()
	ib_dev_status = status_check(ib_dev_summary, ib_dev_result)

	# IBstat info
	ib_stat_summary, ib_stat_result = get_ibstat()
	ib_stat_status = status_check(ib_stat_summary, ib_stat_result)

	# AMD GPU driver version
	GPU_driver_summary, GPU_driver_result = get_gpu_driver()
	pattern = r"^.*amdgpu.*$"
	matching_lines = re.findall(pattern, GPU_driver_result.stdout, flags=re.MULTILINE)
	if len(matching_lines) > 1:
		GPU_driver_status = "WARN"
	else:
		GPU_driver_status = status_check(GPU_driver_summary, GPU_driver_result)

	# DKMS module info
	dkms_summary, dkms_result = get_dkms_status()
	dkms_status = status_check(dkms_summary, dkms_result)

	# IP addr info
	ip_addr_summary, ip_addr_result = get_IP_addr()
	ip_addr_status = status_check(ip_addr_summary, ip_addr_result)

	# IP link info
	ip_link_summary, ip_link_result = get_IP_link()
	ip_link_status = status_check(ip_link_summary, ip_link_result)

	# IP route info
	ip_route_summary, ip_route_result = get_IP_route()
	ip_route_status = status_check(ip_route_summary, ip_route_result)

	# ACS info
	if root_enabled:
		acs_summary, acs_result = get_acs_info()
		acs_status = status_check(acs_summary, acs_result)
	else:
		acs_summary = "Requires script to be run with root access"
		acs_result = CommandResult(stdout="", stderr="Error: " + acs_summary)
		acs_status = "SKIPPED"

	# ROCM info
	rocm_info_summary, rocm_info_result = get_rocminfo()
	rocm_info_status = status_check(rocm_info_summary, rocm_info_result)

	# Check max file descriptors and max lock memory
	limits_summary, limits_result = checklimits()
	limits_status = status_check(limits_summary, limits_result)

	# Get rccl version
	rccl_path = os.getenv("RCCL_PATH")
	if rccl_path is None:
		rccl_version_summary = "Error: RCCL_PATH environment variable is not set"
		rccl_version_result = CommandResult(stdout="", stderr=rccl_version_summary)
		rccl_version_status = status_check(rccl_version_summary, rccl_version_result)
	else:
		rccl_version_summary, rccl_version_result = get_rccl_version(
			rccl_path=rccl_path
		)
		rccl_version_status = status_check(rccl_version_summary, rccl_version_result)

	# Create the summary table
	summary_table = (
		f"\n\n{'='*119}\n"
		f"{'Component':<30}| {'Status':<13} | Value\n"
		f"{'='*119}\n"
		f"Host Name{' ':<21}| {hostname_status:<13} | {hostname_summary}\n"
		f"OS Version{' ':<20}| {os_status:<13} | {os_summary}\n"
		f"ROCm Version{' ':<18}| {ROCm_status:<13} | {ROCm_summary}\n"
		f"HIP Version{' ':<19}| {HIP_status:<13} | {HIP_summary}\n"
		f"RCCL Version{' ':<18}| {rccl_version_status:<13} | {rccl_version_summary}\n"
		f"Vram Information{' ':<14}| {Vram_status:<13} | {Vram_summary}\n"
		f"UCX Version{' ':<19}| {ucx_status:<13} | {ucx_summary}\n"
		f"MPI Version{' ':<19}| {mpi_status:<13} | {mpi_summary}\n"
		f"Linux Kernel Version{' ':<10}| {Lkv_status:<13} | {Lkv_summary}\n"
		f"Resource limits{' ':<15}| {rlv_status:<13} | {rlv_summary}\n"
		f"Environment Configuration{' ':<5}| {env_status:<13} | {env_summary}\n"
		f"RDMA Link Information{' ':<9}| {rdl_status:<13} | {rdl_summary}\n"
		f"NUMA Balancing Information{' ':<4}| {nb_status:<13} | {nb_summary}\n"
		f"NIC Status{' ':<20}| {ibs_status:<13} | {ibs_summary}\n"
		f"Device GUIDs Information{' ':<6}| {GUIDs_status:<13} | {GUIDs_summary}\n"
		f"IB device Information{' ':<9}| {ib_dev_status:<13} | {ib_dev_summary}\n"
		f"IBstat Information{' ':<12}| {ib_stat_status:<13} | {ib_stat_summary}\n"
		f"AMD GPU driver version{' ':<8}| {GPU_driver_status:<13} | {GPU_driver_summary}\n"
		f"DKMS Module Information{' ':<7}| {dkms_status:<13} | {dkms_summary}\n"
		f"IP Address Information{' ':<8}| {ip_addr_status:<13} | {ip_addr_summary}\n"
		f"IP Link Information{' ':<11}| {ip_link_status:<13} | {ip_link_summary}\n"
		f"IP Route Information{' ':<10}| {ip_route_status:<13} | {ip_route_summary}\n"
		f"ACS Disabled{' ':<18}| {acs_status:<13} | {acs_summary}\n"
		f"Node Status{' ':<19}| {rocm_info_status:<13} | {rocm_info_summary}\n"
		f"File Descriptor Information{' ':<3}| {limits_status:<13} | {limits_summary}\n"
		f"{'='*119}"
	)

	# Combine details
	details_width = 120
	details = (
		f"\n\n\nDetailed Output:\n"
		f"{centered_title('Host Name', details_width, '=')}\n"
		f"{hostname_result.stdout.strip()}{hostname_result.stderr.strip()}\n\n"
		f"{centered_title('OS info', details_width, '=')}\n"
		f"{os_result.stdout.strip()}{os_result.stderr.strip()}\n\n"
		f"{centered_title('ROCm Version', details_width, '=')}\n"
		f"{ROCm_result.stdout.strip()}{ROCm_result.stderr.strip()}\n\n"
		f"{centered_title('HIP Version', details_width, '=')}\n"
		f"{HIP_result.stdout.strip()}{HIP_result.stderr.strip()}\n\n"
		f"{centered_title('RCCL Version', details_width, '=')}\n"
		f"{rccl_version_result.stdout.strip()}{rccl_version_result.stderr.strip()}\n\n"
		f"{centered_title('Vram Information', details_width, '=')}\n"
		f"{Vram_result.stdout.strip()}{Vram_result.stderr.strip()}\n\n"
		f"{centered_title('UCX Version', details_width, '=')}\n"
		f"{ucx_result.stdout.strip()}{ucx_result.stderr.strip()}\n\n"
		f"{centered_title('MPI Version', details_width, '=')}\n"
		f"{mpi_result.stdout.strip()}{mpi_result.stderr.strip()}\n\n"
		f"{centered_title('Linux Kernel Version', details_width, '=')}\n"
		f"{Lkv_result.stdout.strip()}{Lkv_result.stderr.strip()}\n\n"
		f"{centered_title('Resource limits', details_width, '=')}\n"
		f"{rlv_result.stdout.strip()}{rlv_result.stderr.strip()}\n\n"
		f"{centered_title('Environment Configuration', details_width, '=')}\n"
		f"{env_result.stdout.strip()}{env_result.stderr.strip()}\n\n"
		f"{centered_title('RDMA Link Information', details_width, '=')}\n"
		f"{rdl_result.stdout.strip()}{rdl_result.stderr.strip()}\n\n"
		f"{centered_title('NUMA Balancing Information', details_width, '=')}\n"
		f"{nb_result.stdout.strip()}{nb_result.stderr.strip()}\n\n"
		f"{centered_title('Network Interface Controller (NIC) Information', details_width, '=')}\n"
		f"{ibs_result.stdout.strip()}{ibs_result.stderr.strip()}\n\n"
		f"{centered_title('IBdevices', details_width, '=')}\n"
		f"{GUIDs_result.stdout.strip()}{GUIDs_result.stderr.strip()}\n\n"
		f"{centered_title('IBdevinfo', details_width, '=')}\n"
		f"{ib_dev_result.stdout.strip()}{ib_dev_result.stderr.strip()}\n\n"
		f"{centered_title('IBstat', details_width, '=')}\n"
		f"{ib_stat_result.stdout.strip()}{ib_stat_result.stderr.strip()}\n\n"
		f"{centered_title('GPU Driver Version', details_width, '=')}\n"
		f"{GPU_driver_result.stdout.strip()}{GPU_driver_result.stderr.strip()}\n\n"
		f"{centered_title('DKMS Status', details_width, '=')}\n"
		f"{dkms_result.stdout.strip()}{dkms_result.stderr.strip()}\n\n"
		f"{centered_title('IP Addr', details_width, '=')}\n"
		f"{ip_addr_result.stdout.strip()}{ip_addr_result.stderr.strip()}\n\n"
		f"{centered_title('IP Link', details_width, '=')}\n"
		f"{ip_link_result.stdout.strip()}{ip_link_result.stderr.strip()}\n\n"
		f"{centered_title('IP Route', details_width, '=')}\n"
		f"{ip_route_result.stdout.strip()}{ip_route_result.stderr.strip()}\n\n"
		f"{centered_title('ACS', details_width, '=')}\n"
		f"{acs_result.stdout.strip()}{acs_result.stderr.strip()}\n\n"
		f"{centered_title('ROCm Information', details_width, '=')}\n"
		f"{rocm_info_result.stdout.strip()}{rocm_info_result.stderr.strip()}\n\n"
		f"{centered_title('File Descriptor Limits', details_width, '=')}\n"
		f"{limits_result.stdout.strip()}{limits_result.stderr.strip()}\n\n"
	)
	return summary_table, details


def main():
	args = parse_arguments()
	hostname = os.uname().nodename
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	file_name = f"config.{hostname}.{timestamp}.txt"
	if os.geteuid() == 0:
		root_enabled = True
	else:
		root_enabled = False
	summary_table, details = get_config(root_enabled)

	# Print summary out to cli
	print(summary_table)
	current_directory = os.getcwd()
	print("Detailed output file is at: " + current_directory + "/" + file_name)
	# Write the summary table and details to the output file
	with open(file_name, "w") as file:
		file.write(summary_table)
		file.write(details)


if __name__ == "__main__":
	main()
