#!/usr/bin/env python3

import os
import subprocess
import logging
import json
import argparse
import glob
import re
import sys
import textwrap
from enum import Enum

# Check for required packages before importing them
def check_required_packages():
    """Check if required Python packages are installed"""
    missing_packages = []
    required_packages = {
        'prettytable': 'prettytable',
        'yaml': 'PyYAML'
    }

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print("\n" + "="*70)
        print("WARNING: Missing Required Python Packages")
        print("="*70)
        print(f"\nThe following packages are required but not installed:")
        for pkg in missing_packages:
            print(f"  - {pkg}")

        print("\nTo install the missing packages, run:")
        print(f"  pip3 install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("  pip3 install -r <ROCM_INSTALL_PATH>/share/rdhc/requirements.txt")
        print("  Or\n  pip3 install -r requirements.txt")
        print("\n" + "="*70 + "\n")

        print("Exiting...")
        sys.exit(1)
    else:
        return True

# Check packages before importing
packages_available = check_required_packages()

# Now import the packages
try:
    from prettytable import PrettyTable
    import yaml
except ImportError:
    print("WARNING: Unable to import the required Python Packages !!!!")
    print("Exiting...")
    sys.exit(1)


# Define test status enum
class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_INSTALLED = "NOT INSTALLED"
    NOT_TESTED = "NOT TESTED"

def run_command(command, shell=False):
    """Run a command and return stdout, stderr, and return code"""
    try:
        if isinstance(command, str) and not shell:
            command = command.split()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    except Exception as e:
        logging.error(f"Error executing command: {command}, error: {str(e)}")
        return "", str(e), 1

def generate_table_report(results):
    """Generate a pretty table report of test results"""
    table = PrettyTable()
    table.title = "RDHC Test Results"
    table.field_names = ["Test Name", "Description", "Status", "Details"]
    table.align = "l"  # Left align all columns
    # Standard test descriptions
    descriptions = {
        "gpu_presence": "Check for AMD GPUs in the system",
        "amdgpu_driver": "Check if AMDGPU driver is working properly",
        "rocminfo": "Check if rocminfo is working properly",
        "amd_smi": "Check if amd-smi is working properly",
        "lib_dependencies": "Check rocm libraries runtime dependencies"
    }

    for test_name, result in results.items():
        # For component tests, create a standard description
        if test_name.startswith("rocm-") or test_name.startswith("hip-"):
            description = f"Verify {test_name} usability"
        else:
            description = descriptions.get(test_name, f"Check {test_name} usability")

        table.add_row([
            test_name,
            description,
            result["status"],
            result["reason"][:50] + "..." if len(result["reason"]) > 50 else result["reason"]
        ])

    return table

def generate_table_system_info(system_info):
    """Generate a pretty table report of amdgpu driver information"""
    table = PrettyTable()

    table.align = "l"  # Left align all columns
    table.title = "General Information"
    table.header = False  # No header row
    # If system_info is empty, add a placeholder row
    if not system_info:
        table.add_row(["No information available", "N/A"])
    else:
        for key, value in system_info.items():
            # Add a row for each key-value pair
            table.add_row([key, value])
    return table

def generate_table_gpu_info(gpu_info_dict):
    """Generate a pretty table report of GPU information"""

    # Create a function to flatten the nested dictionary
    def flatten_dict(d, parent_key='', sep=':'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # create the table
    table = PrettyTable()
    table.title = "GPU Device Information"
    table.align = "l"  # Left align all columns
    col_width = 25  # Maximum width for each column

    # Flatten each GPU dictionary
    flattened_gpus = {}
    for gpu_key, gpu_data in gpu_info_dict.items():
        flattened_gpus[gpu_key] = flatten_dict(gpu_data)

    # Get all unique keys across all GPUs while preserving order
    all_keys = []
    for gpu_data in flattened_gpus.values():
        for key in gpu_data.keys():
            if key not in all_keys:
                all_keys.append(key)

    # table.field_names = ["##, "Property", "GPU_O", "GPU_1", ...."]
    table.field_names = ["##", "Property"] + list(flattened_gpus.keys())

    # Add rows to the table
    for idx, key in enumerate(all_keys):
        row = [idx, key]    # Add row number as first column
        for gpu_key in flattened_gpus.keys():
            # row.append(flattened_gpus[gpu_key].get(key, "N/A"))
            value = flattened_gpus[gpu_key].get(key, "N/A")
            # Convert to string if not already
            value_str = str(value)
            # print(f"Processing key: {key}, value_str: {value_str} ; value :{value}")

            # Apply text wrapping if value exceeds max_width
            if len(value_str) > col_width:
                wrapped_value = textwrap.fill(value_str, width=col_width)
                row.append(wrapped_value)
            else:
                row.append(value_str)

        table.add_row(row)

    return table

def generate_table_firmware_info(firmware_info):
    """Generate a pretty table report of amdgpu firmware version informations"""

    gpu_dict = firmware_info
    # Create a flattened table with FW_ID as rows and GPUs as columns
    table = PrettyTable()
    table.align = "l"  # Left align all columns
    table.title = "AMDGPU Firmware Version Information"
    # table.field_names = ["##, "FW_ID", "GPU_O", "GPU_1", ...."]
    table.field_names = ["##","FW_ID"] + list(gpu_dict.keys())

    # Get all firmware IDs while preserving order
    fw_ids = []
    for gpu_key, gpu_data in gpu_dict.items():
        for fw_key, fw_data in gpu_data['FW_LIST'].items():
            if fw_data['FW_ID'] not in fw_ids:
                fw_ids.append(fw_data['FW_ID'])

    # Add rows to the table
    for idx, fw_id in enumerate(fw_ids):
        row = [idx, fw_id]    # Add row number and FW_ID as first two columns
        for gpu_key in gpu_dict.keys():
            # Find the version for this firmware ID in this GPU
            version = "N/A"
            for fw_key, fw_data in gpu_dict[gpu_key]['FW_LIST'].items():
                if fw_data['FW_ID'] == fw_id:
                    version = fw_data['FW_VERSION']
                    break
            row.append(version)
        table.add_row(row)

    return table

def export_to_json(results, filename):
    """Export test results to a JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results exported to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error exporting results to JSON: {e}")
        return False

class ROCMHealthCheck:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger("RDHC")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger

        # List of all possible ROCm components to check
        self.all_components = [
            "hipcc",
            "hip-runtime-amd",
            "hipblas", "hipfft", "hipcub-dev", "hipsolver",
            "rocblas", "rocfft", "rocprim-dev" , "rocrand", "rocsolver",
            "rocsparse", "rocthrust-dev",
            "miopen-hip",
            "applications"
        ]

        # Components to exclude from testing
        self.exclude_list = ["rocm-utils", "rocm-cmake"]

        # Categorized rocm-example targets
        self.rocm_examples_targets = {}

        # Store system & amdgpu driver information
        self.system_info = {}
        self.gpu_info_dict = {}
        self.gpus = []
        self.fw_info = ""
        self.gpu_fw_info_dict = {}
        self.amdgpu_firmware_info = {}

        # Store test results
        self.results = {}

        # Get ROCM version
        self.rocm_version_str = self.get_rocm_version()
        self.rocm_version_num = self.get_rocm_version_num()
        self.logger.info(f"ROCm version string: {self.rocm_version_str}")
        self.logger.info(f"ROCm version number: {self.rocm_version_num}")
        self.system_info["ROCm version"] = self.rocm_version_str

        # Find installed components
        self.installed_components = self.get_installed_components()
        self.logger.info(f"Installed components: {self.installed_components}")

    def get_rocm_version(self):
        """Get the ROCm version string from /opt/rocm/.info/version"""
        try:
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            with open(f"{rocm_path}/.info/version", "r") as f:
                return f.read().strip()
        except Exception as e:
            self.logger.error(f"Error reading ROCm version: {e}")
            return "Unknown"

    def get_rocm_version_num(self):
        """Convert version string (e.g., '6.4.0-15121') to numeric format (e.g., '60400')"""
        try:
            if self.rocm_version_str == "Unknown":
                return "00000"

            # Extract version numbers using regex (e.g., "6.4.0" from "6.4.0-15121")
            match = re.match(r"(\d+)\.(\d+)\.(\d+)", self.rocm_version_str)
            if match:
                major, minor, patch = match.groups()
                # Format as XXYYZZ
                return f"{major.zfill(1)}{minor.zfill(2)}{patch.zfill(2)}"
            return "00000"
        except Exception as e:
            self.logger.error(f"Error processing ROCm version number: {e}")
            return "00000"

    def detect_os_type(self):
        """Detect the operating system type"""
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                os_info = f.read().lower()
                if "ubuntu" in os_info:
                    return "ubuntu"
                elif "rhel" in os_info or "centos" in os_info or\
                     "fedora" in os_info or "almalinux" in os_info or\
                     "azurelinux" in os_info:
                    return "rhel"
                elif "sles" in os_info or "suse" in os_info:
                    return "sles"
        # Default to ubuntu if can't determine
        return "ubuntu"

    def get_installed_components(self):
        """Get list of installed ROCm components based on OS type and installation method"""
        installed = []
        package_installed = []
        folder_installed = []

        # First, try to detect components via package managers
        os_type = self.detect_os_type()
        package_installed = self._get_components_from_packages(os_type)

        # If no packages found, or if ROCM_PATH points to a non-standard location,
        # check for folder-based installation
        if not package_installed:
            rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
            folder_installed = self._get_components_from_folders(rocm_path)

        # Log the detection method used
        if package_installed:
            installed = package_installed
            self.logger.info(f"Detected components for a quick test via package manager: {len(package_installed)}")
        elif folder_installed:
            installed = folder_installed
            self.logger.info(f"Detected components for a quick test via folder structure: {len(folder_installed)}")
        else:
            self.logger.warning("!!! No ROCm components detected via packages or folders.")

        return installed

    def _get_components_from_packages(self, os_type):
        """Get installed components from package managers"""
        installed = []

        for component in self.all_components:
            if os_type == "ubuntu":
                stdout, _, ret_code = run_command(f"dpkg -l {component}*", shell=True)
                if ret_code == 0 and "ii" in stdout:
                    # Extract exact package name from dpkg output
                    for line in stdout.split("\n"):
                        if line.strip().startswith("ii"):
                            parts = line.split()
                            if len(parts) > 1 and parts[1].startswith(component):
                                installed.append(parts[1])
                                break

            elif os_type == "rhel":
                stdout, _, ret_code = run_command(f"rpm -q {component}", shell=True)
                if ret_code == 0:
                    # Extract package name from rpm output
                    for line in stdout.split("\n"):
                        if component in line:
                            installed.append(line.strip())
                            break

            elif os_type == "sles":
                stdout, _, ret_code = run_command(f"zypper se -i {component}", shell=True)
                if ret_code == 0 and "i  | " in stdout:
                    # Extract package name from zypper output
                    for line in stdout.split("\n"):
                        if "i  | " in line and component in line:
                            parts = line.split("|")
                            if len(parts) > 1:
                                installed.append(parts[1].strip())
                                break

        return installed

    def _get_components_from_folders(self, rocm_path):
        """Get available components from ROCm folder structure"""
        installed = []

        if not os.path.exists(rocm_path):
            self.logger.debug(f"ROCm path does not exist: {rocm_path}")
            return installed

        # Define component detection strategies
        component_detection = {
            "hipcc": [
                f"{rocm_path}/bin/hipcc"
            ],
            "hip-runtime-amd": [
                f"{rocm_path}/lib/libamdhip64.so*"
            ],
            "hipblas": [
                f"{rocm_path}/lib/libhipblas.so*"
            ],
            "hipfft": [
                f"{rocm_path}/lib/libhipfft.so*"
            ],
            "hipcub-dev": [
                f"{rocm_path}/include/hipcub/hipcub.hpp"
            ],
            "hipsolver": [
                f"{rocm_path}/lib/libhipsolver.so*"
            ],
            "rocblas": [
                f"{rocm_path}/lib/librocblas.so*"
            ],
            "rocfft": [
                f"{rocm_path}/lib/librocfft.so*"
            ],
            "rocprim-dev": [
                f"{rocm_path}/include/rocprim/rocprim.hpp"
            ],
            "rocrand": [
                f"{rocm_path}/lib/librocrand.so*"
            ],
            "rocsolver": [
                f"{rocm_path}/lib/librocsolver.so*"
            ],
            "rocsparse": [
                f"{rocm_path}/lib/librocsparse.so*"
            ],
            "rocthrust-dev": [
                f"{rocm_path}/include/thrust",
                f"{rocm_path}/lib/cmake/rocthrust"
            ],
            "miopen-hip": [
                f"{rocm_path}/lib/libMIOpen.so*",
                f"{rocm_path}/bin/MIOpenDriver"
            ]
        }

        # Check each component
        for component in self.all_components:
            if component in component_detection:
                component_found = False
                detection_paths = component_detection[component]

                for path_pattern in detection_paths:
                    # Use glob to handle wildcards like *.so*
                    matching_paths = glob.glob(path_pattern)
                    if matching_paths:
                        # Check if any matching path actually exists
                        for path in matching_paths:
                            if os.path.exists(path):
                                installed.append(component)
                                component_found = True
                                self.logger.debug(f"Found {component} at: {path}")
                                break
                        if component_found:
                            break
                    elif os.path.exists(path_pattern):
                        installed.append(component)
                        component_found = True
                        self.logger.debug(f"Found {component} at: {path_pattern}")
                        break

        return installed

    def test_GPUPresence(self):
        """Test if AMD GPU is present in the system"""

        # AMD GPUs PCI class codes: 03xx (Display controllers ), 12xx (Processing accelerators)
        # use class codes also to identify AMD GPUs
        stdout, _, ret_code = run_command( "lspci -d 1002: -nn | grep -Ei \
                                           'Display controller|Processing accelerators|\[03[[:xdigit:]]{2}\]|\[12[[:xdigit:]]{2}\]' ",\
                                            shell=True)
        gpu_hw = stdout.strip()
        if ret_code == 0 and gpu_hw:
            self.logger.debug(f"--Found AMD GPU(s): \n{gpu_hw}")
            return TestStatus.PASS.value, "Found AMD GPU(s)."
        return TestStatus.FAIL.value, "No AMD GPU detected."

    def test_amdgpu_driver(self):
        """Test if AMDGPU driver is installed and working properly"""
        issues = []
        all_checks_passed = True

        # Check if amdgpu driver is loaded
        stdout, _, ret_code = run_command("lsmod | grep amdgpu", shell=True)
        if ret_code != 0 or not stdout.strip():
            return TestStatus.FAIL.value, "AMDGPU driver module is not loaded."

        # Check DKMS status
        self.logger.info("--Checking DKMS status for amdgpu driver...")
        # Get current running kernel version
        stdout, stderr, ret_code = run_command("uname -r", shell=True)
        if ret_code != 0:
            self.logger.debug(f"----Failed to get Linux kernel version")

        current_kernel = stdout.strip()

        stdout, stderr, ret_code = run_command("dkms status", shell=True)
        stdout = stdout.strip()
        if ret_code != 0:
            self.logger.debug(f"----Failed to check DKMS status")
        else:
            if current_kernel:
                # Highlight the dkms status with "*" for the current kernel installed
                dkms_output = []
                for line in stdout.split('\n'):
                    if "amdgpu" in line and current_kernel in line:
                        dkms_output.append(f"{line.strip()} *")
                    else:
                        dkms_output.append(line.strip())
                self.system_info["dkms status"] = "\n".join(dkms_output)
            else:
                self.system_info["dkms status"] = stdout

        if "amdgpu" in stdout and "installed" in stdout:
            self.logger.debug("--AMDGPU DKMS module is installed.")
        else:
            all_checks_passed = False
            issues.append("AMDGPU DKMS driver not found or not installed.")

        # Check driver initialization state
        self.logger.info("--Checking AMDGPU driver initialization state...")
        init_state_checked = False
        if os.path.exists("/sys/module/amdgpu/initstate"):
            try:
                with open("/sys/module/amdgpu/initstate", "r") as f:
                    init_state = f.read().strip()
                    if init_state:
                        self.logger.debug(f"--AMDGPU init state: {init_state}")
                        init_state_checked = True
                    else:
                        all_checks_passed = False
                        issues.append("AMDGPU driver not initialized properly.")
                        #self.logger.debug("--AMDGPU driver not initialized properly.")
            except Exception as e:
                all_checks_passed = False
                issues.append(f"Could not read AMDGPU init state: {e}")
        else:
            all_checks_passed = False
            issues.append("AMDGPU init state file not found.")

        # Check power management
        # cat /sys/class/drm/card*/device/pp_dpm_sclk	=> "If it exists and returns a value,
        # then power management is enabled. That means the driver loaded and is using features
        # from firmware which is a safe indicator that things are working properly.
        self.logger.info("--Checking power management status...")
        sclk_files = glob.glob("/sys/class/drm/card*/device/pp_dpm_sclk")
        if sclk_files:
            sclk_checked = False
            for sclk_file in sclk_files:
                try:
                    with open(sclk_file, "r") as f:
                        sclk_info = f.read().strip()
                        if sclk_info:
                            self.logger.debug(f"--Power management is enabled. \n {sclk_file}: \n {sclk_info}")
                            sclk_checked = True
                            break
                except Exception as e:
                    self.logger.warning(f"!!! Could not read {sclk_file}: \n {e}")

            if not sclk_checked:
                all_checks_passed = False
                issues.append("Power management not enabled.")
        else:
            all_checks_passed = False
            issues.append("No power management files found.")

        if all_checks_passed:
            return TestStatus.PASS.value, "AMDGPU driver is fully functional."
        else:
            # Driver is loaded but with issues
            self.logger.error(f"--AMDGPU driver loaded but with issues: {', --'.join(issues)}")
            return TestStatus.PASS.value, f"AMDGPU driver loaded but with issues."

    def test_rocminfo(self):
        """Test if rocminfo works properly"""
        stdout, stderr, ret_code = run_command("rocminfo")
        if ret_code != 0:
            self.logger.error(f"--rocminfo command failed: \n{stderr}")
            return TestStatus.FAIL.value, f"rocminfo command failed."

        # Check if GPU agents are detected
        if "Device Type" in stdout:
            gpu_count = stdout.count("Device Type:             GPU")
            cpu_count = stdout.count("Device Type:             CPU")
            self.logger.info(f"--rocminfo detected {gpu_count} GPU agent(s) and {cpu_count} CPU agent(s).")
            return TestStatus.PASS.value, f"rocminfo detected {gpu_count} GPU agent(s) and {cpu_count} CPU agent(s)."
        else:
            return TestStatus.FAIL.value, "rocminfo executed but no GPU agents detected."

    def test_rocm_agent_enumerator(self):
        """Test if rocm_agent_enumerator works properly"""
        stdout, stderr, ret_code = run_command("rocm_agent_enumerator")
        if ret_code != 0:
            self.logger.error(f"--rocm_agent_enumerator command failed: \n{stderr}")
            return TestStatus.FAIL.value, f"rocm_agent_enumerator command failed."

        # Check if GPU agents are detected
        if "gfx" in stdout:
            agents = ", ".join(stdout.splitlines())
            self.logger.info(f"--Detected gpu agents: {agents}")
            self.system_info["GPU Arch "] = stdout.splitlines()[0]  # Store first line as detected agents
            return TestStatus.PASS.value, f"Detected gpus: {agents}."
        else:
            self.logger.error("--rocm_agent_enumerator executed but no GPU agents detected.")
            return TestStatus.FAIL.value, "rocm_agent_enumerator executed but no GPU agents detected."

    def test_amd_smi(self):
        """Test if amd-smi works properly.
            Get all GPU related information using amd-smi command.
        """
        results = {}

        # Test basic amd-smi command
        stdout, stderr, ret_code = run_command("amd-smi version")
        self.logger.debug(f"--amd-smi version: \n {stdout.strip()}")
        if ret_code != 0:
            self.logger.error(f"--amd-smi command failed: \n{stderr}")
            return TestStatus.FAIL.value, f"amd-smi command failed: {stderr}"

        # Test list options and save the data for report
        stdout, stderr, ret_code = run_command("amd-smi list")
        stdout = stdout.strip()
        if ret_code == 0  and stdout:
            self.logger.debug(f"amd-smi list : \n {stdout}")
            results["list"] = "Passed"
            self.gpu_info_dict = self._convert_string_to_dict(stdout)
        else:
            self.logger.warning(f"!!! amd-smi list failed: {stderr}")
            results["list"] = "Failed"

        # Test static options and save the data for report
        smi_static_dict = {}
        stdout, stderr, ret_code = run_command("amd-smi static --asic --bus --vbios --driver --vram")
        stdout = stdout.strip()
        if ret_code == 0  and stdout:
            self.logger.debug(f"amd-smi static : \n {stdout}")
            results["static"] = "Passed"
            smi_static_dict = self._convert_string_to_dict(stdout)
        else:
            self.logger.warning(f"!!! amd-smi static failed: {stderr}")
            results["static"] = "Failed"

        # Update static information in gpu_info_dict
        if smi_static_dict:
            for gpu_key, gpu_data in self.gpu_info_dict.items():
                if gpu_key in smi_static_dict:
                    # Merge static information into the existing GPU data
                    gpu_data.update(smi_static_dict[gpu_key])

        # Check firmware option
        stdout, stderr, ret_code = run_command("amd-smi firmware")
        stdout = stdout.strip()
        if ret_code == 0 and  stdout:
            self.logger.debug(f"amd-smi firmware: \n {stdout}")
            results["firmware"] = "Passed"
            # Store firmware info in gpu_fw_info_dict
            # Format the string to make it valid YAML
            self.gpu_fw_info_dict = self._convert_string_to_dict(stdout)
        else:
            self.logger.warning(f"!!! amd-smi firmware failed: {stderr}")
            results["firmware"] = "Failed"

        # Check if any tests failed
        if "Failed" in results.values():
            self.logger.error(f"Some amd-smi commands failed: {results}")
            return TestStatus.FAIL.value, f"Some amd-smi commands failed: {results}"

        return TestStatus.PASS.value, f"amd-smi tests passed: {', '.join(k for k in results.keys())}"

    def _convert_string_to_dict(self, stdout_str):
        """Convert a string to a valid YAML format and return as a dictionary"""
        # Format the string to make it valid YAML
        # Need to add quotes around the GPU key to make it a string
        try:
            # Replace 'GPU: 0' with 'GPU_0:' to make it a valid YAML key
            valid_yaml_str = re.sub(r'GPU: (\d+)', r'"GPU_\1":', stdout_str)

            # Use a custom loader to preserve all values as strings
            class StringPreservingLoader(yaml.SafeLoader):
                pass

            # Override the resolver to treat all scalar values as strings
            def string_constructor(loader, node):
                return str(loader.construct_scalar(node))

            # Register our custom string constructor for all scalar values
            StringPreservingLoader.add_constructor(
                yaml.resolver.Resolver.DEFAULT_SCALAR_TAG,
                string_constructor
            )

            # Disable YAML's type inference by overriding all the resolvers
            # This will prevent YAML from identifying hex/integers/etc.
            StringPreservingLoader.yaml_implicit_resolvers = {}

            # Parse YAML with our custom loader
            return yaml.load(valid_yaml_str, StringPreservingLoader)

        except yaml.YAMLError as e:
            self.logger.error(f"Error converting string to YAML: {e}")
            return {}

    def test_check_lib_dependencies(self):
        """Check library dependencies of installed ROCm components"""

        # Determine ROCm installation path
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_lib_path = os.path.join(rocm_path, "lib")

        max_depth = os.environ.get("LIBDIR_MAX_DEPTH", "")
        self.logger.debug(f"-- Env LIBDIR_MAX_DEPTH = {max_depth}")
        max_depth_arg = f"-maxdepth {max_depth}" if max_depth else ""

        if not os.path.exists(rocm_lib_path):
            self.logger.error(f"!!! ROCm library path not found: {rocm_lib_path}")
            return TestStatus.FAIL.value, "ROCm library path not found."

        # Get list of libraries in the ROCm path
        stdout, stderr, ret_code = run_command(f"find {rocm_lib_path} {max_depth_arg} -name '*.so*'", shell=True)
        if ret_code != 0:
            self.logger.error(f"--Error finding libraries in {rocm_lib_path}: \n{stderr}")
            return TestStatus.FAIL.value, f"Error finding libraries: {stderr}"

        libraries = stdout.strip().split('\n')
        if not libraries:
            self.logger.warning("!!! No libraries found in ROCm library path.")
            return TestStatus.NOT_TESTED.value, "No libraries found in ROCm library path."

        # Check libraries in the ROCm library path
        # Check its dependencies as well.
        self.logger.info(f"--Checking {len(libraries)} library files in ROCm library path: {rocm_lib_path}...")
        self.logger.info(f"--Checking shared library dependencies and its linked path...")
        missing_deps, wrong_path_warnings = self._check_rocm_libs_dependency(libraries, rocm_lib_path)

        # Log any warnings about libraries linked outside of ROCm library path
        if wrong_path_warnings:
            self.logger.warning(f"!!! Found {len(wrong_path_warnings)} warnings : rocm library path linked to outside of ROCm lib PATH. \n")
            self.logger.debug(f"!!! : \n{json.dumps(wrong_path_warnings, indent=2)}")

        # If there are any missing dependencies, log them and return failure
        if missing_deps:
            self.logger.error(f"!!! Found library dependency issues: \n{json.dumps(missing_deps, indent=2)}")
            return TestStatus.FAIL.value, f"Found library dependency issues."

        if wrong_path_warnings:
            return TestStatus.PASS.value, f"{len(wrong_path_warnings)} Path warnings are found. But all library dependencies are satisfied."
        else:
            return TestStatus.PASS.value, "All library dependencies are satisfied."

    def _check_rocm_libs_dependency(self, libraries, rocm_lib_path):
        missing_deps = {}
        wrong_path_warnings = {}

        # get the actual rocm lib path without symlink
        real_rocm_lib_path = os.path.realpath(rocm_lib_path)

        # create a list of rocm libraries basenames
        rocm_lib_basenames = [os.path.basename(lib) for lib in libraries]

        # Check each library with ldd
        for lib in libraries:
            missing = []
            path_warnings = []

            if not os.path.exists(lib):
                continue

            if os.path.islink(lib):
                # Resolve symlink to get actual library path
                rplib = os.path.realpath(lib)

                if not os.path.exists(rplib):
                    self.logger.debug(f"!!! Library symlink {lib} points to a non-existent file <{rplib}>.")
                    continue

                # Check if the symlink is within the ROCm library path
                if not (rplib.startswith(real_rocm_lib_path) or rplib.startswith(rocm_lib_path)):
                    wrong_path_warnings[lib] = f"Library symlink pointing to ->{rplib} ; outside of ROCm library path {rocm_lib_path}."
                    self.logger.debug(f"!!! Library symlink {lib}->{rplib} ; pointing outside of ROCm library path {rocm_lib_path}.")
                continue

            stdout, stderr, ret_code = run_command(f"ldd {lib}", shell=True)
            # Check if its not a dynamic library
            if "not a dynamic executable" in stderr:
                continue

            if ret_code != 0:
                missing_deps[lib] = f"Error running ldd: {stderr}"
                continue

            self.logger.debug(f"----Checking dependencies & link paths for {lib}...")
            # Parse ldd output for any libraries that are not found in the system
            # and for any linked libraries that are not in the ROCm library path and raise the warning.
            for line in stdout.splitlines():
                if "not found" in line:
                    missing.append(line.strip())
                elif "=>" in line:
                    # Ex: "libamdhip64.so => /opt/rocm/lib/libamdhip64.so (0x00007f8c3c000000)"
                    # Check if the library is outside of the ROCm library path
                    parts = line.split("=>")
                    if len(parts) > 1:
                        dep_lib = parts[0].strip()
                        dep_lib_path = parts[1].strip().split()[0]
                        # dep_lib_path can be relative path, so we need to resolve it
                        # Check if the link is relative or absolute
                        if not os.path.isabs(dep_lib_path):
                            # If it's relative, resolve it against the library path
                            # normalize the path to remove any redundant separators
                            dep_lib_path = os.path.normpath(os.path.join(os.path.dirname(lib), dep_lib_path))

                        # check if the lib is a ROCm library, else # skip the check
                        if dep_lib in rocm_lib_basenames:
                            # If the dependency path is not within the ROCm library path, raise a warning
                            # Check if dep_lib_path starts with rocm_lib_path(/opt/rocm/lib/) or real_rocm_lib_path(/opt/rocm-7.0.0/lib/) without symlink.
                            if not (dep_lib_path.startswith(rocm_lib_path) or dep_lib_path.startswith(real_rocm_lib_path)):
                                # self.logger.debug(f"!!! Library {dep_lib} is linked to {dep_lib_path} which is outside of ROCm library path {rocm_lib_path}.")
                                path_warnings.append(f"Library {dep_lib} is linked to {dep_lib_path} which is outside of ROCm library path {rocm_lib_path}.")

            if missing:
                missing_deps[lib] = missing
            if path_warnings:
                wrong_path_warnings[lib] = path_warnings

        return missing_deps, wrong_path_warnings

    def test_check_kernel_parameters(self):
        """Check ROCm-related environment variables and system settings"""

        self.logger.info("--Checking kernel params/environment settings for ROCm...")
        warnings = 0
        errors = 0

        # 1. Check kernel parameters using data-driven approach
        self.logger.info("----Checking kernel parameters...")

        # Define kernel parameter checks
        kernel_param_checks = [
            {
                "name": "numa_balancing",
                "description": "numa_balancing setting",
                "file_path": "/proc/sys/kernel/numa_balancing",
                "expected_value": "0",
                "check_type": "file_content",  # file_content or cmdline_param
                "error_message": "numa_balancing is not disabled. For optimal performance, set numa_balancing=0",
                "warning_message": None,
                "is_error": True  # True for error, False for warning
            },
            {
                "name": "amd_iommu",
                "description": "amd_iommu & iommu settings",
                "file_path": "/proc/cmdline",
                "expected_value": "amd_iommu=on",
                "check_type": "cmdline_param",
                "error_message": "amd_iommu=on is not set in kernel parameters",
                "warning_message": None,
                "is_error": True
            },
            {
                "name": "iommu",
                "description": "amd_iommu & iommu settings",
                "file_path": "/proc/cmdline",
                "expected_value": "iommu=pt",
                "check_type": "cmdline_param",
                "error_message": "iommu=pt is not set in kernel parameters",
                "warning_message": None,
                "is_error": True
            },
            {
                "name": "pci_realloc",
                "description": "pci=realloc=off settings",
                "file_path": "/proc/cmdline",
                "expected_value": "pci=realloc=off",
                "check_type": "cmdline_param",
                "error_message": "pci=realloc=off is not set in kernel parameters",
                "warning_message": None,
                "is_error": True
            },
            {
                "name": "cwsr_enable",
                "description": "Compute Wavefront Save and Restore [CWSR] settings",
                "file_path": "/sys/module/amdgpu/parameters/cwsr_enable",
                "expected_value": "0",
                "check_type": "file_content",
                "error_message": None,
                "warning_message": "amdgpu.cwsr_enable is set, should be 0 for optimal performance",
                "is_error": False
            }
        ]

        # Process each kernel parameter check
        for check in kernel_param_checks:
            self.logger.info(f"------Checking {check['description']}...")
            try:
                actual_value = None

                # Read the file if it exists
                if os.path.exists(check['file_path']):
                    with open(check['file_path'], 'r') as f:
                        file_content = f.read().strip()

                    actual_value = file_content

                # Evaluate the check
                check_passed = False
                if actual_value is not None:
                    if check['check_type'] == 'file_content':
                        check_passed = (actual_value == check['expected_value'])
                    elif check['check_type'] == 'cmdline_param':
                        check_passed = (check['expected_value'] in actual_value)

                # Handle failed checks
                if not check_passed:
                    if check['is_error'] and check['error_message']:
                        self.logger.error(f"!!! {check['error_message']}")
                        errors += 1
                    elif not check['is_error'] and check['warning_message']:
                        self.logger.warning(f"!!! {check['warning_message']}")
                        warnings += 1

            except Exception as e:
                self.logger.warning(f"!!! Error checking {check['name']}: {str(e)}")
                warnings += 1

        # 2. Check Large BAR is enabled - should be enabled for better performance
        self.logger.info("----Checking Large BAR setting...")
        try:
            large_bar_enabled = True
            error, warning = self._check_large_bar()
            errors += error
            warnings += warning

        except Exception as e:
            self.logger.warning(f"!!! Error checking BAR setting for GPU devices: {str(e)}")
            warnings += 1

        # Return results
        if errors > 0:
            return TestStatus.FAIL.value, f"{errors}  Errors & {warnings} warnings detected in kernel parameters/environment settings."
        elif warnings > 0:
            return TestStatus.PASS.value, f"{warnings} warnings detected in kernel parameters/environment settings."
        else:
            return TestStatus.PASS.value, "All kernel parameters/environment settings for ROCm appear to be configured correctly"

    def _check_large_bar(self):
        """Check if Large BAR is enabled for all GPUs in the system"""

        # read the GPUs VRAM total size from /sys/class/drm/card*/device/mem_info_vram_total
        # read the CPUs VRAM visible size from /sys/class/drm/card*/device/mem_info_vis_vram_total
        # if it has the same value, then large BAR is enabled.
        # Check this for all the GPUs in the system
        errors = 0
        warnings = 0

        # Get all GPU devices
        gpu_devices = glob.glob("/sys/class/drm/card*/device")
        if not gpu_devices:
            self.logger.error("!!! No GPU devices found.")
            errors += 1
            return errors, warnings

        for device_path in gpu_devices:
            card_num = os.path.basename(os.path.dirname(device_path))
            vram_total_path = os.path.join(device_path, "mem_info_vram_total")
            vis_vram_total_path = os.path.join(device_path, "mem_info_vis_vram_total")
            unique_id_path = os.path.join(device_path, "unique_id")

            if not os.path.exists(vram_total_path) or not os.path.exists(vis_vram_total_path):
                self.logger.debug(f"!!! VRAM info files not found for {card_num}. Skipping...")
                continue

            try:
                with open(vram_total_path, 'r') as f:
                    vram_total = int(f.read().strip())
                with open(vis_vram_total_path, 'r') as f:
                    vis_vram_total = int(f.read().strip())
                with open(unique_id_path, 'r') as f:
                    unique_id = f.read().strip()

                # Format memory values for display
                vram_total_mb = vram_total / (1024*1024)
                vis_vram_total_mb = vis_vram_total / (1024*1024)

                if vram_total != vis_vram_total:
                    self.logger.warning(f"!!! Large BAR is not enabled for {card_num}[SerialNo:{unique_id}]. VRAM total: {vram_total_mb}MB, VRAM total Visible to CPU: {vis_vram_total_mb}MB")
                    warnings += 1
                else:
                    self.logger.info(f"Large BAR is enabled for {card_num}[SerialNo:{unique_id}]. VRAM total: {vram_total_mb}MB, VRAM total Visible to CPU: {vis_vram_total_mb}MB")

            except Exception as e:
                self.logger.error(f"!!! Error reading VRAM info for {device_path}: {str(e)}")
                errors += 1

        return errors, warnings


    def test_check_env_variables(self):
        """Check ROCm-related environment variables settings"""

        # Check ROCm-related environment variables
        self.logger.info("--Checking environment variables...")
        warnings = 0
        rocm_env_vars = {
            # List of ROCM stack related ENV variables here
            # if possible with its recommended value.
            # TODO : Need a single source of truth for these ENV variables.
            # have it in a yaml or json file and read it here
        }

        # Check if any of these variables are present
        found_env_vars = []
        missing_env_vars = []
        for var, default_val in rocm_env_vars.items():
            if var in os.environ:
                found_env_vars.append(f"{var}={os.environ[var]}")
            else:
                missing_env_vars.append(f"{var} (recommended: {default_val})")

        if found_env_vars:
            self.logger.info(f"------Found ROCm environment variables:\n {', '.join(found_env_vars)}")

        if missing_env_vars:
            self.logger.warning(f"!!! Missing some recommended ROCm environment variables: {', '.join(missing_env_vars)}")
            warnings += 1

        # Look for any ROCm-related environment variables not in our list
        additional_rocm_vars = []

        # TODO: Make this list more comprehensive based on actual ROCm environment variables
        rocm_env_key_words = ['ROCM', 'HIP', 'HSA', 'ROCR', 'AMD', 'GPU',  'CL_', 'OPENCL',
                              'MIOPEN', 'ROCBLAS', 'ROCSPARSE', 'ROCALUTION', 'ROCSOLVER', 'ROCRAND' ]

        # TODO: Optimize this search to avoid multiple loop search.
        for var in os.environ:
            if any(x in var.upper() for x in rocm_env_key_words):
                if var not in rocm_env_vars:
                    additional_rocm_vars.append(f"{var}={os.environ[var]}")

        if additional_rocm_vars:
            self.logger.warning(f"!!! Additional ROCm-related environment variables set :\n {'; '.join(additional_rocm_vars)}")
            warnings += 1

        # Return results
        if warnings > 0:
            return TestStatus.PASS.value, f"{warnings} warnings detected in ENV settings."
        else:
            return TestStatus.PASS.value, "All ROCm environment settings appear to be set correctly"

    def _get_nic_brands(self, nic_cards):
        """Extract unique NIC brands from the list of NIC cards"""

        nic_brands = set()
        for card in nic_cards:
            # Use regex to extract brand name after the controller type
            # Pattern: controller type [code]: Brand Name ...
            match = re.search(r'(?:Ethernet controller|Network controller|Infiniband controller)\s*\[\w+\]:\s*(\w+)', card, re.IGNORECASE)
            if match:
                brand = match.group(1)
                nic_brands.add(brand)

        # Convert to list for easier handling
        nic_brands_list = list(nic_brands)
        return nic_brands_list[0] if nic_brands_list else None

    def _check_nic_drivers(self, nic_brand):
        """Check for specific NIC drivers based on the detected NIC brand

        Args:
            nic_brand (str): The detected NIC brand (e.g., "Mellanox", "Broadcom", "HPE")

        Returns:
            tuple: (nic_drivers_found, driver_issues) - lists of found drivers and issues
        """
        nic_drivers_found = []
        driver_issues = []

        if not nic_brand:
            driver_issues.append("No NIC brand provided for driver check")
            return nic_drivers_found, driver_issues

        # Define driver mapping for different NIC brands
        driver_mapping = {
            "Mellanox": {
                "modules": ["mlx5_core", "mlx5_ib", "mlx4_core", "mlx4_ib"],
                "name": "Mellanox"
            },
            "Broadcom": {
                "modules": ["bnxt_en", "bnxt_re"],
                "name": "Broadcom"
            },
            "HPE": {
                "modules": ["cxi_core", "cxi_eth", "cxi_user"],
                "name": "HPE-Cassini"
            },
            "Cray": {
                "modules": ["cxi_core", "cxi_eth", "cxi_user"],
                "name": "HPE-Cassini"
            },
            "Cassini": {
                "modules": ["cxi_core", "cxi_eth", "cxi_user"],
                "name": "HPE-Cassini"
            },
            "Intel": {
                "modules": ["i40e", "ice", "ixgbe", "igb", "e1000e"],
                "name": "Intel"
            }
        }

        # Get driver configuration for the detected brand
        driver_config = driver_mapping.get(nic_brand)
        if not driver_config:
            driver_issues.append(f"No driver configuration found for NIC brand: {nic_brand}")
            self.logger.warning(f"!!! No driver configuration found for NIC brand: {nic_brand}")
            return nic_drivers_found, driver_issues

        # Check if the specified drivers are loaded
        for module in driver_config["modules"]:
            stdout_mod, _, ret_mod = run_command(f"lsmod | grep {module}", shell=True)
            if ret_mod == 0 and stdout_mod.strip():
                driver_name = f"{driver_config['name']}-{module}"
                nic_drivers_found.append(driver_name)
                self.logger.debug(f"--------{nic_brand} driver {module} is loaded")

        # Check if any drivers were found for this brand
        if not nic_drivers_found:
            driver_issues.append(f"{nic_brand} NIC present but drivers not loaded")
            self.logger.warning(f"!!! {nic_brand} NIC detected but drivers ({', '.join(driver_config['modules'])}) not loaded")

        return nic_drivers_found, driver_issues

    def _check_system_limits_configuration(self):
        """Check /etc/security/limits.conf for proper ulimit settings

        Returns:
            int: Number of warnings found
        """
        warnings = 0
        limits_conf_path = "/etc/security/limits.conf"

        self.logger.info("----Checking system limits configuration in /etc/security/limits.conf...")

        if not os.path.exists(limits_conf_path):
            self.logger.warning(f"!!! {limits_conf_path} not found. Cannot verify system-wide limit settings.")
            return 1

        try:
            with open(limits_conf_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            self.logger.warning(f"!!! Error reading {limits_conf_path}: {e}")
            return 1

        # Initialize tracking variables
        found_limits = {
            'soft_memlock': None,
            'hard_memlock': None,
            'soft_nofile': None,
            'hard_nofile': None
        }

        # Parse non-commented lines
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Split line into parts (domain, type, item, value)
            parts = line.split()
            if len(parts) < 4:
                continue

            domain, limit_type, item, value = parts[0], parts[1], parts[2], parts[3]

            # Check for our target limits
            if limit_type == 'soft' and item == 'memlock':
                found_limits['soft_memlock'] = value
                self.logger.info(f"--------Found soft memlock: {value} (line {line_num})")
            elif limit_type == 'hard' and item == 'memlock':
                found_limits['hard_memlock'] = value
                self.logger.info(f"--------Found hard memlock: {value} (line {line_num})")
            elif limit_type == 'soft' and item == 'nofile':
                found_limits['soft_nofile'] = value
                self.logger.info(f"--------Found soft nofile: {value} (line {line_num})")
            elif limit_type == 'hard' and item == 'nofile':
                found_limits['hard_nofile'] = value
                self.logger.info(f"--------Found hard nofile: {value} (line {line_num})")

        # Check memlock limits (should be 'unlimited')
        for limit_key in ['soft_memlock', 'hard_memlock']:
            if found_limits[limit_key] is None:
                warnings += 1
                limit_type = limit_key.split('_')[0]
                self.logger.warning(f"!!! Missing {limit_type} memlock setting in {limits_conf_path}")
                self.logger.warning(f"!!!   Add: * {limit_type} memlock unlimited")
            elif found_limits[limit_key] != 'unlimited':
                warnings += 1
                limit_type = limit_key.split('_')[0]
                self.logger.warning(f"!!! {limit_type} memlock is set to '{found_limits[limit_key]}', should be 'unlimited'")
                self.logger.warning(f"!!!   Change to: * {limit_type} memlock unlimited")

        # Check nofile limits (should be >= 1048576)
        for limit_key in ['soft_nofile', 'hard_nofile']:
            if found_limits[limit_key] is None:
                warnings += 1
                limit_type = limit_key.split('_')[0]
                self.logger.warning(f"!!! Missing {limit_type} nofile setting in {limits_conf_path}")
                self.logger.warning(f"!!!   Add: * {limit_type} nofile 1048576")
            else:
                try:
                    nofile_value = int(found_limits[limit_key])
                    if nofile_value < 1048576:
                        warnings += 1
                        limit_type = limit_key.split('_')[0]
                        self.logger.warning(f"!!! {limit_type} nofile is set to {nofile_value}, should be >= 1048576")
                        self.logger.warning(f"!!!   Change to: * {limit_type} nofile 1048576")
                except ValueError:
                    warnings += 1
                    limit_type = limit_key.split('_')[0]
                    self.logger.warning(f"!!! {limit_type} nofile has invalid value '{found_limits[limit_key]}', should be >= 1048576")
                    self.logger.warning(f"!!!   Change to: * {limit_type} nofile 1048576")

        return warnings

    def test_check_multinode_cluster_readiness(self):
        """Test if this node is enabled for multinode cluster"""
        self.logger.info("--Checking if this node is enabled for multinode cluster...")
        errors = 0
        warnings = 0
        cluster_readiness_issues = []

        # 1. Check if mpirun command is in the PATH environment
        self.logger.info("----Checking MPI availability...")
        stdout, stderr, ret_code = run_command("which mpirun")
        if ret_code != 0:
            warnings += 1
            self.logger.warning("!!! mpirun is not found in PATH. Install OpenMPI or MPICH.")
        else:
            # Get MPI version for additional info
            stdout_ver, _, _ = run_command("mpirun --version")
            mpi_version = stdout_ver.split('\n')[1] if stdout_ver else "Unknown version"
            self.logger.info(f"------Found MPI: {mpi_version}")

        # 2. Check if network cards (NICs) are present in hardware list
        self.logger.info("----Checking for network interface cards...")
        nic_brand = None
        nic_cards, stderr, ret_code = run_command("lspci -nn | grep -i 'ethernet\|network\|infiniband'", shell=True)
        if ret_code != 0 or not nic_cards.strip():
            errors += 1
            cluster_readiness_issues.append("No network cards found in hardware")
            self.logger.error("!!! No Ethernet/Network cards found in the system. This node cannot work as part of a multinode cluster setup.")
        else:
            nic_cards = nic_cards.strip().split('\n')
            self.logger.info(f"------Found {len(nic_cards)} network card(s)")
            for idx, card in enumerate(nic_cards):
                self.logger.debug(f"--------NIC {idx}: {card.strip()}")

            nic_brand = self._get_nic_brands(nic_cards)

        if nic_brand:
            self.logger.info(f"------Detected NIC brand: {nic_brand}")
        else:
            self.logger.warning("!!! Could not extract brand names from NIC information")


        # 3. Check for specific NIC drivers (Mellanox, Broadcom, HPE Cray/Cassini)
        self.logger.info("----Checking NIC drivers...")
        nic_drivers_found, driver_issues = self._check_nic_drivers(nic_brand)

        if nic_drivers_found:
            self.logger.info(f"------Active NIC drivers: {', '.join(nic_drivers_found)}")
        else:
            errors += 1
            cluster_readiness_issues.append("No high-performance NIC drivers loaded")
            self.logger.error("!!! No high-performance NIC drivers detected")

        # Add driver issues to warnings count
        warnings += len([issue for issue in driver_issues if "not loaded" in issue])

        # 4. # Check for RDMA kernel modules
        self.logger.info("----Checking RDMA kernel modules...")

        rdma_modules = ["rdma_cm", "ib_core", "ib_uverbs", "rdma_ucm"]
        rdma_modules_loaded = []
        for module in rdma_modules:
            stdout_mod, _, ret_mod = run_command(f"lsmod | grep {module}", shell=True)
            if ret_mod == 0 and stdout_mod.strip():
                rdma_modules_loaded.append(module)

        if rdma_modules_loaded:
            self.logger.info(f"------RDMA modules loaded: {', '.join(rdma_modules_loaded)}")
        else:
            errors += 1
            cluster_readiness_issues.append("RDMA modules not loaded.")
            self.logger.error("!!! No RDMA kernel modules detected")

        # 5. Check RDMA link status
        self.logger.info("----Checking RDMA link...")
        stdout_rdma, stderr, ret_code = run_command("rdma link", shell=True)
        if ret_code == 0 and stdout_rdma.strip():
            self.logger.info(f"------: \n{stdout_rdma.strip()}")
        else:
            warnings += 1
            self.logger.warning("!!! No RDMA links detected. This may affect performance in a multinode cluster setup.")

        # 6 Check ulimit settings
        self.logger.info("----Checking ulimit settings...")

        ulimit_warnings = self._check_system_limits_configuration()
        if ulimit_warnings == 0:
            self.logger.info(f"------All required limits are properly configured for ulimit.")
        else:
            warnings += ulimit_warnings
            self.logger.warning(f"!!! Found {warnings} limit configuration issues for ulimit.")
            self.logger.warning(f"!!! Note: Recommended to set the [ulimit -n 1048576 and ulimit -l unlimited] ")

        # 7. Final assessment based on all checks
        self.logger.info("----Final multinode cluster readiness assessment...")

        # Performance warnings
        performance_warnings = []
        if not nic_drivers_found:
            performance_warnings.append("No high-performance NIC drivers")
        if not rdma_modules_loaded:
            performance_warnings.append("RDMA modules not loaded")

        # Generate final result
        if errors:
            error_msg = f"Multinode cluster readiness check failed with {errors} errors. Issues: {'; '.join(cluster_readiness_issues)}"
            self.logger.error(f"!!! {error_msg}")
            return TestStatus.FAIL.value, error_msg
        elif warnings:
            warning_msg = f"Found {len(nic_cards)} NICs and Found {warnings} warnings."
            self.logger.warning(f"!!! {warning_msg}")
            return TestStatus.PASS.value, warning_msg
        else:
            success_msg = f"Found {len(nic_cards)} NICs and required drivers are loaded."
            self.logger.info(f" {success_msg}")
            return TestStatus.PASS.value, success_msg

    # Example component specific tests (these should be customized for each component)
    def test_check_hipcc(self):
        """Test hipcc package"""
        # Check if hipcc is available
        stdout, stderr, ret_code = run_command("which hipcc")
        if ret_code != 0:
            return TestStatus.FAIL.value, "hipcc not found in PATH."

        # Check version of hipcc
        stdout, stderr, ret_code = run_command("hipcc --version")
        if ret_code != 0:
            return TestStatus.FAIL.value, f"hipcc version check failed: {stderr}"

        # Build and test a simple program
        # test_target_name = "hip_bit_extract"
        test_target_name = self._get_build_target("hipcc", 0)
        return self._build_target_and_run("hipcc", test_target_name)

    def test_check_hip_runtime_amd(self):
        """Test hip-runtime-amd package"""
        test_target_name = "hip_runtime_compilation"
        # test_target_name = self._get_build_target("hip-runtime-amd", 0)
        return self._build_target_and_run("hip-runtime-amd", test_target_name)

    def test_check_hipblas(self):
        """Test hipblas package"""
        # test_target_name = "hipblas_gemm_strided_batched"
        test_target_name = self._get_build_target("hipblas", 0)
        return self._build_target_and_run("hipblas", test_target_name)

    def test_check_hipfft(self):
        """Test hipfft package"""
        # test_target_name = "hipfft_plan_d2z"
        test_target_name = self._get_build_target("hipfft", 0)
        return self._build_target_and_run("hipfft", test_target_name)

    def test_check_hipcub_dev(self):
        """Test hipcub-dev package"""
        # test_target_name = "hipcub_device_radix_sort"
        test_target_name = self._get_build_target("hipcub-dev", 0)
        return self._build_target_and_run("hipcub-dev", test_target_name)

    def test_check_hipsolver(self):
        """Test hipsolver package"""
        # test_target_name = "hipsolver_gels"
        test_target_name = self._get_build_target("hipsolver", 0)
        return self._build_target_and_run("hipsolver", test_target_name)

    def test_check_rocblas(self):
        """Test rocblas package"""
        # test_target_name = "rocblas_axpy"
        test_target_name = self._get_build_target("rocblas", 0)
        return self._build_target_and_run("rocblas", test_target_name)

    def test_check_rocfft(self):
        """Test rocfft package"""
        # test_target_name = "rocfft_callback"
        test_target_name = self._get_build_target("rocfft", 0)
        return self._build_target_and_run("rocfft", test_target_name)

    def test_check_rocprim_dev(self):
        """Test rocprim package"""
        # test_target_name = "rocprim_block_sum"
        test_target_name = self._get_build_target("rocprim-dev", 0)
        return self._build_target_and_run("rocprim-dev", test_target_name)

    def test_check_rocrand(self):
        """Test rocrand package"""
        # test_target_name = "rocrand_simple_distributions_cpp"
        test_target_name = self._get_build_target("rocrand", 0)
        return self._build_target_and_run("rocrand", test_target_name)

    def test_check_rocsolver(self):
        """Test rocsolver package"""
        # test_target_name = "rocsolver_getf2"
        test_target_name = self._get_build_target("rocsolver", 0)
        return self._build_target_and_run("rocsolver", test_target_name)

    def test_check_rocsparse(self):
        """Test rocsparse package"""
        # test_target_name = "rocsparse_bsrmv"
        test_target_name = self._get_build_target("rocsparse", 0)
        return self._build_target_and_run("rocsparse", test_target_name)

    def test_check_rocthrust_dev(self):
        """Test rocthrust package"""
        #test_target_name = "rocthrust_norm"
        test_target_name = self._get_build_target("rocthrust-dev", 0)
        return self._build_target_and_run("rocthrust-dev", test_target_name)


    def _get_build_target(self, comp_name, item_index=0):
        """Get a build target for the specified component.

        Args:
            comp_name (str): Component name (e.g., rocblas, hipfft)
            item_index (int, optional): Index of target to retrieve. Defaults to 0.

        Returns:
            str or None: Target name at the specified index or None if not found
        """
        # Handle special cases for component name mapping
        component_mapping = {
            "hipcc": "hip",
            "hip-runtime-amd": "hip",
            "hipcub-dev": "hipcub",
            "rocprim-dev": "rocprim",
            "rocthrust-dev": "rocthrust",
            # Add more mappings as needed
        }

        # Get the actual component key to use
        comp_key = component_mapping.get(comp_name, comp_name)

        # Check if the component exists and has targets
        if comp_key in self.rocm_examples_targets and len(self.rocm_examples_targets[comp_key]) > item_index:
            return self.rocm_examples_targets[comp_key][item_index]

        return None

    def _build_target_and_run(self, comp_name, test_target_name):
        """Build and run a specific target from rocm-examples

        Args:
            comp_name: Component name (e.g., 'rocblas', 'hipfft')
            test_target_name: Target name for cmake/ctest (e.g., 'rocblas_axpy')

        Returns:
            tuple: (TestStatus, message)
        """
        self.logger.info(f"--Checking {comp_name} with a simple program [{test_target_name}]...")
        stdout, stderr, ret_code = run_command(
            f"cmake --build build --target {test_target_name}; ctest --test-dir build -R \"^{test_target_name}$\"", shell=True)
        self.logger.debug(f"\n{stdout.strip()}")
        if ret_code != 0:
            self.logger.error(f"--Failed to compile rocm-examples ({test_target_name}): \n{stderr}")
            return TestStatus.FAIL.value, f"{comp_name} check failed: {stderr}"
        else:
            self.logger.debug(f"--Successfully executed {test_target_name}.")

        return TestStatus.PASS.value, f"{comp_name} is working."

    def test_check_miopen_hip(self):
        """Test miopen-hip package"""
        # Find ROCM path
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        miopen_driver = os.path.join(rocm_path, "bin", "MIOpenDriver")

        # Check if MIOpenDriver exists
        if not os.path.exists(miopen_driver):
            return TestStatus.NOT_INSTALLED.value, "MIOpenDriver not found"

        self.logger.info("--Checking MIOpen with MIOpenDriver utility...")
        test_results = []

        # Test 1: Simple convolution test
        self.logger.debug("----Checking MIOpen convolution with default parameters...")
        conv_cmd = f"{miopen_driver} conv"
        stdout, stderr, ret_code = run_command(conv_cmd, shell=True)
        if ret_code != 0:
            self.logger.error(f"!!!! MIOpen convolution test failed: \n{stderr}")
            test_results.append(("Convolution", False, stderr))
        else:
            self.logger.debug("----MIOpen convolution test passed.")
            test_results.append(("Convolution", True, ""))

        # Test 2: Pooling test
        self.logger.debug("----Checking MIOpen pooling with default parameters...")
        pool_cmd = f"{miopen_driver} pool"
        stdout, stderr, ret_code = run_command(pool_cmd, shell=True)
        if ret_code != 0:
            self.logger.error(f"!!!! MIOpen pooling test failed: \n{stderr}")
            test_results.append(("Pooling", False, stderr))
        else:
            self.logger.debug("----MIOpen pooling test passed.")
            test_results.append(("Pooling", True, ""))

        # Test 3: Activation test
        self.logger.debug("----Checking MIOpen activation test with default parameters...")
        activ_cmd = f"{miopen_driver} activ -m relu"
        stdout, stderr, ret_code = run_command(activ_cmd, shell=True)
        if ret_code != 0:
            self.logger.error(f"!!!! MIOpen activation test failed: \n{stderr}")
            test_results.append(("Activation", False, stderr))
        else:
            self.logger.debug("----MIOpen activation test passed.")
            test_results.append(("Activation", True, ""))

        # Evaluate overall results
        failed_tests = [test[0] for test in test_results if not test[1]]
        if failed_tests:
            return TestStatus.FAIL.value, f"MIOpen tests failed for: {', '.join(failed_tests)}"
        else:
            return TestStatus.PASS.value, "MIOpen is working correctly for basic operations"

    def test_component(self, component):
        """Test a specific component by dynamically calling the appropriate test function"""
        test_method_name = f"test_check_{component.replace('-', '_').replace('+', '_plus_')}"
        test_method = getattr(self, test_method_name, None)

        if component in self.exclude_list:
            return TestStatus.NOT_TESTED.value, f"{component} is in exclude list."

        if component not in self.installed_components:
            return TestStatus.NOT_INSTALLED.value, f"{component} is not installed."

        if test_method:
            return test_method()
        else:
            # Default test for components without specific tests
            return self.test_check_basic_component(component)

    def test_check_basic_component(self, component):
        """Basic test for components without specific test methods"""
        # Check if component packge files installed
        # TODO
        return TestStatus.PASS.value, f"{component} is installed but no specific test available."

    def run_default_tests(self):
        """Run the default set of tests"""
        results = {}

        # Test 1: GPU Presence
        self.logger.info("Running test: GPU Presence...")
        status, reason = self.test_GPUPresence()
        results["gpu_presence"] = {"status": status, "reason": reason}

        # Test 2: AMDGPU Driver
        self.logger.info("Running test: AMDGPU Driver...")
        status, reason = self.test_amdgpu_driver()
        results["amdgpu_driver"] = {"status": status, "reason": reason}

        # Test 3: Kernel Parameters
        self.logger.info("Running test: Kernel Parameters...")
        status, reason = self.test_check_kernel_parameters()
        results["kernel_parameters"] = {"status": status, "reason": reason}

        # Test 4: rocminfo
        self.logger.info("Running test: rocminfo...")
        status, reason = self.test_rocminfo()
        results["rocminfo"] = {"status": status, "reason": reason}

        # Test 5: rocm_agent_enumerator
        self.logger.info("Running test: rocm_agent_enumerator...")
        status, reason = self.test_rocm_agent_enumerator()
        results["rocm_agent_enumerator"] = {"status": status, "reason": reason}

        # Test 6: amd-smi
        self.logger.info("Running test: amd-smi...")
        status, reason = self.test_amd_smi()
        results["amd_smi"] = {"status": status, "reason": reason}

        # Test 7: Library Dependencies
        self.logger.info("Running test: Library Dependencies...")
        status, reason = self.test_check_lib_dependencies()
        results["lib_dependencies"] = {"status": status, "reason": reason}

        # Test 8: Environment Variables
        self.logger.info("Running test: ENV variables...")
        status, reason = self.test_check_env_variables()
        results["env_variables"] = {"status": status, "reason": reason}

        # Test 9: Multinode cluster readiness
        self.logger.info("Running test: Multinode cluster readiness...")
        status, reason = self.test_check_multinode_cluster_readiness()
        results["Multinode_Readiness"] = {"status": status, "reason": reason}
        return results

    def run_component_tests(self):
        """Run tests for installed components"""
        results = {}

        for component in self.installed_components:
            if component not in self.exclude_list:
                self.logger.info(f"Running component test: {component}...")
                status, reason = self.test_component(component)
                results[component] = {"status": status, "reason": reason}

        return results

    def run_applications_tests(self):
        """Run tests for applications in rocm-examples"""
        results = {}

        # Check if rocm-examples targets are available
        if not self.rocm_examples_targets:
            return {"applications": {"status": TestStatus.NOT_TESTED.value, "reason": "No rocm-examples targets available for applications."}}

        # Run tests for each application target
        for target in self.rocm_examples_targets.get("applications", []):
            self.logger.info(f"Running application test: [ {target} ]...")
            status, reason = self._build_target_and_run(target, target)
            results[target] = {"status": status, "reason": reason}

        return results

    def run_tests(self, run_all=False, temp_dir="/tmp/rdhc/"):
        """Run tests based on the run_all flag"""
        # Always run default tests
        self.results = self.run_default_tests()

        # Run component tests if run_all is True
        if run_all:
            # Clone and configure rocm-examples repository if its not already done.
            # self.logger.info("Cloning rocm-examples repository...")

            # Store original directory
            original_dir = os.getcwd()

            try:
                # Ensure temp directory exists
                os.makedirs(temp_dir, exist_ok=True)

                # Check if rocm-examples already exists
                examples_dir = os.path.join(temp_dir, "rocm-examples")
                if not os.path.exists(examples_dir):
                    # Navigate to temp directory
                    os.chdir(temp_dir)

                    # Clone repository
                    self.logger.info("Cloning rocm-examples repository...")
                    stdout, stderr, ret_code = run_command(
                        "git clone https://github.com/ROCm/rocm-examples.git", shell=True)
                    if ret_code != 0:
                        self.logger.error(f"Failed to clone rocm-examples: \n{stderr}")
                    else:
                        self.logger.info("Successfully cloned rocm-examples repository.")
                else:
                    self.logger.info("rocm-examples repository already exists, skipping git clone.")

                # Navigate to the repository directory
                os.chdir(examples_dir)

                # Check if build directory exists
                if not os.path.exists(os.path.join(examples_dir, "build")):
                    # Configure with cmake
                    self.logger.info("Configuring rocm-examples with cmake...")
                    stdout, stderr, ret_code = run_command(
                        "cmake -S . -B build")
                    if ret_code != 0:
                        self.logger.error(f"Failed to configure rocm-examples: \n{stderr}")
                    else:
                        self.logger.info("Successfully configured rocm-examples.")
                else:
                    self.logger.info("rocm-examples build directory already exists, skipping cmake configuration.")

                # Get the avilabale build targets dynamically.
                self.logger.info("Retrieving available build targets...")
                stdout, stderr, ret_code = run_command(
                    "cmake --build build --target help", shell=True)
                if ret_code != 0:
                    self.logger.error(f"Failed to retrieve build targets: \n{stderr}")
                else:
                    # Parse the output to find targets
                    self.rocm_examples_targets = self._parse_rocm_example_targets(stdout)
                    self.logger.debug(f"Available build targets from rocm-examples source:\n{json.dumps(self.rocm_examples_targets, indent=2)}")

            except Exception as e:
                self.logger.error(f"Error during rocm-examples setup: \n{str(e)}")
            finally:
                # Run component tests
                component_results = self.run_component_tests()
                self.results.update(component_results)

                # Run Simple Application tests
                app_results = self.run_applications_tests()
                self.results.update(app_results)

                # Return to original directory
                os.chdir(original_dir)


        return self.results

    def _parse_rocm_example_targets(self, cmake_target_help_output):
        """Parse cmake target help output and categorize targets by component.

        Args:
            cmake_target_help_output (str): Output from 'cmake --build build --target help'

        Returns:
            dict: Dictionary with component names as keys and lists of targets as values
        """
        # Initialize the result dictionary
        component_targets = {}

        # Split the output into lines
        lines = cmake_target_help_output.strip().split('\n')

        # Process each line
        for line in lines:
            line = line.strip()

            if not line.startswith("..."):
                continue

            # Remove the "..." prefix
            target = line.replace("...", "").strip()

            # Skip special targets without underscore
            if "_" not in target:
                continue

            # Skip certain special targets
            if target in ["list_install_components", "edit_cache", "rebuild_cache"]:
                continue

            # Extract component name (part before the first underscore)
            component = target.split("_")[0]

            # Add target to the appropriate component list
            if component not in component_targets:
                component_targets[component] = []
            component_targets[component].append(target)

        return component_targets

# =======================================================================================

def setup_logger(verbose=False, silent=False):
    """Setup the logger with appropriate log level"""
    log_level = logging.ERROR if silent else (logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger("RDHC")
    logger.setLevel(log_level)

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ROCm Deployment Health Check Tool",
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    usage="sudo -E ./rdhc.py [options]",
                                    epilog="Refer the README @<ROCM_INSTALL_PATH>/share/rdhc/README.md \n"+
                                        "Usage examples:\n"+
                                        "# Run quick test (default tests only)\n" +
                                        "sudo -E ./rdhc.py\n" +
                                        "\n"+
                                        "# Run all tests including compile and execute the rocm-example program for each component\n"+
                                        "sudo -E ./rdhc.py --all\n" +
                                        "\n"+
                                        "# Run all tests with verbose output\n" +
                                        "sudo -E ./rdhc.py --all -v\n" +
                                        "\n"+
                                        "# Enable verbose output\n" +
                                        "sudo -E ./rdhc.py -v\n" +
                                        "\n"+
                                        "# Run in silent mode (only errors shown)\n" +
                                        "sudo -E ./rdhc.py -s\n" +
                                        "\n"+
                                        "# Export results to a specific JSON file\n" +
                                        "sudo -E ./rdhc.py --all --json rdhc-results.json\n" +
                                        "\n"+
                                        "# Specify a directory for temp files and logs (default: /tmp/rdhc/)\n" +
                                        "sudo -E ./rdhc.py -d /home/user/rdhc-dir/\n" +
                                        " ",
                                    )

    parser.add_argument("--quick", action="store_true", help="Run quick tests only (default)")
    parser.add_argument("--all", action="store_true", help="Default tests + Compile and executes simple program for each component.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-s", "--silent", action="store_true", help="Silent mode (errors only)")
    parser.add_argument("-j", "--json", metavar="FILE", help="Export results to JSON file", default="rdhc_results.json")
    parser.add_argument("-d", "--dir", metavar="DIR", help="Directory path for temporary files (default: /tmp/rdhc/)", default="/tmp/rdhc/")
    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(args.verbose, args.silent)

    # Ensure temp directory exists
    temp_dir = args.dir
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Using temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Failed to create temporary directory {temp_dir}: {e}")
        logger.info("Falling back to current directory")
        temp_dir = "./"

    # Create the health check instance
    health_check = ROCMHealthCheck(logger)

    # Run tests with the temp_dir
    health_check.run_tests(run_all=args.all, temp_dir=temp_dir)

    # Generate and print report
    print("\nROCm Deployment Health Check Results:")
    health_check.system_info["RDHC directory"] = temp_dir
    health_check.system_info["Json output file"] = args.json

    table = generate_table_system_info(health_check.system_info)
    print(table)
    if health_check.gpu_info_dict:
        table = generate_table_gpu_info(health_check.gpu_info_dict)
        print(table)
    if health_check.gpu_fw_info_dict:
        table = generate_table_firmware_info(health_check.gpu_fw_info_dict)
        print(table)

    table = generate_table_report(health_check.results)
    print(table)

    # Export results to JSON if requested
    if args.json:
        # If json path is not absolute, place it in the specified temp directory
        json_path = args.json
        if not os.path.isabs(json_path):
            json_path = os.path.join(temp_dir, json_path)

        logger.info(f"Exporting results to JSON file: {json_path}")
        # Create a combined data dictionary with all information
        combined_data = {
            "system_info": health_check.system_info,
            "gpu_info": health_check.gpu_info_dict,
            "firmware_info": health_check.gpu_fw_info_dict,
            "test_results": health_check.results
        }
        export_to_json(combined_data, json_path)

if __name__ == "__main__":
    main()
