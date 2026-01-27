#!/usr/bin/env python3
"""ROCm_SMI_LIB CLI Tool - AMD-SMI Backend

This tool acts as a command line interface for manipulating
and monitoring the amdgpu kernel, using the AMD-SMI library
instead of the rocm_smi_lib API.
This tool uses the amdsmi Python package to call the AMD SMI library.
Recommended: At least one AMD GPU with ROCm driver installed
Required: AMD SMI library installed (pip install amdsmi)
"""

from __future__ import print_function
import sys
import os
import logging

# Version information
SMI_MAJ = 4
SMI_MIN = 0
SMI_PAT = 0
SMI_HASH = 'amdsmi'
__version__ = '%s.%s.%s+%s' % (SMI_MAJ, SMI_MIN, SMI_PAT, SMI_HASH)

# Set to 1 if an error occurs
RETCODE = 0

# If we want JSON format output instead
PRINT_JSON = False
PRINT_CSV = False

# Output formatting
headerString = ' ROCm System Management Interface '
footerString = ' End of ROCm SMI Log '
appWidth = 106

# DRM ioctl calculation helpers
# Reference: include/uapi/asm-generic/ioctl.h from Linux kernel
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/asm-generic/ioctl.h
_IOC_NRBITS = 8      # Number of bits for command number
_IOC_TYPEBITS = 8    # Number of bits for device type
_IOC_SIZEBITS = 14   # Number of bits for data size
_IOC_DIRBITS = 2     # Number of bits for direction

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_WRITE = 1       # ioctl direction: write (userland -> kernel)

def _IOC(dir, type, nr, size):
    """Create ioctl command number
    Reference: _IOC() macro from include/uapi/asm-generic/ioctl.h
    """
    return (dir << _IOC_DIRSHIFT) | (type << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)

def _IOW(type, nr, size):
    """Create write ioctl command number
    Reference: _IOW() macro from include/uapi/asm-generic/ioctl.h
    """
    return _IOC(_IOC_WRITE, type, nr, size)

# DRM and AMDGPU ioctl constants
# Reference: include/uapi/drm/drm.h and include/uapi/drm/amdgpu_drm.h from Linux kernel
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/drm/drm.h
# https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/drm/amdgpu_drm.h
DRM_IOCTL_BASE = ord('d')  # 0x64 - DRM ioctl type character (from drm.h)
DRM_COMMAND_BASE = 0x40     # Base for driver-specific commands (from drm.h)
DRM_AMDGPU_INFO = 0x05      # AMDGPU info command number (from amdgpu_drm.h, line 51)
# Size of struct drm_amdgpu_info - we use minimal size for the ioctl call
# Reference: struct drm_amdgpu_info in include/uapi/drm/amdgpu_drm.h
DRM_AMDGPU_INFO_SIZE = 32

# Calculate DRM_IOCTL_AMDGPU_INFO
# Reference: Line 68 in amdgpu_drm.h:
# #define DRM_IOCTL_AMDGPU_INFO DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_INFO, struct drm_amdgpu_info)
DRM_IOCTL_AMDGPU_INFO = _IOW(DRM_IOCTL_BASE, DRM_COMMAND_BASE + DRM_AMDGPU_INFO, DRM_AMDGPU_INFO_SIZE)

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

def driverInitialized():
    """ Returns true if amdgpu is found in the list of initialized modules
    """
    driverInitialized = False
    if os.path.exists("/sys/module/amdgpu"):
        if os.path.exists("/sys/module/amdgpu/initstate"):
            # amdgpu is loadable module
            with open("/sys/module/amdgpu/initstate") as initstate:
                if 'live' in initstate.read():
                    driverInitialized = True
        else:
            # amdgpu is built into the kernel
            driverInitialized = True
    return driverInitialized

def find_kfd_topology_path():
    """Find KFD topology path dynamically"""
    possible_paths = [
        "/sys/class/kfd/kfd/topology/nodes",
        "/sys/devices/virtual/kfd/kfd/topology/nodes"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def find_drm_path():
    """Find DRM class path dynamically"""
    possible_paths = [
        "/sys/class/drm",
        "/sys/devices/virtual/drm"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def initializeRsmi():
    """Initialize AMD SMI library - equivalent to rocm_smi.initializeRsmi()"""
    try:
        import amdsmi
        amdsmi.amdsmi_init()
        return True
    except Exception as e:
        logging.error(f"Failed to initialize AMD SMI: {e}")
        return False

def listDevices():
    """Get list of GPU devices - equivalent to rocm_smi.listDevices()"""
    try:
        import amdsmi
        processors = amdsmi.amdsmi_get_processor_handles()
        return processors
    except Exception as e:
        logging.error(f"Failed to get processor handles: {e}")
        return []

def checkAmdGpus(deviceList):
    """Check if AMD GPUs are present - equivalent to rocm_smi.checkAmdGpus()"""
    if not deviceList:
        logging.warning('No AMD GPUs specified')
        return False
    return True

def check_runtime_pm_status(card_path):
    """
    Check if a specific GPU card is in runtime PM suspended state.
    Equivalent to rocm_smi.check_runtime_pm_status()

    Args:
        card_path: Path to the card (e.g., /sys/class/drm/card0)

    Returns:
        bool: True if suspended, False if active or runtime PM disabled
    """
    try:
        runtime_status = os.path.join(card_path, "device/power/runtime_status")
        runtime_enabled = os.path.join(card_path, "device/power/runtime_enabled")

        # Check if runtime PM is enabled
        if os.path.exists(runtime_enabled):
            with open(runtime_enabled) as f:
                enabled = f.read().strip()
                if "disabled" in enabled or "forbidden" in enabled:
                    return False

        # Check runtime status
        if os.path.exists(runtime_status):
            with open(runtime_status) as f:
                status = f.read().strip()
                return "suspended" in status

        return False
    except:
        return False

def wake_device(card_path):
    """
    Wake a GPU device from runtime PM suspended state.
    Equivalent to rocm_smi.wake_device()

    Uses DRM_IOCTL_AMDGPU_INFO ioctl to wake the device from runtime suspend.

    Args:
        card_path: Path to the card (e.g., /sys/class/drm/card0)

    Returns:
        bool: True if device was woken or already active, False on error
    """
    try:
        import re
        import fcntl
        import struct
        import errno

        # Extract card number from card_path (e.g., card0 -> 0)
        card_name = os.path.basename(card_path)
        card_match = re.match(r'card(\d+)', card_name)
        if not card_match:
            return False

        # Find renderD device in the same DRM directory
        drm_path = os.path.dirname(card_path)
        render_name = None

        # Look for renderD* in the drm directory
        for entry in os.listdir(drm_path):
            if re.match(r'renderD\d+', entry):
                # Check if this renderD is associated with our card
                # by verifying they point to the same device
                entry_link = os.path.join(drm_path, entry)
                card_link = card_path

                try:
                    entry_real = os.path.realpath(entry_link)
                    card_real = os.path.realpath(card_link)

                    # They should share the same parent device directory
                    if os.path.dirname(entry_real) == os.path.dirname(card_real):
                        render_name = entry
                        break
                except:
                    continue

        if not render_name:
            return False

        render_path = f"/dev/dri/{render_name}"

        # Open the DRM device and send ioctl to wake it
        if os.path.exists(render_path):
            try:
                fd = os.open(render_path, os.O_RDWR | os.O_CLOEXEC)

                # Create empty drm_amdgpu_info struct (all zeros)
                request = struct.pack('I' * 8, *([0] * 8))  # 32 bytes of zeros

                # Send DRM_IOCTL_AMDGPU_INFO ioctl to wake the device
                try:
                    fcntl.ioctl(fd, DRM_IOCTL_AMDGPU_INFO, request)
                except OSError as e:
                    # EINVAL is acceptable - device may already be awake
                    if e.errno != errno.EINVAL:
                        os.close(fd)
                        return False

                os.close(fd)
                return True
            except OSError:
                return False

        return False
    except:
        return False

def check_runtime_status():
    """
    Check if any GPUs are in low power state and wake them if needed.
    Equivalent to rocm_smi.check_runtime_status()

    Returns:
        bool: True if all devices are active, False if any are suspended
    """
    try:
        drm_path = find_drm_path()
        if not drm_path:
            return True
        
        all_active = True
        for card in os.listdir(drm_path):
            if card.startswith("card") and "-" not in card:
                card_path = os.path.join(drm_path, card)

                # Check if this card is suspended
                is_suspended = check_runtime_pm_status(card_path)
                if is_suspended:
                    all_active = False
                    # Try to wake the device
                    wake_device(card_path)

        return all_active
    except:
        return True

def get_bdf(processor):
    """Get BDF (Bus:Device.Function)"""
    try:
        import amdsmi
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(processor)
        return bdf
    except:
        return None

def bdf_to_location_id(bdf):
    """Convert BDF string to location_id integer used by KFD
    BDF format: "0000:44:00.0" -> location_id calculation
    """
    try:
        # Parse BDF: domain:bus:device.function
        parts = bdf.replace(':', ' ').replace('.', ' ').split()
        if len(parts) >= 4:
            domain = int(parts[0], 16)
            bus = int(parts[1], 16)
            device = int(parts[2], 16)
            function = int(parts[3], 16)
            
            # KFD location_id = (bus << 8) | (device << 3) | function
            location_id = (bus << 8) | (device << 3) | function
            return location_id
    except:
        pass
    return None

def get_device_info(processor, bdf=None):
    """Get device ID and GPU ID (GUID) from KFD topology"""
    try:
        import amdsmi
        asic_info = amdsmi.amdsmi_get_gpu_asic_info(processor)
        device_id = asic_info.get('device_id', 'N/A')
        
        # Get GPU ID from KFD topology by matching location_id
        kfd_path = find_kfd_topology_path()
        gpu_id = 'N/A'
        
        if kfd_path and os.path.exists(kfd_path) and bdf:
            location_id = bdf_to_location_id(bdf)
            if location_id is not None:
                # Look through KFD nodes to find matching location_id
                for node_dir in os.listdir(kfd_path):
                    props_file = os.path.join(kfd_path, node_dir, "properties")
                    if os.path.exists(props_file):
                        with open(props_file) as f:
                            props_dict = {}
                            for line in f:
                                if ' ' in line:
                                    parts = line.strip().split(None, 1)
                                    if len(parts) == 2:
                                        props_dict[parts[0]] = parts[1]
                            
                            # Match by location_id
                            if props_dict.get('location_id') == str(location_id):
                                gpu_id_file = os.path.join(kfd_path, node_dir, "gpu_id")
                                if os.path.exists(gpu_id_file):
                                    with open(gpu_id_file) as gf:
                                        gpu_id = gf.read().strip()
                                break
        
        return {
            'device_id': device_id,
            'guid': gpu_id
        }
    except Exception as e:
        return {'device_id': 'N/A', 'guid': 'N/A'}

def get_kfd_node_id(processor, bdf=None):
    """Get KFD node ID by matching BDF to location_id in topology"""
    try:
        import amdsmi
        
        if not bdf:
            return amdsmi.amdsmi_topo_get_numa_node_number(processor)
        
        kfd_path = find_kfd_topology_path()
        if kfd_path and os.path.exists(kfd_path):
            location_id = bdf_to_location_id(bdf)
            if location_id is not None:
                # Look through KFD nodes to find matching location_id
                for node_dir in os.listdir(kfd_path):
                    props_file = os.path.join(kfd_path, node_dir, "properties")
                    if os.path.exists(props_file):
                        with open(props_file) as f:
                            props_dict = {}
                            for line in f:
                                if ' ' in line:
                                    parts = line.strip().split(None, 1)
                                    if len(parts) == 2:
                                        props_dict[parts[0]] = parts[1]
                            
                            # Match by location_id
                            if props_dict.get('location_id') == str(location_id):
                                return int(node_dir)
        
        # Fallback to NUMA node
        return amdsmi.amdsmi_topo_get_numa_node_number(processor)
    except:
        return None

def get_temperature(processor, temp_type='edge'):
    """Get GPU temperature"""
    try:
        import amdsmi
        # Map temp_type string to enum
        temp_type_map = {
            'edge': amdsmi.AmdSmiTemperatureType.EDGE,
            'junction': amdsmi.AmdSmiTemperatureType.HOTSPOT,
            'hotspot': amdsmi.AmdSmiTemperatureType.HOTSPOT,
            'memory': amdsmi.AmdSmiTemperatureType.VRAM,
            'vram': amdsmi.AmdSmiTemperatureType.VRAM,
        }
        
        temp_enum = temp_type_map.get(temp_type.lower(), amdsmi.AmdSmiTemperatureType.EDGE)
        
        temp = amdsmi.amdsmi_get_temp_metric(
            processor,
            temp_enum,
            amdsmi.AmdSmiTemperatureMetric.CURRENT
        )
        return temp
    except:
        # Try other temperature types
        try:
            for ttype in [amdsmi.AmdSmiTemperatureType.HOTSPOT, 
                         amdsmi.AmdSmiTemperatureType.EDGE,
                         amdsmi.AmdSmiTemperatureType.VRAM]:
                try:
                    temp = amdsmi.amdsmi_get_temp_metric(
                        processor,
                        ttype,
                        amdsmi.AmdSmiTemperatureMetric.CURRENT
                    )
                    return temp
                except:
                    continue
        except:
            pass
        return None

def getTemperatureLabel(deviceList):
    """Discover the first available temperature label - matches rocm-smi priority"""
    import amdsmi
    if not deviceList:
        return 'edge'
    
    # ROCm-SMI always prefers edge temperature first, then junction, then memory
    # This matches the original rocm-smi behavior
    processor = deviceList[0]
    temp_types = [
        ('edge', amdsmi.AmdSmiTemperatureType.EDGE),
        ('junction', amdsmi.AmdSmiTemperatureType.HOTSPOT),
        ('memory', amdsmi.AmdSmiTemperatureType.VRAM),
    ]
    
    for label, temp_type in temp_types:
        try:
            temp = amdsmi.amdsmi_get_temp_metric(
                processor,
                temp_type,
                amdsmi.AmdSmiTemperatureMetric.CURRENT
            )
            if temp is not None and temp > 0:
                return label
        except:
            continue
    
    return 'edge'

def getPowerLabel(deviceList):
    """Discover the first available power label - matches rocm-smi output"""
    import amdsmi
    if not deviceList:
        return '(Avg)'
    
    # ROCm-SMI always uses "(Avg)" for Average Graphics Package Power
    # Return (Avg) to match rocm-smi behavior
    return '(Avg)'

def get_power_from_sysfs(bdf):
    """Get power from sysfs without waking GPU"""
    try:
        if not bdf:
            return None
        
        drm_path = find_drm_path()
        if not drm_path:
            return None
        
        for card in os.listdir(drm_path):
            if card.startswith("card") and "-" not in card:
                card_bdf_path = os.path.join(drm_path, card, "device/uevent")
                if os.path.exists(card_bdf_path):
                    with open(card_bdf_path) as f:
                        content = f.read()
                        if bdf in content:
                            hwmon_path = os.path.join(drm_path, card, "device/hwmon")
                            if os.path.exists(hwmon_path):
                                for hwmon in os.listdir(hwmon_path):
                                    power_file = os.path.join(hwmon_path, hwmon, "power1_average")
                                    if os.path.exists(power_file):
                                        with open(power_file) as f:
                                            power_uw = int(f.read().strip())
                                            return power_uw / 1000000.0
                            break
        return None
    except:
        return None

def get_power(processor, bdf=None):
    """Get power consumption"""
    try:
        import amdsmi
        # Try sysfs first (doesn't wake GPU)
        power_sysfs = get_power_from_sysfs(bdf)
        if power_sysfs is not None:
            return power_sysfs
        
        # Fallback to API
        power_dict = amdsmi.amdsmi_get_power_info(processor)
        if 'average_socket_power' in power_dict:
            power_w = power_dict['average_socket_power']
            if power_w != 'N/A' and isinstance(power_w, (int, float)):
                return float(power_w)
        if 'socket_power' in power_dict:
            power_w = power_dict['socket_power']
            if power_w != 'N/A' and isinstance(power_w, (int, float)):
                return float(power_w)
        return None
    except:
        return None

def get_clock_freq_from_sysfs(bdf, clk_type):
    """Get clock frequency from sysfs without waking GPU"""
    try:
        import amdsmi
        if not bdf:
            return None
        
        clk_file_map = {
            amdsmi.AmdSmiClkType.SYS: 'pp_dpm_sclk',
            amdsmi.AmdSmiClkType.MEM: 'pp_dpm_mclk',
        }
        
        if clk_type not in clk_file_map:
            return None
        
        drm_path = find_drm_path()
        if not drm_path:
            return None
        
        for card in os.listdir(drm_path):
            if card.startswith("card") and "-" not in card:
                card_bdf_path = os.path.join(drm_path, card, "device/uevent")
                if os.path.exists(card_bdf_path):
                    with open(card_bdf_path) as f:
                        content = f.read()
                        if bdf in content:
                            dpm_file = os.path.join(drm_path, card, f"device/{clk_file_map[clk_type]}")
                            if os.path.exists(dpm_file):
                                with open(dpm_file) as f:
                                    for line in f:
                                        if '*' in line:
                                            parts = line.split(':')
                                            if len(parts) >= 2:
                                                freq_str = parts[1].replace('*', '').strip().replace('Mhz', '').replace('MHz', '')
                                                return float(freq_str)
                            break
        return None
    except:
        return None

def get_clock_freq(processor, clk_type, bdf=None):
    """Get clock frequency"""
    try:
        import amdsmi
        # Try sysfs first (doesn't wake GPU)
        freq_sysfs = get_clock_freq_from_sysfs(bdf, clk_type)
        if freq_sysfs is not None:
            return freq_sysfs
        
        # Fallback to API
        freq_dict = amdsmi.amdsmi_get_clk_freq(processor, clk_type)
        if 'current' in freq_dict and 'frequency' in freq_dict:
            current_idx = freq_dict['current']
            frequencies = freq_dict['frequency']
            if current_idx < len(frequencies):
                freq_hz = frequencies[current_idx]
                return freq_hz / 1000000
        return None
    except:
        return None

def get_gpu_usage_from_sysfs(bdf):
    """Get GPU usage from sysfs without waking GPU"""
    try:
        if not bdf:
            return None
        
        drm_path = find_drm_path()
        if not drm_path:
            return None
        
        for card in os.listdir(drm_path):
            if card.startswith("card") and "-" not in card:
                card_bdf_path = os.path.join(drm_path, card, "device/uevent")
                if os.path.exists(card_bdf_path):
                    with open(card_bdf_path) as f:
                        content = f.read()
                        if bdf in content:
                            busy_file = os.path.join(drm_path, card, "device/gpu_busy_percent")
                            if os.path.exists(busy_file):
                                with open(busy_file) as f:
                                    return int(f.read().strip())
                            break
        return None
    except:
        return None

def get_gpu_usage(processor, bdf=None):
    """Get GPU utilization percentage"""
    try:
        import amdsmi
        # Try sysfs first (doesn't wake GPU)
        usage_sysfs = get_gpu_usage_from_sysfs(bdf)
        if usage_sysfs is not None:
            return usage_sysfs
        
        # Fallback to API
        usage = amdsmi.amdsmi_get_gpu_busy_percent(processor)
        return usage
    except:
        return None

def get_memory_usage(processor):
    """Get VRAM usage percentage"""
    try:
        import amdsmi
        vram_used = amdsmi.amdsmi_get_gpu_memory_usage(processor, amdsmi.AmdSmiMemoryType.VRAM)
        vram_total = amdsmi.amdsmi_get_gpu_memory_total(processor, amdsmi.AmdSmiMemoryType.VRAM)
        if vram_total > 0:
            return (vram_used / vram_total) * 100
        return None
    except:
        return None

def get_fan_speed(processor):
    """Get fan speed percentage"""
    try:
        import amdsmi
        fan_speed = amdsmi.amdsmi_get_gpu_fan_speed(processor, 0)
        return fan_speed
    except:
        # Fan not supported (e.g., OAM modules) - return 0
        return 0

def get_perf_level(processor):
    """Get performance level"""
    try:
        import amdsmi
        perf_level = amdsmi.amdsmi_get_gpu_perf_level(processor)
        if isinstance(perf_level, str):
            if 'AUTO' in perf_level:
                return 'auto'
            elif 'LOW' in perf_level:
                return 'low'
            elif 'HIGH' in perf_level:
                return 'high'
            elif 'MANUAL' in perf_level:
                return 'manual'
            elif 'STABLE_STD' in perf_level:
                return 'stable_std'
            elif 'STABLE_PEAK' in perf_level:
                return 'stable_peak'
            elif 'STABLE_MIN_MCLK' in perf_level:
                return 'stable_min_mclk'
            elif 'STABLE_MIN_SCLK' in perf_level:
                return 'stable_min_sclk'
        return perf_level
    except:
        return None

def get_power_cap(processor):
    """Get power cap in Watts"""
    try:
        import amdsmi
        power_cap_info = amdsmi.amdsmi_get_power_cap_info(processor)
        if 'power_cap' in power_cap_info:
            return power_cap_info['power_cap'] / 1000000.0
        return None
    except:
        return None

def get_memory_partition(processor):
    """Get memory partition"""
    try:
        import amdsmi
        partition = amdsmi.amdsmi_get_gpu_memory_partition(processor)
        # partition is already a string like 'NPS1', 'NPS2', etc.
        if isinstance(partition, str):
            return partition
        # If it's an enum, convert it
        partition_names = {
            amdsmi.AmdSmiMemoryPartitionType.NPS1: 'NPS1',
            amdsmi.AmdSmiMemoryPartitionType.NPS2: 'NPS2',
            amdsmi.AmdSmiMemoryPartitionType.NPS4: 'NPS4',
            amdsmi.AmdSmiMemoryPartitionType.NPS8: 'NPS8',
        }
        return partition_names.get(partition, str(partition))
    except:
        return 'N/A'

def get_compute_partition(processor):
    """Get compute partition"""
    try:
        import amdsmi
        partition = amdsmi.amdsmi_get_gpu_compute_partition(processor)
        # partition is already a string like 'SPX', 'DPX', etc.
        if isinstance(partition, str):
            return partition
        # If it's an enum, convert it
        partition_names = {
            amdsmi.AmdSmiComputePartitionType.SPX: 'SPX',
            amdsmi.AmdSmiComputePartitionType.DPX: 'DPX',
            amdsmi.AmdSmiComputePartitionType.TPX: 'TPX',
            amdsmi.AmdSmiComputePartitionType.QPX: 'QPX',
            amdsmi.AmdSmiComputePartitionType.CPX: 'CPX',
        }
        return partition_names.get(partition, str(partition))
    except:
        return 'N/A'

def printLogSpacer(displayString=None, fill='=', contentSizeToFit=0):
    """ Prints formatted spacer line

    :param displayString: String to display in the center
    :param fill: Character to use for padding
    :param contentSizeToFit: Width to fit content (overrides appWidth)
    """
    global appWidth, PRINT_JSON
    resizeValue = appWidth
    if contentSizeToFit != 0:
        resizeValue = contentSizeToFit
    if resizeValue % 2:  # if odd -> make even
        resizeValue += 1

    if not PRINT_JSON:
        if displayString:
            if len(displayString) % 2:
                displayString += fill
            logSpacer = fill * int((resizeValue - len(displayString)) / 2) + displayString + fill * int(
                (resizeValue - len(displayString)) / 2)
        else:
            logSpacer = fill * resizeValue
        print(logSpacer)

def printLog(device, logstr, logname, useItalics=False):
    """ Print out to the log

    :param device: Device to print (can be None)
    :param logstr: String to print
    :param logname: Log name (can be None)
    :param useItalics: Whether to use italics (not used in this implementation)
    """
    global PRINT_JSON
    if not PRINT_JSON:
        print(logstr)

def showAllConcise(deviceList):
    """
    Equivalent to rocm_smi.showAllConcise()
    Display critical info for all devices in a concise format
    """
    global PRINT_JSON, headerString, footerString
    import amdsmi
    
    if PRINT_JSON:
        print('NOT_SUPPORTED: Cannot print JSON/CSV output for concise output')
        sys.exit(1)
    
    silent = True
    
    # CRITICAL: Cache BDF first to minimize GPU wake-up
    bdf_cache = {}
    for i, processor in enumerate(deviceList):
        bdf_cache[i] = get_bdf(processor)
    
    # Detect temperature and power labels
    available_temp_type = getTemperatureLabel(deviceList)
    temp_label = f"({available_temp_type.capitalize()})"
    power_label = getPowerLabel(deviceList)
    
    # Header and subheader setup
    header = ['Device', 'Node', 'IDs', '', 'Temp', 'Power', 'Partitions',
              'SCLK', 'MCLK', 'Fan', 'Perf', 'PwrCap', 'VRAM%', 'GPU%']
    subheader = ['', '', '(DID,', 'GUID)', temp_label, power_label,
                 '(Mem, Compute, ID)',
                 '', '', '', '', '', '', '']
    
    # Add additional spaces to match header
    for idx, item in enumerate(subheader):
        header_size = len(header[idx])
        subheader_size = len(subheader[idx])
        if header_size != subheader_size:
            numSpacesToFill_subheader = header_size - subheader_size
            numSpacesToFill_header = subheader_size - header_size
            if numSpacesToFill_subheader > 0:
                subheader[idx] = subheader[idx] + (' ' * numSpacesToFill_subheader)
            if numSpacesToFill_header > 0:
                header[idx] = header[idx] + (' ' * numSpacesToFill_header)
    
    head_widths = [len(head) + 2 for head in header]
    values = {}
    degree_sign = u'\N{DEGREE SIGN}'
    
    # Collect data for each device
    for i, processor in enumerate(deviceList):
        bdf = bdf_cache.get(i)
        
        # Read sysfs values FIRST (before other API calls that might wake GPU)
        sclk = get_clock_freq(processor, amdsmi.AmdSmiClkType.SYS, bdf)
        sclk_str = f"{sclk:.0f}Mhz" if sclk is not None else "N/A"
        
        mclk = get_clock_freq(processor, amdsmi.AmdSmiClkType.MEM, bdf)
        mclk_str = f"{mclk:.0f}Mhz" if mclk is not None else "N/A"
        
        gpu_usage = get_gpu_usage(processor, bdf)
        gpu_str = f"{gpu_usage}%" if gpu_usage is not None else "N/A"
        
        power = get_power(processor, bdf)
        power_str = f"{power:.1f}W" if power is not None else "N/A"
        
        # Now get other info (may wake GPU but critical sysfs values already read)
        node = get_kfd_node_id(processor, bdf)
        if node is None:
            node = i
        
        device_info = get_device_info(processor, bdf)
        device_id = device_info['device_id']
        guid_str = str(device_info['guid'])
        if len(guid_str) > 10:
            guid_str = guid_str[-5:]
        
        temp = get_temperature(processor, available_temp_type)
        temp_val = f"{temp:.1f}{degree_sign}C" if temp is not None else "N/A"
        
        mem_part = get_memory_partition(processor)
        comp_part = get_compute_partition(processor)
        combined_partition_data = f"{mem_part}, {comp_part}, 0"
        
        fan = get_fan_speed(processor)
        fan_str = f"{fan}%" if fan is not None else "N/A"
        
        perf = get_perf_level(processor)
        perf_str = perf if perf is not None else "N/A"
        
        pwr_cap = get_power_cap(processor)
        pwr_cap_str = f"{pwr_cap:.1f}W" if pwr_cap is not None else "N/A"
        
        vram = get_memory_usage(processor)
        vram_str = f"{vram:.0f}%" if vram is not None else "N/A"
        
        # Store values
        values[f'card{i}'] = [i, node,
                              str(device_id) + ", ",
                              str(guid_str),
                              temp_val, power_str,
                              combined_partition_data,
                              sclk_str, mclk_str, fan_str, perf_str,
                              pwr_cap_str,
                              vram_str,
                              gpu_str]
    
    # Calculate column widths
    val_widths = {}
    for i in range(len(deviceList)):
        val_widths[i] = [len(str(val)) + 2 for val in values[f'card{i}']]
    
    max_widths = head_widths
    for i in range(len(deviceList)):
        for col in range(len(val_widths[i])):
            max_widths[col] = max(max_widths[col], val_widths[i][col])
    
    # Display concise info
    header_output = "".join(word.ljust(max_widths[col]) for col, word in zip(range(len(max_widths)), header))
    subheader_output = "".join(word.ljust(max_widths[col]) for col, word in zip(range(len(max_widths)), subheader))
    
    printLogSpacer(headerString, contentSizeToFit=len(header_output))
    printLogSpacer(' Concise Info ', contentSizeToFit=len(header_output))
    printLog(None, header_output, None)
    printLog(None, subheader_output, None, useItalics=True)
    printLogSpacer(fill='=', contentSizeToFit=len(header_output))
    
    for i in range(len(deviceList)):
        printLog(None, "".join(str(word).ljust(max_widths[col]) for col, word in
                               zip(range(len(max_widths)), values[f'card{i}'])), None)
    
    printLogSpacer(footerString, contentSizeToFit=len(header_output))

def main():
    """
    Main function - equivalent to rocm_smi.py main execution
    """
    global RETCODE
    
    # Import amdsmi
    try:
        import amdsmi
    except ImportError:
        print("ERROR: Could not import amdsmi module")
        print("Install with: pip install amdsmi")
        return 1
    
    # Check if driver is initialized
    if not driverInitialized():
        logging.error('Unable to detect amdgpu driver. Exiting...')
        return 1
    
    # Initialize AMD SMI (equivalent to rocm_smi.initializeRsmi())
    if not initializeRsmi():
        logging.error("Failed to initialize AMD SMI")
        return 1
    
    try:
        # Get processor handles (equivalent to rocm_smi.listDevices())
        deviceList = listDevices()
        
        if not deviceList:
            logging.error("No AMD GPU devices found")
            return 1
        
        # Check AMD GPUs (equivalent to rocm_smi.checkAmdGpus())
        if not checkAmdGpus(deviceList):
            logging.warning('No AMD GPUs specified')
        
        # Check runtime status (equivalent to rocm_smi.check_runtime_status())
        if not check_runtime_status():
            print("\nWARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status\n")
        
        # Call showAllConcise (equivalent to rocm_smi.showAllConcise(deviceList))
        showAllConcise(deviceList)
        
    finally:
        # Shutdown AMD SMI
        try:
            amdsmi.amdsmi_shut_down()
        except:
            pass
    
    return RETCODE

if __name__ == '__main__':
    sys.exit(main())
