##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

from utils.logger import console_debug, console_error, console_warning

# Constants for MI series
# NOTE: Currently supports MI50, MI100, MI200, MI300
MI50 = 0
MI100 = 1
MI200 = 2
MI300 = 3
MI350 = 4

MI_CONSTANS = {
    MI50: "mi50",
    MI100: "mi100",
    MI200: "mi200",
    MI300: "mi300",
    MI350: "mi350",
}

# ----------------------------
# Data Class handling to preserve the hierarchical gpu information
# ----------------------------


@dataclass
class MIGPUSpecs:
    _instance = None

    _gpu_series_dict = {}  # key: gpu_arch
    _gpu_model_dict = {}  # key: gpu_arch
    _num_xcds_dict = {}  # key: gpu_model
    _chip_id_dict = {}  # key: chip_id (int)
    _perfmon_config = {}  # key: gpu_arch

    _gpu_arch_to_compute_partition_dict = {}  # key: gpu_arch, used for gpu archs containing only one gpu model and thus one compute partition

    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize()
        return cls._instance

    @classmethod
    def _initialize(cls):
        if not cls._initialized:
            cls._parse_mi_gpu_spec()
            cls._initialized = True

    # ----------------------------
    # YAML Parsing and Data Handling
    # ----------------------------

    @classmethod
    def _load_yaml(cls, file_path: str) -> Dict[str, Any]:
        """
        Loads MI GPU YAML data /util into a Python dictionary.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            Dict[str, Any]: Parsed YAML data as a nested dictionary.
                            Exit with console error if an error occurs.
        """
        console_debug("[load_yaml]")
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                return data
        except FileNotFoundError:
            console_error(f"Error: The file '{file_path}' was not found.")
        except yaml.YAMLError as exc:
            console_error(f"Error parsing YAML file '{file_path}': {exc}")
        except Exception as e:
            console_error(
                f"An unexpected error occurred while loading YAML file '{file_path}': {e}"
            )

    @classmethod
    def _parse_mi_gpu_spec(cls):
        """
        Parse out mi gpu data from yaml file and store in memory.
        MI GPUs
        |-- series
            |-- architecture (list)
                    |-- perfmon_config
                    |-- gpu model
                    |-- chip_ids
                        | -- physical
                        | -- virtual
                    |-- partition_mode
                        | -- compute partition mode
                        | -- memory partition mode
        """

        current_dir = os.path.dirname(__file__)
        yaml_file_path = os.path.join(current_dir, "mi_gpu_spec.yaml")

        # Load the YAML data
        yaml_data = cls._load_yaml(yaml_file_path)

        for series in yaml_data["mi_gpu_spec"]:
            curr_gpu_series = series["gpu_series"]
            console_debug("[parse_mi_gpu_spec] Processing series: %s" % curr_gpu_series)
            for archs in series["gpu_archs"]:
                curr_gpu_arch = archs["gpu_arch"]
                cls._gpu_series_dict[curr_gpu_arch] = curr_gpu_series
                cls._perfmon_config[curr_gpu_arch] = archs["perfmon_config"]
                cls._gpu_model_dict[curr_gpu_arch] = []
                for models in archs["models"]:
                    curr_gpu_model = models["gpu_model"]
                    cls._gpu_model_dict[curr_gpu_arch].append(curr_gpu_model)
                    cls._num_xcds_dict[curr_gpu_model] = (
                        models.get("partition_mode", {})
                        .get("compute_partition_mode", {})
                        .get("num_xcds", {})
                    )
                    if "chip_ids" in models and "physical" in models["chip_ids"]:
                        cls._chip_id_dict[models["chip_ids"]["physical"]] = (
                            curr_gpu_model
                        )
                    if "chip_ids" in models and "virtual" in models["chip_ids"]:
                        cls._chip_id_dict[models["chip_ids"]["virtual"]] = (
                            curr_gpu_model
                        )

        # detect gpu arch to compute partition relationships
        cls._populate_gpu_arch_to_compute_partition_dict()

    @classmethod
    def _populate_gpu_arch_to_compute_partition_dict(cls):
        """
        This creates a mapping of gpu_arch -> compute_partition for architectures
        where there's only one model (and therefore one partition configuration).
        """
        for gpu_arch, gpu_models in cls._gpu_model_dict.items():
            if len(gpu_models) == 1:
                single_model = gpu_models[0]
                compute_partition = cls._num_xcds_dict.get(single_model)

                if compute_partition is not None:
                    cls._gpu_arch_to_compute_partition_dict[gpu_arch] = (
                        compute_partition
                    )
                    console_debug(
                        "[populate_single_arch_partition_dict] Single model arch found: "
                        "%s -> %s (partition: %s)"
                        % (gpu_arch, single_model, compute_partition)
                    )

    @classmethod
    def get_gpu_series_dict(cls):
        if not cls._gpu_series_dict:
            console_error(
                "gpu_series_dict not yet populated, did you run parse_mi_gpu_spec()?"
            )
            return None
        return cls._gpu_series_dict

    @classmethod
    def get_gpu_series(cls, gpu_arch_):
        if not cls._gpu_series_dict:
            console_error(
                "gpu_series_dict not yet populated, did you run parse_mi_gpu_spec()?"
            )
            return None

        # Normalize the key by checking both the raw and lowercase versions
        gpu_series = cls._gpu_series_dict.get(gpu_arch_) or cls._gpu_series_dict.get(
            gpu_arch_.lower()
        )
        if gpu_series:
            return gpu_series.upper()

        console_warning(f"No matching gpu series found for gpu arch: {gpu_arch_}")
        return None

    @classmethod
    def get_perfmon_config(cls, gpu_arch_):
        # Check that gpu_model_dict is populated first
        if not cls._perfmon_config:
            console_error(
                "gpu_model_dict not yet populated. Did you run parse_mi_gpu_spec()?"
            )
            return None

        gpu_arch_lower = gpu_arch_.lower()

        return cls._perfmon_config.get(gpu_arch_lower, None)

    @classmethod
    def get_gpu_model(cls, gpu_arch_, chip_id_):
        # Check that gpu_model_dict is populated first
        if not cls._gpu_model_dict:
            console_error(
                "gpu_model_dict not yet populated. Did you run parse_mi_gpu_spec()?"
            )
            return None

        gpu_arch_lower = gpu_arch_.lower()

        # Handle gfx942 with chip_id mapping
        if gpu_arch_lower not in ("gfx908", "gfx90a"):
            if chip_id_ and int(chip_id_) in cls._chip_id_dict:
                gpu_model = cls._chip_id_dict.get(int(chip_id_))
            else:
                console_warning(f"No gpu model found for chip id: {chip_id_}")
                return None

        # Otherwise use gpu_model_dict mapping for other mi architectures
        elif gpu_arch_lower in cls._gpu_model_dict:
            # NOTE: take the first element works for now
            gpu_model = cls._gpu_model_dict[gpu_arch_lower][0]
        else:
            console_warning(f"No gpu model found for gpu arch: {gpu_arch_lower}")
            return None

        if not gpu_model:
            console_warning(f"No gpu model found for gpu arch: {gpu_arch_lower}")
            return None

        return gpu_model.upper()

    @classmethod
    def set_default_gpu_settings(self, gpu_arch, gpu_model, compute_partition):
        """
        Set default GPU settings when model is unknown or cannot be determined.
        NOTE: This is a fallback to gfx942 settings - consider making this architecture-specific.
        """
        DEFAULT_COMPUTE_PARTITION = "SPX"
        DEFAULT_NUM_XCD = 8
        console_warning(
            f"Unable to determine xcd count from:\n\t"
            f"GPU arch: '{gpu_arch}', model: '{gpu_model}', partition: '{compute_partition}'"
        )
        console_warning(
            f"Applying default gfx942 settings:\n"
            f"\t- Compute partition: {DEFAULT_COMPUTE_PARTITION}\n"
            f"\t- Number of XCDs: {DEFAULT_NUM_XCD}"
        )

        return DEFAULT_NUM_XCD

    @classmethod
    def get_num_xcds(
        cls, gpu_arch: str = None, gpu_model: str = None, compute_partition: str = None
    ):
        """
        Retrieve the number of XCDs based on GPU architecture, model, and compute partition.

        Priority order:
        1. Legacy GPU check (returns 1 XCD for older architectures/models)
        2. Architecture-based lookup (preferred)
        3. Model + partition-based lookup (fallback)
        4. Default settings (last resort)
        """
        # Constants for legacy GPUs that don't support compute partitions
        LEGACY_ARCHS = {"gfx908", "gfx90a"}
        LEGACY_MODELS = {"mi50", "mi60", "mi100", "mi210", "mi250", "mi250x"}

        # Normalize inputs to lowercase for consistent comparison
        gpu_arch_norm = gpu_arch.lower().strip() if gpu_arch else ""
        gpu_model_norm = gpu_model.lower().strip() if gpu_model else ""
        partition_norm = compute_partition.lower().strip() if compute_partition else ""

        # 1. Return 1 XCDs for archs/models not supporting compute partition
        # NOTE: gpu arch is enough to verify this logic, gpu model is used as a backup.
        if gpu_arch_norm in LEGACY_ARCHS or gpu_model_norm in LEGACY_MODELS:
            return 1

        # 2. Try architecture-based lookup first (preferred method)
        if gpu_arch_norm and hasattr(cls, "_gpu_arch_to_compute_partition_dict"):
            arch_dict = cls._gpu_arch_to_compute_partition_dict
            if gpu_arch_norm in arch_dict:
                num_xcds = arch_dict[gpu_arch_norm]
                if num_xcds is not None:
                    return num_xcds
                else:
                    console_warning(
                        f"No compute partition data found for architecture '{gpu_arch.upper()}'"
                    )

        # 3. Fall back to model + partition-based lookup
        if gpu_model_norm:
            # Validate XCD dictionary is populated
            if not hasattr(cls, "_num_xcds_dict") or not cls._num_xcds_dict:
                console_error(
                    "mi300_num_xcds_dict not populated. Did you run parse_mi_gpu_spec()?"
                )
            elif gpu_model_norm not in cls._num_xcds_dict:
                console_warning(
                    f"Unknown gpu model provided for num xcds lookup: {gpu_model}."
                )
            else:
                model_dict = cls._num_xcds_dict[gpu_model_norm]

                if not partition_norm:
                    console_warning(
                        "No compute partition provided for model-based lookup"
                    )
                elif partition_norm not in model_dict:
                    console_warning(
                        f"Unknown compute partition '{compute_partition}' for model '{gpu_model}'"
                    )
                else:
                    num_xcds = model_dict[partition_norm]
                    if num_xcds is not None:
                        return num_xcds
                    else:
                        console_warning(
                            f"Unknown compute partition found for {compute_partition} / {gpu_model}"
                        )
        else:
            console_warning("No gpu model provided for num xcds lookup.")

        # 4. Last resort: use default settings
        return cls.set_default_gpu_settings(gpu_arch, gpu_model, compute_partition)

    @classmethod
    def get_chip_id_dict(cls):
        if cls._chip_id_dict:
            return cls._chip_id_dict
        else:
            console_error()

    @classmethod
    def get_num_xcds_dict(cls):
        if cls._num_xcds_dict:
            return cls._num_xcds_dict
        else:
            console_error()

    @classmethod
    def get_gpu_arch_to_compute_partition_dict(cls):
        return cls._gpu_arch_to_compute_partition_dict


# pre-initialize the instance when module loads
mi_gpu_specs = MIGPUSpecs()
