import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from utils.logger import console_debug, console_error, console_log, console_warning

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

    _gpu_series_dict = {}  # key: gpu arch
    _gpu_model_dict = {}  # key: gpu_arch
    _num_xcds_dict = {}  # key: gpu model
    _chip_id_dict = {}  # key: chip id (int)

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
                    |-- gpu model
                    |-- chip_ids
                    |-- partition_mode
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
                        cls._chip_id_dict[models["chip_ids"]["physical"]] = curr_gpu_model
                    if "chip_ids" in models and "virtual" in models["chip_ids"]:
                        cls._chip_id_dict[models["chip_ids"]["virtual"]] = curr_gpu_model

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
    def get_gpu_model(cls, gpu_arch_, chip_id_):
        # Check that gpu_model_dict is populated first
        if not cls._gpu_model_dict:
            console_error(
                "gpu_model_dict not yet populated. Did you run parse_mi_gpu_spec()?"
            )
            return None

        gpu_arch_lower = gpu_arch_.lower()

        # Handle gfx942 with chip_id mapping
        if gpu_arch_lower not in ("gfx906", "gfx908", "gfx90a"):
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
    def get_num_xcds(cls, gpu_model_, compute_partition_):
        # Only gpu in and above mi 300 series have more than one XCDs
        if gpu_model_.lower() in ("mi50", "mi60", "mi100", "mi210", "mi250", "mi250x"):
            return 1

        if not cls._num_xcds_dict:
            console_error(
                "mi300_num_xcds_dict not yet populated, did you run parse_mi_gpu_spec()?"
            )
            return None

        gpu_model_lower = gpu_model_.lower()
        partition_lower = compute_partition_.lower()

        if gpu_model_lower not in cls._num_xcds_dict:
            return None

        model_dict = cls._num_xcds_dict[gpu_model_lower]
        if partition_lower not in model_dict:
            console_log(f"Unknown compute partition: {compute_partition_}")
            return None

        num_xcds = model_dict[partition_lower]
        if not num_xcds:
            console_warning(
                "Unknown compute partition found for %s / %s",
                compute_partition_,
                gpu_model_,
            )
            return None

        return num_xcds

    @classmethod
    def get_chip_id_dict(cls):
        if cls._chip_id_dict:
            return cls._chip_id_dict
        else:
            console_error()


# pre-initialize the instance when module loads

mi_gpu_specs = MIGPUSpecs()
