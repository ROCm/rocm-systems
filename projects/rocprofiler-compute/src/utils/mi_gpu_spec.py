import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml

from utils.logger import console_debug, console_error, console_log, console_warning

# Constants for MI series
# NOTE: Currently supports MI50, MI100, MI200, MI300
MI50 = 0
MI100 = 1
MI200 = 2
MI300 = 3

MI_CONSTANS = {MI50: "mi50", MI100: "mi100", MI200: "mi200", MI300: "mi300"}

gpu_series_dict = {}  # key: gpu arch
gpu_model_dict = {}  # key: gpu_arch
mi300_num_xcds_dict = {}  # key: gpu model
mi300_nps_dict = {}  # key: gpu model
mi300_chip_id_dict = {}  # key: chip id (int)


# ----------------------------
# YAML Parsing and Data Handling
# ----------------------------


def load_yaml(file_path: str) -> Dict[str, Any]:
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


def parse_mi_gpu_spec():
    """
    Parse out mi gpu data from yaml file and store in memory.
    MI GPUs
      |-- series
          |-- architecture (list)
              |-- models
                  |-- chip_ids
                  |-- mi300_arch
                  |-- partition_mode
    """

    current_dir = os.path.dirname(__file__)
    yaml_file_path = os.path.join(current_dir, "mi_gpu_spec.yaml")

    # Load the YAML data
    yaml_data = load_yaml(yaml_file_path)
    mi300_models_dict = {}

    for mi_index, mi_series in MI_CONSTANS.items():
        if mi_series != MI_CONSTANS[MI300]:
            console_debug("[parse_mi_gpu_spec] Processing series: %s" % mi_series)
            for key, value in yaml_data.items():
                # parse out gpu series and gpu model information for mi50, 100, 200
                curr_gpu_arch = value[mi_index]["gpu_archs"][0]["gpu_arch"]
                gpu_series_dict[curr_gpu_arch] = mi_series
                gpu_model_dict[curr_gpu_arch] = []
                for models in value[mi_index]["gpu_archs"][0]["models"]:
                    gpu_model_dict[curr_gpu_arch].append(models["gpu_model"])
        elif mi_series == MI_CONSTANS[MI300]:
            # MI300 requires specific processing
            for key, value in yaml_data.items():
                mi300_gpu_archs_list = []
                # NOTE: only MI300 have multiple architectures
                for archs in value[MI300]["gpu_archs"]:
                    curr_gpu_arch = archs["gpu_arch"]
                    mi300_gpu_archs_list.append(curr_gpu_arch)
                    gpu_series_dict[curr_gpu_arch] = mi_series

                for idx, arch in enumerate(mi300_gpu_archs_list):
                    mi300_models_dict[arch] = []
                    for models in value[MI300]["gpu_archs"][idx]["models"]:
                        gpu_model = models["gpu_model"]

                        # 1. Parse compute partition. NOTE: compute partition mode num xcds is available for all mi300 gpu models
                        mi300_num_xcds_dict[gpu_model] = models["partition_mode"][
                            "compute_partition_mode"
                        ]["num_xcds"]

                        # 2. Parse memory_partition. NOTE: memory partition mode nps is available for all mi300 gpu models
                        mi300_nps_dict[gpu_model] = models["partition_mode"][
                            "memory_partition_mode"
                        ]

                        # 3. Parse chip id (physical and virtual).
                        if models["chip_ids"]["physical"]:
                            # save chip_id, gpu_model pair if chip id is available
                            # NOTE: chip id is available for all gfx942 machines
                            mi300_chip_id_dict[models["chip_ids"]["physical"]] = models[
                                "gpu_model"
                            ]

                        if models["chip_ids"]["virtual"]:
                            # save chip_id, gpu_model pair if chip id is available
                            # NOTE: chip id is available for all gfx942 machines
                            mi300_chip_id_dict[models["chip_ids"]["virtual"]] = models[
                                "gpu_model"
                            ]

                        mi300_models_dict[arch].append(gpu_model)

    gpu_model_dict.update(mi300_models_dict)


def get_gpu_series_dict():
    if not gpu_series_dict:
        console_error(
            "gpu_series_dict not yet populated, did you run parse_mi_gpu_spec()?"
        )
        return None
    return gpu_series_dict


def get_gpu_series(gpu_arch_):
    if not gpu_series_dict:
        console_error(
            "gpu_series_dict not yet populated, did you run parse_mi_gpu_spec()?"
        )
        return None

    # Normalize the key by checking both the raw and lowercase versions
    gpu_series = gpu_series_dict.get(gpu_arch_) or gpu_series_dict.get(gpu_arch_.lower())
    if gpu_series:
        return gpu_series.upper()

    console_warning(f"No matching gpu series found for gpu arch: {gpu_arch_}")
    return None


def get_gpu_model(gpu_arch_, chip_id_):
    # Check that gpu_model_dict is populated first
    if not gpu_model_dict:
        console_error(
            "gpu_model_dict not yet populated. Did you run parse_mi_gpu_spec()?"
        )
        return None

    gpu_arch_lower = gpu_arch_.lower()

    # Handle gfx942 with chip_id mapping
    if gpu_arch_lower == "gfx942":
        if chip_id_ and int(chip_id_) in mi300_chip_id_dict:
            gpu_model = mi300_chip_id_dict.get(int(chip_id_))
        else:
            console_warning(f"No gpu model found for chip id: {chip_id_}")
            return None

    # Otherwise use gpu_model_dict mapping for other mi architectures
    elif gpu_arch_lower in gpu_model_dict:
        # NOTE: take the first element works for now
        gpu_model = gpu_model_dict[gpu_arch_lower][0]
    else:
        console_warning(f"No gpu model found for gpu arch: {gpu_arch_lower}")
        return None

    if not gpu_model:
        console_warning(f"No gpu model found for gpu arch: {gpu_arch_lower}")
        return None

    return gpu_model.upper()


def get_mi300_num_xcds(gpu_model_, compute_partition_):
    if not mi300_num_xcds_dict:
        console_error(
            "mi300_num_xcds_dict not yet populated, did you run parse_mi_gpu_spec()?"
        )
        return None

    gpu_model_lower = gpu_model_.lower()
    partition_lower = compute_partition_.lower()

    if gpu_model_lower not in mi300_num_xcds_dict:
        return None

    model_dict = mi300_num_xcds_dict[gpu_model_lower]
    if partition_lower not in model_dict:
        console_log(f"Unknown compute partition: {compute_partition_}")
        return None

    num_xcds = model_dict[partition_lower]
    if not num_xcds:
        console_warning(
            "Unknown compute partition found for %s / %s", compute_partition_, gpu_model_
        )
        return None

    return num_xcds


def get_mi300_chip_id_dict():
    if mi300_chip_id_dict:
        return mi300_chip_id_dict
    else:
        console_error(
            "mi300_chip_id_dict not yet populated, did you run parse_mi_gpu_spec()?"
        )
