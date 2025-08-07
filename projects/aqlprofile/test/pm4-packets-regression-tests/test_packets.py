import os
import csv
import pytest

curr_script_dir_path = os.path.dirname(os.path.realpath(__file__))


def read_dump_file(file_path):
    ret = dict()
    with open(file_path, 'r') as file:
        for line in file:
            record = line[:-1]
            record_name = record[:record.find(")") + 1]
            record_data = record[len(record_name):]
            record_data = record_data.split(", ")
            ret[record_name] = record_data
    return ret

# @pytest.fixture(scope="session")
def pmc_dump_file_names(gfx_name):
    if gfx_name is None: return []
    dump_dir = os.path.join(curr_script_dir_path, 'baselines/pmc_builder_dump', gfx_name)
    if not os.path.exists(dump_dir): return []
    pmc_dump_file_names = [x for x in os.listdir(dump_dir) if "PMC_" in x and x.endswith('.txt')]
    return pmc_dump_file_names


def sqtt_dump_file_names(gfx_name):
    if gfx_name is None: return []
    dump_dir = os.path.join(curr_script_dir_path, 'baselines/sqtt_builder_dump', gfx_name)
    if not os.path.exists(dump_dir): return []
    sqtt_dump_file_names = [x for x in os.listdir(dump_dir) if "SQTT_" in x and x.endswith('.txt')]
    return sqtt_dump_file_names


def spm_dump_file_names(gfx_name):
    if gfx_name is None: return []
    dump_dir = os.path.join(curr_script_dir_path, 'baselines/spm_builder_dump', gfx_name)
    if not os.path.exists(dump_dir): return []
    spm_dump_file_names = [x for x in os.listdir(dump_dir) if "SPM_" in x and x.endswith('.txt')]
    return spm_dump_file_names


def compare_dump_files(baseline_dir_name, output_dir_dir_name, dump_file_name):
    dir_path = curr_script_dir_path
    file_path1 = os.path.join(dir_path, baseline_dir_name, dump_file_name)
    file_path2 = os.path.join(dir_path, output_dir_dir_name, dump_file_name)
    data1 = read_dump_file(file_path1)
    data2 = read_dump_file(file_path2)
    error_messages = []
    for record_name, value in data1.items():
        checks_in_record = len(value)
        failed_checks_in_record = 0
        if record_name not in data2:
            error_messages.append(f'Record {record_name} is not present in "output/{dump_file_name}".')
            break
        else:
            value2 = data2[record_name]
            if len(value2) != len(value):
                error_messages.append(f'Number of values are not same for function: "{record_name}"')
                continue
            for index in range(len(value)):
                if value2[index] != value[index]:
                    failed_checks_in_record += 1
        if failed_checks_in_record > 0:
            error_messages.append(f'{failed_checks_in_record}/{checks_in_record} checks failed for "output/{dump_file_name}".')
    if len(error_messages) > 0:
        fail_msg = (
            f"Please inspect {baseline_dir_name}/debug_trace.txt and {output_dir_dir_name}/debug_trace.txt for details."
        )
        pytest.fail(fail_msg)


def test_pmc_packets(pmc_dump_file_name, gfx_name):
    if pmc_dump_file_name == "__NO_PMC_FILES__":
        pytest.skip(f"No PMC dump files found for gfx_name={gfx_name}, skipping PMC tests.")
    compare_dump_files(
        os.path.join('baselines/pmc_builder_dump', gfx_name),
        os.path.join('output/pmc_builder_dump', gfx_name),
        pmc_dump_file_name
    )


def test_sqtt_packets(sqtt_dump_file_name, gfx_name):
    if sqtt_dump_file_name == "__NO_SQTT_FILES__":
        pytest.skip(f"No SQTT dump files found for gfx_name={gfx_name}, skipping SQTT tests.")
    compare_dump_files(
        os.path.join('baselines/sqtt_builder_dump', gfx_name),
        os.path.join('output/sqtt_builder_dump', gfx_name),
        sqtt_dump_file_name
    )


def test_spm_packets(spm_dump_file_name, gfx_name):
    if spm_dump_file_name == "__NO_SPM_FILES__":
        pytest.skip(f"No SPM dump files found for gfx_name={gfx_name}, skipping SPM tests.")
    compare_dump_files(
        os.path.join('baselines/spm_builder_dump', gfx_name),
        os.path.join('output/spm_builder_dump', gfx_name),
        spm_dump_file_name
    )