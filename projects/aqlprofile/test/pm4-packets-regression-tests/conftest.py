import sys
import pytest


def pytest_addoption(parser):
    parser.addoption("--gfx-name", action="store", default=None, help="GFX architecture name to filter tests")


@pytest.fixture(scope="session")
def gfx_name(pytestconfig):
    gfx_name = pytestconfig.getoption("gfx_name")
    if gfx_name is None:
        raise pytest.UsageError("--gfx-name is required to run these tests")
    return gfx_name


def pytest_generate_tests(metafunc):
    # If any of the dump file parameters are present, require --gfx-name
    needs_gfx = (
        "pmc_dump_file_name" in metafunc.fixturenames or
        "sqtt_dump_file_name" in metafunc.fixturenames or
        "spm_dump_file_name" in metafunc.fixturenames
    )
    if needs_gfx:
        gfx_name = metafunc.config.getoption("gfx_name")
        if gfx_name is None:
            raise pytest.UsageError("--gfx-name is required to run these tests")

    if "pmc_dump_file_name" in metafunc.fixturenames:
        from test_packets import pmc_dump_file_names
        file_names = pmc_dump_file_names(gfx_name)
        if not file_names:
            metafunc.parametrize("pmc_dump_file_name", ["__NO_PMC_FILES__"])
        else:
            metafunc.parametrize("pmc_dump_file_name", file_names)

    if "sqtt_dump_file_name" in metafunc.fixturenames:
        from test_packets import sqtt_dump_file_names
        file_names = sqtt_dump_file_names(gfx_name)
        if not file_names:
            metafunc.parametrize("sqtt_dump_file_name", ["__NO_SQTT_FILES__"])
        else:
            metafunc.parametrize("sqtt_dump_file_name", file_names)

    if "spm_dump_file_name" in metafunc.fixturenames:
        from test_packets import spm_dump_file_names
        file_names = spm_dump_file_names(gfx_name)
        if not file_names:
            metafunc.parametrize("spm_dump_file_name", ["__NO_SPM_FILES__"])
        else:
            metafunc.parametrize("spm_dump_file_name", file_names)


