// Per-GPU readings of the fake amd-smi. fake_amdsmi.cc expands this table as an X-macro and
// tests/ci/fake_gpu_golden.py parses it, so the test expectations always match the fake.
//
// FAKE_VALUE(name, base, step, modulo): GPU g reports (base + step * g), wrapped at modulo
// unless modulo is 0. Units are amd-smi's. Bases are chosen so that, for up to 8 GPUs, no
// two readings of one GPU are equal and no reading equals another GPU's reading of the same
// kind, which lets a test tell which GPU and which sensor a value came from.
FAKE_VALUE(TEMP_EDGE_C, 30, 1, 0)
FAKE_VALUE(TEMP_HOTSPOT_C, 40, 1, 0)
FAKE_VALUE(TEMP_MEM_C, 50, 1, 0)
FAKE_VALUE(VOLTAGE_MV, 800, 1, 0)
FAKE_VALUE(SOCKET_POWER_W, 100, 10, 0)
FAKE_VALUE(GFX_ACTIVITY_PCT, 11, 7, 100)
FAKE_VALUE(UMC_ACTIVITY_PCT, 61, 3, 100)
FAKE_VALUE(GFX_CLK_MHZ, 1000, 10, 0)
FAKE_VALUE(MEM_CLK_MHZ, 900, 1, 0)
FAKE_VALUE(SOC_CLK_MHZ, 150, 1, 0)
FAKE_VALUE(VRAM_USED_GIB, 1, 1, 0)
FAKE_VALUE(VRAM_TOTAL_GIB, 64, 8, 0)
FAKE_VALUE(PCIE_BANDWIDTH_MBPS, 1100, 1, 0)
FAKE_VALUE(PCI_SENT_BYTES, 1500, 1000, 0)
FAKE_VALUE(PCI_RECEIVED_BYTES, 2000, 2000, 0)
FAKE_VALUE(ECC_UMC_CORRECTABLE, 3, 1, 0)
