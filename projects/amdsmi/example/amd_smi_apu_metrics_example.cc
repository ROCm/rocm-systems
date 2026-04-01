/*
 * Standalone debug tool to read and print APU metrics from a gpu_metrics binary file.
 * Supports gpu_metrics v2.4 and v3.0.
 *
 * Build:
 *   g++ -std=c++17 -o apu_metrics_dump example/amd_smi_apu_metrics_dump.cc
 *
 * Usage:
 *   ./apu_metrics_dump /path/to/gpu_metrics
 *   ./apu_metrics_dump /sys/class/drm/card0/device/gpu_metrics
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// -- Header (matches kernel layout) --
struct metrics_header {
  uint16_t structure_size;
  uint8_t  format_revision;
  uint8_t  content_revision;
};

// -- APU Metrics v2.4 (format_rev=2, content_rev=4) --
struct apu_metrics_v24 {
  metrics_header header;

  uint16_t temperature_gfx;
  uint16_t temperature_soc;
  uint16_t temperature_core[8];
  uint16_t temperature_l3[2];

  uint16_t average_gfx_activity;
  uint16_t average_mm_activity;

  uint64_t system_clock_counter;

  uint16_t average_socket_power;
  uint16_t average_cpu_power;
  uint16_t average_soc_power;
  uint16_t average_gfx_power;
  uint16_t average_core_power[8];

  uint16_t average_gfxclk_frequency;
  uint16_t average_socclk_frequency;
  uint16_t average_uclk_frequency;
  uint16_t average_fclk_frequency;
  uint16_t average_vclk_frequency;
  uint16_t average_dclk_frequency;

  uint16_t current_gfxclk;
  uint16_t current_socclk;
  uint16_t current_uclk;
  uint16_t current_fclk;
  uint16_t current_vclk;
  uint16_t current_dclk;
  uint16_t current_coreclk[8];
  uint16_t current_l3clk[2];

  uint32_t throttle_status;

  uint16_t fan_pwm;
  uint16_t padding[3];

  uint64_t indep_throttle_status;

  uint16_t average_temperature_gfx;
  uint16_t average_temperature_soc;
  uint16_t average_temperature_core[8];
  uint16_t average_temperature_l3[2];

  uint16_t average_cpu_voltage;
  uint16_t average_soc_voltage;
  uint16_t average_gfx_voltage;

  uint16_t average_cpu_current;
  uint16_t average_soc_current;
  uint16_t average_gfx_current;
};

// -- APU Metrics v3.0 (format_rev=3, content_rev=0) --
struct apu_metrics_v30 {
  metrics_header header;

  uint16_t temperature_gfx;
  uint16_t temperature_soc;
  uint16_t temperature_core[16];
  uint16_t temperature_skin;

  uint16_t average_gfx_activity;
  uint16_t average_vcn_activity;
  uint16_t average_ipu_activity[8];
  uint16_t average_core_c0_activity[16];
  uint16_t average_dram_reads;
  uint16_t average_dram_writes;
  uint16_t average_ipu_reads;
  uint16_t average_ipu_writes;

  uint64_t system_clock_counter;

  uint32_t average_socket_power;
  uint16_t average_ipu_power;
  uint32_t average_apu_power;
  uint32_t average_gfx_power;
  uint32_t average_dgpu_power;
  uint32_t average_all_core_power;
  uint16_t average_core_power[16];
  uint16_t average_sys_power;
  uint16_t stapm_power_limit;
  uint16_t current_stapm_power_limit;

  uint16_t average_gfxclk_frequency;
  uint16_t average_socclk_frequency;
  uint16_t average_vpeclk_frequency;
  uint16_t average_ipuclk_frequency;
  uint16_t average_fclk_frequency;
  uint16_t average_vclk_frequency;
  uint16_t average_uclk_frequency;
  uint16_t average_mpipu_frequency;

  uint16_t current_coreclk[16];
  uint16_t current_core_maxfreq;
  uint16_t current_gfx_maxfreq;

  uint32_t throttle_residency_prochot;
  uint32_t throttle_residency_spl;
  uint32_t throttle_residency_fppt;
  uint32_t throttle_residency_sppt;
  uint32_t throttle_residency_thm_core;
  uint32_t throttle_residency_thm_gfx;
  uint32_t throttle_residency_thm_soc;

  uint32_t time_filter_alphavalue;
};

// -- Helpers --

static constexpr uint16_t U16_NA = 0xFFFF;
static constexpr uint32_t U32_NA = 0xFFFFFFFF;
static constexpr uint64_t U64_NA = 0xFFFFFFFFFFFFFFFFULL;

static std::string fmt_u16(uint16_t v, const char* unit) {
  if (v == U16_NA) return "N/A";
  return std::to_string(v) + " " + unit;
}

static std::string fmt_temp_centi(uint16_t v) {
  if (v == U16_NA || v == 0) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%.1f C", v / 100.0);
  return buf;
}

static std::string fmt_pwr_mw(uint16_t v) {
  if (v == U16_NA) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%u mW", v);
  return buf;
}

static std::string fmt_pwr_mw32(uint32_t v) {
  if (v == U32_NA) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%u mW", v);
  return buf;
}

static std::string fmt_mhz(uint16_t v) {
  if (v == U16_NA) return "N/A";
  return std::to_string(v) + " MHz";
}

static std::string fmt_percent_centi(uint16_t v) {
  if (v == U16_NA) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%.1f %%", v / 100.0);
  return buf;
}

static std::string fmt_percent(uint16_t v) {
  if (v == U16_NA) return "N/A";
  return std::to_string(v) + " %";
}

// -- v2.4 printer --

static void print_v24(const std::vector<uint8_t>& buf) {
  apu_metrics_v24 m;
  memcpy(&m, buf.data(), sizeof(m));

  printf("APU METRICS v2.4\n");
  printf("================\n\n");

  printf("    TEMPERATURE:\n");
  printf("        GFX:  %s\n", fmt_temp_centi(m.temperature_gfx).c_str());
  printf("        SOC:  %s\n", fmt_temp_centi(m.temperature_soc).c_str());
  for (int i = 0; i < 8; ++i)
    printf("        CORE[%d]: %s\n", i, fmt_temp_centi(m.temperature_core[i]).c_str());
  for (int i = 0; i < 2; ++i)
    printf("        L3[%d]:   %s\n", i, fmt_temp_centi(m.temperature_l3[i]).c_str());

  printf("\n    AVERAGE TEMPERATURE:\n");
  printf("        GFX:  %s\n", fmt_temp_centi(m.average_temperature_gfx).c_str());
  printf("        SOC:  %s\n", fmt_temp_centi(m.average_temperature_soc).c_str());
  for (int i = 0; i < 8; ++i)
    printf("        CORE[%d]: %s\n", i, fmt_temp_centi(m.average_temperature_core[i]).c_str());
  for (int i = 0; i < 2; ++i)
    printf("        L3[%d]:   %s\n", i, fmt_temp_centi(m.average_temperature_l3[i]).c_str());

  printf("\n    UTILIZATION:\n");
  printf("        GFX_ACTIVITY: %s\n", fmt_percent_centi(m.average_gfx_activity).c_str());
  printf("        MM_ACTIVITY:  %s\n", fmt_percent_centi(m.average_mm_activity).c_str());

  printf("\n    POWER:\n");
  printf("        SOCKET_POWER: %s\n", fmt_pwr_mw(m.average_socket_power).c_str());
  printf("        CPU_POWER:    %s\n", fmt_pwr_mw(m.average_cpu_power).c_str());
  printf("        SOC_POWER:    %s\n", fmt_pwr_mw(m.average_soc_power).c_str());
  printf("        GFX_POWER:    %s\n", fmt_pwr_mw(m.average_gfx_power).c_str());
  for (int i = 0; i < 8; ++i)
    printf("        CORE_POWER[%d]: %s\n", i, fmt_pwr_mw(m.average_core_power[i]).c_str());

  printf("\n    VOLTAGE:\n");
  printf("        CPU_VOLTAGE: %s\n", fmt_u16(m.average_cpu_voltage, "mV").c_str());
  printf("        SOC_VOLTAGE: %s\n", fmt_u16(m.average_soc_voltage, "mV").c_str());
  printf("        GFX_VOLTAGE: %s\n", fmt_u16(m.average_gfx_voltage, "mV").c_str());
  printf("\n    CURRENT:\n");
  printf("        CPU_CURRENT: %s\n", fmt_u16(m.average_cpu_current, "mA").c_str());
  printf("        SOC_CURRENT: %s\n", fmt_u16(m.average_soc_current, "mA").c_str());
  printf("        GFX_CURRENT: %s\n", fmt_u16(m.average_gfx_current, "mA").c_str());

  printf("\n    AVERAGE CLOCK:\n");
  printf("        GFXCLK: %s\n", fmt_mhz(m.average_gfxclk_frequency).c_str());
  printf("        SOCCLK: %s\n", fmt_mhz(m.average_socclk_frequency).c_str());
  printf("        UCLK:   %s\n", fmt_mhz(m.average_uclk_frequency).c_str());
  printf("        FCLK:   %s\n", fmt_mhz(m.average_fclk_frequency).c_str());
  printf("        VCLK:   %s\n", fmt_mhz(m.average_vclk_frequency).c_str());
  printf("        DCLK:   %s\n", fmt_mhz(m.average_dclk_frequency).c_str());

  printf("\n    CURRENT CLOCK:\n");
  printf("        GFXCLK: %s\n", fmt_mhz(m.current_gfxclk).c_str());
  printf("        SOCCLK: %s\n", fmt_mhz(m.current_socclk).c_str());
  printf("        UCLK:   %s\n", fmt_mhz(m.current_uclk).c_str());
  printf("        FCLK:   %s\n", fmt_mhz(m.current_fclk).c_str());
  printf("        VCLK:   %s\n", fmt_mhz(m.current_vclk).c_str());
  printf("        DCLK:   %s\n", fmt_mhz(m.current_dclk).c_str());
  for (int i = 0; i < 8; ++i)
    printf("        CORECLK[%d]: %s\n", i, fmt_mhz(m.current_coreclk[i]).c_str());
  for (int i = 0; i < 2; ++i)
    printf("        L3CLK[%d]:   %s\n", i, fmt_mhz(m.current_l3clk[i]).c_str());

  printf("\n    THROTTLE:\n");
  if (m.throttle_status == U32_NA)
    printf("        STATUS: N/A\n");
  else if (m.throttle_status == 0)
    printf("        STATUS: UNTHROTTLED\n");
  else
    printf("        STATUS: 0x%08X\n", m.throttle_status);

  if (m.indep_throttle_status == U64_NA)
    printf("        INDEP_STATUS: N/A\n");
  else
    printf("        INDEP_STATUS: 0x%016lX\n", (unsigned long)m.indep_throttle_status);

  printf("\n    FAN:\n");
  printf("        PWM: %s\n", fmt_u16(m.fan_pwm, "").c_str());

  printf("\n    TIMESTAMP:\n");
  printf("        SYSTEM_CLOCK_COUNTER: %lu ns\n", (unsigned long)m.system_clock_counter);
}

// -- v3.0 printer --

static void print_v30(const std::vector<uint8_t>& buf) {
  apu_metrics_v30 m;
  memcpy(&m, buf.data(), sizeof(m));

  printf("APU METRICS v3.0\n");
  printf("================\n\n");

  printf("    TEMPERATURE:\n");
  printf("        GFX:  %s\n", fmt_temp_centi(m.temperature_gfx).c_str());
  printf("        SOC:  %s\n", fmt_temp_centi(m.temperature_soc).c_str());
  printf("        SKIN: %s\n", fmt_temp_centi(m.temperature_skin).c_str());
  for (int i = 0; i < 16; ++i) {
    if (m.temperature_core[i] != 0 && m.temperature_core[i] != U16_NA)
      printf("        CORE[%d]: %s\n", i, fmt_temp_centi(m.temperature_core[i]).c_str());
  }

  printf("\n    UTILIZATION:\n");
  printf("        GFX_ACTIVITY: %s\n", fmt_percent_centi(m.average_gfx_activity).c_str());
  printf("        VCN_ACTIVITY: %s\n", fmt_percent_centi(m.average_vcn_activity).c_str());

  printf("        IPU_ACTIVITY: [");
  for (int i = 0; i < 8; ++i) {
    if (i > 0) printf(", ");
    printf("%s", fmt_percent_centi(m.average_ipu_activity[i]).c_str());
  }
  printf("]\n");

  printf("        CORE_C0_ACTIVITY: [");
  for (int i = 0; i < 16; ++i) {
    if (i > 0) printf(", ");
    printf("%s", fmt_percent(m.average_core_c0_activity[i]).c_str());
  }
  printf("]\n");

  printf("        DRAM_READS:  %s\n", fmt_u16(m.average_dram_reads, "MB/s").c_str());
  printf("        DRAM_WRITES: %s\n", fmt_u16(m.average_dram_writes, "MB/s").c_str());
  printf("        IPU_READS:   %s\n", fmt_u16(m.average_ipu_reads, "MB/s").c_str());
  printf("        IPU_WRITES:  %s\n", fmt_u16(m.average_ipu_writes, "MB/s").c_str());

  printf("\n    POWER:\n");
  printf("        SOCKET_POWER:    %s\n", fmt_pwr_mw32(m.average_socket_power).c_str());
  printf("        APU_POWER:       %s\n", fmt_pwr_mw32(m.average_apu_power).c_str());
  printf("        GFX_POWER:       %s\n", fmt_pwr_mw32(m.average_gfx_power).c_str());
  printf("        DGPU_POWER:      %s\n", fmt_pwr_mw32(m.average_dgpu_power).c_str());
  printf("        ALL_CORE_POWER:  %s\n", fmt_pwr_mw32(m.average_all_core_power).c_str());
  printf("        IPU_POWER:       %s\n", fmt_pwr_mw(m.average_ipu_power).c_str());
  printf("        SYS_POWER:       %s\n", fmt_pwr_mw(m.average_sys_power).c_str());
  printf("        STAPM_LIMIT:     %s\n", fmt_pwr_mw(m.stapm_power_limit).c_str());
  printf("        CUR_STAPM_LIMIT: %s\n", fmt_pwr_mw(m.current_stapm_power_limit).c_str());

  printf("        CORE_POWER: [");
  for (int i = 0; i < 16; ++i) {
    if (i > 0) printf(", ");
    printf("%s", fmt_pwr_mw(m.average_core_power[i]).c_str());
  }
  printf("]\n");

  printf("\n    AVERAGE CLOCK:\n");
  printf("        GFXCLK:  %s\n", fmt_mhz(m.average_gfxclk_frequency).c_str());
  printf("        SOCCLK:  %s\n", fmt_mhz(m.average_socclk_frequency).c_str());
  printf("        UCLK:    %s\n", fmt_mhz(m.average_uclk_frequency).c_str());
  printf("        FCLK:    %s\n", fmt_mhz(m.average_fclk_frequency).c_str());
  printf("        VCLK:    %s\n", fmt_mhz(m.average_vclk_frequency).c_str());
  printf("        VPECLK:  %s\n", fmt_mhz(m.average_vpeclk_frequency).c_str());
  printf("        IPUCLK:  %s\n", fmt_mhz(m.average_ipuclk_frequency).c_str());
  printf("        MPIPU:   %s\n", fmt_mhz(m.average_mpipu_frequency).c_str());

  printf("\n    CURRENT CLOCK:\n");
  printf("        GFX_MAXFREQ:  %s\n", fmt_mhz(m.current_gfx_maxfreq).c_str());
  printf("        CORE_MAXFREQ: %s\n", fmt_mhz(m.current_core_maxfreq).c_str());
  for (int i = 0; i < 16; ++i) {
    if (m.current_coreclk[i] != 0 && m.current_coreclk[i] != U16_NA)
      printf("        CORECLK[%2d]: %s\n", i, fmt_mhz(m.current_coreclk[i]).c_str());
  }

  printf("\n    THROTTLE RESIDENCY:\n");
  auto fmt_thr = [](uint32_t v) -> std::string {
    if (v == U32_NA) return "N/A";
    if (v == 0) return "0";
    return std::to_string(v);
  };
  printf("        PROCHOT:  %s\n", fmt_thr(m.throttle_residency_prochot).c_str());
  printf("        SPL:      %s\n", fmt_thr(m.throttle_residency_spl).c_str());
  printf("        FPPT:     %s\n", fmt_thr(m.throttle_residency_fppt).c_str());
  printf("        SPPT:     %s\n", fmt_thr(m.throttle_residency_sppt).c_str());
  printf("        THM_CORE: %s\n", fmt_thr(m.throttle_residency_thm_core).c_str());
  printf("        THM_GFX:  %s\n", fmt_thr(m.throttle_residency_thm_gfx).c_str());
  printf("        THM_SOC:  %s\n", fmt_thr(m.throttle_residency_thm_soc).c_str());

  printf("\n    TIMESTAMP:\n");
  printf("        SYSTEM_CLOCK_COUNTER: %lu ns\n", (unsigned long)m.system_clock_counter);
  printf("        TIME_FILTER_ALPHA:    %u us\n", m.time_filter_alphavalue);
}

// -- main --

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <gpu_metrics_binary_file>\n", argv[0]);
    fprintf(stderr, "  e.g. %s /sys/class/drm/card0/device/gpu_metrics\n", argv[0]);
    fprintf(stderr, "       %s /home/user/gpu_metrics_1\n", argv[0]);
    return 1;
  }

  const char* path = argv[1];
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "Error: Cannot open '%s'\n", path);
    return 1;
  }

  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());

  if (buf.size() < sizeof(metrics_header)) {
    fprintf(stderr, "Error: File too small (%zu bytes)\n", buf.size());
    return 1;
  }

  metrics_header hdr;
  memcpy(&hdr, buf.data(), sizeof(hdr));

  printf("GPU: 0\n");
  printf("    HEADER:\n");
  printf("        STRUCTURE_SIZE:   %u bytes\n", hdr.structure_size);
  printf("        FORMAT_REVISION:  %u\n", hdr.format_revision);
  printf("        CONTENT_REVISION: %u\n", hdr.content_revision);
  printf("\n");

  if (hdr.format_revision == 2 && hdr.content_revision == 4) {
    if (buf.size() < sizeof(apu_metrics_v24)) {
      fprintf(stderr, "Error: File too small for v2.4 (%zu < %zu)\n",
              buf.size(), sizeof(apu_metrics_v24));
      return 1;
    }
    print_v24(buf);
  } else if (hdr.format_revision == 3 && hdr.content_revision == 0) {
    if (buf.size() < sizeof(apu_metrics_v30)) {
      fprintf(stderr, "Error: File too small for v3.0 (%zu < %zu)\n",
              buf.size(), sizeof(apu_metrics_v30));
      return 1;
    }
    print_v30(buf);
  } else {
    fprintf(stderr, "Error: Unsupported gpu_metrics version %u.%u\n",
            hdr.format_revision, hdr.content_revision);
    fprintf(stderr, "       (Expected 2.4 or 3.0 for APU metrics)\n");
    return 1;
  }

  return 0;
}
