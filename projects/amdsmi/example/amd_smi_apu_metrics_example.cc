/*
 * Standalone debug tool to read and print APU metrics from a gpu_metrics binary file.
 * Supports gpu_metrics v2.4 and v3.0.
 *
 * Usage:
 *   ./apu_metrics_dump /path/to/gpu_metrics
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

static std::string fmt_w_from_mw16(uint16_t v) {
  if (v == U16_NA) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%.0f W", v / 1000.0);
  return buf;
}

static std::string fmt_w_from_mw32(uint32_t v) {
  if (v == U32_NA) return "N/A";
  char buf[32];
  snprintf(buf, sizeof(buf), "%.0f W", v / 1000.0);
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

static const char* throttle_status_str(uint32_t v) {
  if (v == U32_NA) return "N/A";
  if (v == 0) return "UNTHROTTLED";
  return "THROTTLED";
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

  printf("    USAGE:\n");
  printf("        GFX_ACTIVITY: %s\n", fmt_percent(m.average_gfx_activity).c_str());
  printf("        UMC_ACTIVITY: N/A\n");
  printf("        MM_ACTIVITY: N/A\n");
  printf("        VCN_ACTIVITY: [%s, N/A, N/A, N/A]\n",
         fmt_percent(m.average_vcn_activity).c_str());
  printf("        JPEG_ACTIVITY: [");
  for (int i = 0; i < 32; ++i) {
    if (i > 0) printf(", ");
    printf("N/A");
  }
  printf("]\n");
  printf("        GFX_BUSY_INST: N/A\n");
  printf("        JPEG_BUSY: N/A\n");
  printf("        VCN_BUSY: N/A\n");

  printf("    POWER:\n");
  printf("        SOCKET_POWER: %s\n", fmt_w_from_mw32(m.average_socket_power).c_str());
  printf("        GFX_VOLTAGE: N/A\n");
  printf("        SOC_VOLTAGE: N/A\n");
  printf("        MEM_VOLTAGE: N/A\n");
  const bool is_throttled =
      (m.throttle_residency_prochot != 0 && m.throttle_residency_prochot != U32_NA) ||
      (m.throttle_residency_spl != 0 && m.throttle_residency_spl != U32_NA) ||
      (m.throttle_residency_fppt != 0 && m.throttle_residency_fppt != U32_NA) ||
      (m.throttle_residency_sppt != 0 && m.throttle_residency_sppt != U32_NA) ||
      (m.throttle_residency_thm_core != 0 && m.throttle_residency_thm_core != U32_NA) ||
      (m.throttle_residency_thm_gfx != 0 && m.throttle_residency_thm_gfx != U32_NA) ||
      (m.throttle_residency_thm_soc != 0 && m.throttle_residency_thm_soc != U32_NA);
  printf("        THROTTLE_STATUS: %s\n", is_throttled ? "THROTTLED" : "UNTHROTTLED");
  printf("        POWER_MANAGEMENT: N/A\n");

  printf("    CLOCK:\n");
  printf("        GFX_0:\n");
  printf("            CLK: %s\n", fmt_mhz(m.average_gfxclk_frequency).c_str());
  printf("            MIN_CLK: N/A\n");
  printf("            MAX_CLK: %s\n", fmt_mhz(m.current_gfx_maxfreq).c_str());
  printf("            DEEP_SLEEP: N/A\n");
  printf("        MEM_0:\n");
  printf("            CLK: %s\n", fmt_mhz(m.average_uclk_frequency).c_str());
  printf("            MIN_CLK: N/A\n");
  printf("            MAX_CLK: N/A\n");
  printf("            DEEP_SLEEP: N/A\n");
  printf("        VCLK_0:\n");
  printf("            CLK: %s\n", fmt_mhz(m.average_vclk_frequency).c_str());
  printf("        DCLK_0:\n");
  printf("            CLK: N/A\n");
  printf("        FCLK_0:\n");
  printf("            CLK: %s\n", fmt_mhz(m.average_fclk_frequency).c_str());
  printf("            MIN_CLK: N/A\n");
  printf("            MAX_CLK: N/A\n");
  printf("            DEEP_SLEEP: N/A\n");
  printf("        SOCCLK_0:\n");
  printf("            CLK: %s\n", fmt_mhz(m.average_socclk_frequency).c_str());
  printf("            MIN_CLK: N/A\n");
  printf("            MAX_CLK: N/A\n");
  printf("            DEEP_SLEEP: N/A\n");

  printf("    TEMPERATURE:\n");
  printf("        EDGE: %s\n", fmt_temp_centi(m.temperature_gfx).c_str());
  printf("        HOTSPOT: %s\n", fmt_temp_centi(m.temperature_soc).c_str());
  printf("        MEM: N/A\n");

  printf("    PCIE:\n");
  printf("        WIDTH: N/A\n");
  printf("        SPEED: N/A\n");
  printf("        BANDWIDTH: N/A\n");
  printf("        REPLAY_COUNT: N/A\n");
  printf("        L0_TO_RECOVERY_COUNT: N/A\n");
  printf("        REPLAY_ROLL_OVER_COUNT: N/A\n");
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
