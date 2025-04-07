#ifndef PCIE_PERFMON_REGISTERS_MI200_H
#define PCIE_PERFMON_REGISTERS_MI200_H

#include <stdint.h>

namespace PCIE_MI200 {

// -------- RX Tile TXCLK Start --------

// Step 1: PORT SEL update
const static uint32_t PCIE_PERF_CNTL_EVENT_CI_PORT_SEL = 0x11180250;

// Step 2: EVENT SEL update
const static uint32_t PCIE_PERF_CNTL_TXCLK1 = 0x11180204;
const static uint32_t PCIE_PERF_CNTL_TXCLK2 = 0x11180210;
const static uint32_t PCIE_PERF_CNTL_TXCLK3 = 0x1118021C;  //#
const static uint32_t PCIE_PERF_CNTL_TXCLK4 = 0x11180228;  //#
const static uint32_t PCIE_PERF_CNTL_TXCLK5 = 0x11180258;
const static uint32_t PCIE_PERF_CNTL_TXCLK6 = 0x11180264;
const static uint32_t PCIE_PERF_CNTL_TXCLK7 = 0x11180888;
const static uint32_t PCIE_PERF_CNTL_TXCLK8 = 0x11180894;
const static uint32_t PCIE_PERF_CNTL_TXCLK9 = 0x111808A0;
const static uint32_t PCIE_PERF_CNTL_TXCLK10 = 0x111808AC;

// Steps 3 & 4: Performance counters initialization, enable:
const static uint32_t PCIE_PERF_COUNT_CNTL = 0x11180200;

// Step 5: Performance counters read:
const static uint32_t PCIE_PERF_COUNT0_TXCLK1 = 0x11180208;
const static uint32_t PCIE_PERF_COUNT0_TXCLK2 = 0x11180214;
const static uint32_t PCIE_PERF_COUNT0_TXCLK3 = 0x11180220;  //#
const static uint32_t PCIE_PERF_COUNT0_TXCLK4 = 0x1118022C;  //#
const static uint32_t PCIE_PERF_COUNT0_TXCLK5 = 0x1118025C;
const static uint32_t PCIE_PERF_COUNT0_TXCLK6 = 0x11180268;
const static uint32_t PCIE_PERF_COUNT0_TXCLK7 = 0x1118088C;
const static uint32_t PCIE_PERF_COUNT0_TXCLK8 = 0x11180898;
const static uint32_t PCIE_PERF_COUNT0_TXCLK9 = 0x111808A4;
const static uint32_t PCIE_PERF_COUNT0_TXCLK10 = 0x111808B0;

const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK1 = 0x111808E8;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK2 = 0x111808F0;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK3 = 0x111808F8;  //#
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK4 = 0x11180900;  //#
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK5 = 0x11180908;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK6 = 0x11180910;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK7 = 0x11180918;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK8 = 0x11180920;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK9 = 0x11180928;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_TXCLK10 = 0x11180930;

const static uint32_t PCIE_PERF_COUNT1_TXCLK1 = 0x1118020C;
const static uint32_t PCIE_PERF_COUNT1_TXCLK2 = 0x11180218;
const static uint32_t PCIE_PERF_COUNT1_TXCLK3 = 0x11180224;  //#
const static uint32_t PCIE_PERF_COUNT1_TXCLK4 = 0x11180230;  //#
const static uint32_t PCIE_PERF_COUNT1_TXCLK5 = 0x11180260;
const static uint32_t PCIE_PERF_COUNT1_TXCLK6 = 0x1118026C;
const static uint32_t PCIE_PERF_COUNT1_TXCLK7 = 0x11180890;
const static uint32_t PCIE_PERF_COUNT1_TXCLK8 = 0x1118089C;
const static uint32_t PCIE_PERF_COUNT1_TXCLK9 = 0x111808A8;
const static uint32_t PCIE_PERF_COUNT1_TXCLK10 = 0x111808B4;

const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK1 = 0x111808EC;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK2 = 0x111808F4;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK3 = 0x111808FC;  //#
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK4 = 0x11180904;  //#
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK5 = 0x1118090C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK6 = 0x11180914;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK7 = 0x1118091C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK8 = 0x11180924;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK9 = 0x1118092C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_TXCLK10 = 0x11180934;


// -------- RX Tile TXCLK End --------

// -------- RX Tile SCLK Start --------

// Step 1: PORT SEL update
// PCIE_PERF_CNTL_EVENT_CI_PORT_SEL

// Step 2: EVENT SEL update
const static uint32_t PCIE_PERF_CNTL_LCLK1 = 0x11180234;
const static uint32_t PCIE_PERF_CNTL_LCLK2 = 0x11180240;
const static uint32_t PCIE_PERF_CNTL_LCLK3 = 0x11180270;
const static uint32_t PCIE_PERF_CNTL_LCLK4 = 0x1118027C;
const static uint32_t PCIE_PERF_CNTL_LCLK5 = 0x111808B8;
const static uint32_t PCIE_PERF_CNTL_LCLK6 = 0x111808C4;
const static uint32_t PCIE_PERF_CNTL_LCLK7 = 0x111808D0;
const static uint32_t PCIE_PERF_CNTL_LCLK8 = 0x111808DC;

// Step 5: Performance counters read:
const static uint32_t PCIE_PERF_COUNT0_LCLK1 = 0x11180238;
const static uint32_t PCIE_PERF_COUNT0_LCLK2 = 0x11180244;
const static uint32_t PCIE_PERF_COUNT0_LCLK3 = 0x11180274;
const static uint32_t PCIE_PERF_COUNT0_LCLK4 = 0x11180280;
const static uint32_t PCIE_PERF_COUNT0_LCLK5 = 0x111808BC;
const static uint32_t PCIE_PERF_COUNT0_LCLK6 = 0x111808C8;
const static uint32_t PCIE_PERF_COUNT0_LCLK7 = 0x111808D4;
const static uint32_t PCIE_PERF_COUNT0_LCLK8 = 0x111808E0;

const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK1 = 0x11180938;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK2 = 0x11180940;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK3 = 0x11180948;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK4 = 0x11180950;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK5 = 0x11180958;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK6 = 0x11180960;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK7 = 0x11180968;
const static uint32_t PCIE_PERF_COUNT0_UPVAL_LCLK8 = 0x11180970;

const static uint32_t PCIE_PERF_COUNT1_LCLK1 = 0x1118023C;
const static uint32_t PCIE_PERF_COUNT1_LCLK2 = 0x11180248;
const static uint32_t PCIE_PERF_COUNT1_LCLK3 = 0x11180278;
const static uint32_t PCIE_PERF_COUNT1_LCLK4 = 0x11180284;
const static uint32_t PCIE_PERF_COUNT1_LCLK5 = 0x111808C0;
const static uint32_t PCIE_PERF_COUNT1_LCLK6 = 0x111808CC;
const static uint32_t PCIE_PERF_COUNT1_LCLK7 = 0x111808D8;
const static uint32_t PCIE_PERF_COUNT1_LCLK8 = 0x111808E4;

const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK1 = 0x1118093C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK2 = 0x11180944;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK3 = 0x1118094C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK4 = 0x11180954;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK5 = 0x1118095C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK6 = 0x11180964;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK7 = 0x1118096C;
const static uint32_t PCIE_PERF_COUNT1_UPVAL_LCLK8 = 0x11180974;

// -------- RX Tile SCLK End ----------

typedef enum {
  TX_TILE_TXCLK = 0,
  TX_TILE_SCLK = 1,
  RX_TILE_TXCLK = 2,
  RX_TILE_SCLK = 3,
  LC_TILE_TXCLK = 4
} pcie_event_category_t;

struct pcie_event_t {
  pcie_event_t(int id, pcie_event_category_t cat) : event_id(id), event_category(cat) {}
  int event_id;
  pcie_event_category_t event_category;
};

const static std::map<std::string, pcie_event_t> pcie_events_table = {
    {"RX_PERF_RXP_RX_TailEdb_A[0]", {2, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEdb_A[1]", {3, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEdb_A[2]", {4, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEdb_A[3]", {5, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEnd_A[0]", {6, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEnd_A[1]", {7, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEnd_A[2]", {8, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_TailEnd_A[3]", {9, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadSdp_A[0]", {10, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadSdp_A[1]", {11, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadSdp_A[2]", {12, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadSdp_A[3]", {13, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadStp_A[0]", {14, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadStp_A[1]", {15, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadStp_A[2]", {16, RX_TILE_TXCLK}},
    {"RX_PERF_RXP_RX_HeadStp_A[3]", {17, RX_TILE_TXCLK}},
    {"RX_PERF_RXCRC_nullified_tlp_A", {18, RX_TILE_TXCLK}},
    {"RX_PERF_RXCRC_valid_crc_A", {19, RX_TILE_TXCLK}},
    {"RX_PERF_RXCRC_invalid_crc_A", {20, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_vendor_type1_A", {21, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_vendor_type0_A", {22, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_set_slot_power_limit_A", {23, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_unlock_A", {24, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_err_fatal_A", {25, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_err_nonfatal_A", {26, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_err_corr_A", {27, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_pme_to_ack_A", {28, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_pme_turn_off_A", {29, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_pm_pme_A", {30, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_pm_active_state_nak_A", {31, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_deassert_intd_A", {32, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_deassert_intc_A", {33, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_deassert_intb_A", {34, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_deassert_inta_A", {35, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_assert_intd_A", {36, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_assert_intc_A", {37, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_assert_intb_A", {38, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_assert_inta_A", {39, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_valid_A", {40, RX_TILE_TXCLK}},
    {"RX_PERF_RMSG_unsupported_A", {41, RX_TILE_TXCLK}},
    {"RX_PERF_RCB_unexpected_cpl_A", {42, RX_TILE_TXCLK}},
    {"RX_PERF_RCB_timeout_cpl_A", {43, RX_TILE_TXCLK}},
    {"RX_PERF_HDS_tlphdrvalid_A", {44, RX_TILE_TXCLK}},
    {"RX_PERF_HDS_tlpdatavalid_A", {45, RX_TILE_TXCLK}},
    {"RX_PERF_GAN_bad_tlp_A", {46, RX_TILE_TXCLK}},
    {"RX_PERF_GAN_nak_A", {47, RX_TILE_TXCLK}},
    {"RX_PERF_GAN_ack_A", {48, RX_TILE_TXCLK}},
    {"RX_PERF_FE_unsupported_req_A", {49, RX_TILE_TXCLK}},
    {"RX_PERF_FE_unsupported_cpl_A", {50, RX_TILE_TXCLK}},
    {"RX_PERF_FE_unexpected_cpl_A", {51, RX_TILE_TXCLK}},
    {"RX_PERF_FE_poisoned_tlp_A", {52, RX_TILE_TXCLK}},
    {"RX_PERF_FE_poisoned_cpl_A", {53, RX_TILE_TXCLK}},
    {"RX_PERF_FE_malformed_tlp_A", {54, RX_TILE_TXCLK}},
    {"RX_PERF_FE_cpl_abort_A", {55, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_MSG_A", {56, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_CFG_WR_A", {57, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_CFG_RD_A", {58, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_IO_WR_A", {59, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_IO_RD_A", {60, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_MEM_WR_A", {61, RX_TILE_TXCLK}},
    {"RX_PERF_FE_request_MEM_RD_A", {62, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_MST_gt16_A", {63, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_MST_9to16_A", {64, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_MST_5to8_A", {65, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_MST_2to4_A", {66, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_MST_1_A", {67, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_gt32_A", {68, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_17to32_A", {69, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_9to16_A", {70, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_5to8_A", {71, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_2to4_A", {72, RX_TILE_TXCLK}},
    {"RX_PERF_FE_length_SLV_1_A", {73, RX_TILE_TXCLK}},
    {"RX_PERF_FE_cpl_status_CA_A", {74, RX_TILE_TXCLK}},
    {"RX_PERF_FE_cpl_status_CRS_A", {75, RX_TILE_TXCLK}},
    {"RX_PERF_FE_cpl_status_UR_A", {76, RX_TILE_TXCLK}},
    {"RX_PERF_FE_cpl_status_SC_A", {77, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_pm_active_state_request_l1_A", {78, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_pm_request_ack_A", {79, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_pm_enter_l23_A", {80, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_pm_enter_l1_A", {81, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_error_A", {82, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_crc_err_A", {83, RX_TILE_TXCLK}},
    {"SB_PERF_FCC_npd_0", {84, RX_TILE_TXCLK}},
    {"SB_PERF_FCC_pd_0", {85, RX_TILE_TXCLK}},
    {"SB_PERF_FCC_nph_0", {86, RX_TILE_TXCLK}},
    {"SB_PERF_FCC_ph_0", {87, RX_TILE_TXCLK}},
    {"SB_PERF_fail_crc_rd_hdr_0", {88, RX_TILE_TXCLK}},
    {"SB_PERF_pass_crc_rd_hdr_0", {89, RX_TILE_TXCLK}},
    {"SB_PERF_fail_crc_wr_hdr_0", {90, RX_TILE_TXCLK}},
    {"SB_PERF_pass_crc_wr_hdr_0", {91, RX_TILE_TXCLK}},
    {"SB_PERF_fail_crc_data_0", {92, RX_TILE_TXCLK}},
    {"SB_PERF_pass_crc_data_0", {93, RX_TILE_TXCLK}},
    {"SB_PERF_invalid_crc_0", {94, RX_TILE_TXCLK}},
    {"SB_PERF_valid_crc_0", {95, RX_TILE_TXCLK}},
    {"SB_PERF_rd_hdr_WEN_0", {96, RX_TILE_TXCLK}},
    {"SB_PERF_wr_hdr_WEN_0", {97, RX_TILE_TXCLK}},
    {"SB_PERF_data_WEN_0", {98, RX_TILE_TXCLK}},
    {"SB_PERF_non_post_rd_from_FE", {99, RX_TILE_TXCLK}},
    {"SB_PERF_non_post_wr_from_FE", {100, RX_TILE_TXCLK}},
    {"SB_PERF_post_req_from_FE", {101, RX_TILE_TXCLK}},
    {"SB_PERF_non_post_rd_from_FE_0", {102, RX_TILE_TXCLK}},
    {"SB_PERF_non_post_wr_from_FE_0", {103, RX_TILE_TXCLK}},
    {"SB_PERF_post_req_from_FE_0", {104, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_nak_A", {111, RX_TILE_TXCLK}},
    {"RX_PERF_DLLP_ack_A", {112, RX_TILE_TXCLK}},
    {"RX_PERF_allErrors_A", {113, RX_TILE_TXCLK}},
    {"perf_PG_COUNT", {175, RX_TILE_TXCLK}},
    {"perf_NOT_POWER_GATED", {176, RX_TILE_TXCLK}},
    {"perf_POWER_GATED", {177, RX_TILE_TXCLK}},

    {"SB_PERF_non_post_rd_to_HI", {2, RX_TILE_SCLK}},
    {"SB_PERF_non_post_wr_to_HI", {3, RX_TILE_SCLK}},
    {"SB_PERF_post_req_to_HI", {4, RX_TILE_SCLK}},
    {"SB_PERF_non_post_rd_to_HI_0", {5, RX_TILE_SCLK}},
    {"SB_PERF_non_post_wr_to_HI_0", {6, RX_TILE_SCLK}},
    {"SB_PERF_post_req_to_HI_0", {7, RX_TILE_SCLK}},
    {"SB_PERF_rd_hdr_REN_0", {8, RX_TILE_SCLK}},
    {"SB_PERF_wr_hdr_REN_0", {9, RX_TILE_SCLK}},
    {"SB_PERF_data_REN_0", {10, RX_TILE_SCLK}},
    {"SB_PERF_rd_hdr_empty_0", {11, RX_TILE_SCLK}},
    {"SB_PERF_wr_hdr_empty_0", {12, RX_TILE_SCLK}},
    {"SB_PERF_data_empty_0", {13, RX_TILE_SCLK}},
    {"CI_PERF_slv_total128BRdCpl", {29, RX_TILE_SCLK}},
    {"CI_PERF_slv_total32BMemRdTx", {30, RX_TILE_SCLK}},
    {"CI_PERF_slv_total64BMemRdTx", {31, RX_TILE_SCLK}},
    {"CI_PERF_slv_total16BMemWrTx", {32, RX_TILE_SCLK}},
    {"CI_PERF_slv_total32BMemWrTx", {33, RX_TILE_SCLK}},
    {"CI_PERF_slv_total64BMemWrTx", {34, RX_TILE_SCLK}},
    {"CI_PERF_slv_totalTx", {35, RX_TILE_SCLK}},
    {"CI_PERF_slv_stallGrantGen", {36, RX_TILE_SCLK}},
    {"CI_PERF_slv_totalGrant", {37, RX_TILE_SCLK}},
    {"CI_PERF_slv_txPending", {38, RX_TILE_SCLK}},
    {"CI_PERF_slv_numMemRdLT32B", {39, RX_TILE_SCLK}},
    {"CI_PERF_slv_numMemRdLT16B", {40, RX_TILE_SCLK}},
    {"CI_PERF_slv_totalMemTx", {41, RX_TILE_SCLK}},
    {"CI_PERF_slv_totalMemRdTx", {42, RX_TILE_SCLK}},
    {"CI_PERF_slv_totalMemWrTx", {43, RX_TILE_SCLK}},
    {"CI_PERF_slv_numGrant0", {44, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntOverFlow_ns0", {45, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntUnderFlow_ns0", {46, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntOverFlow_s0", {47, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntUnderFlow_s0", {48, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntOverFlow0", {49, RX_TILE_SCLK}},
    {"CI_PERF_slv_portCntUnderFlow0", {50, RX_TILE_SCLK}},
    {"CI_PERF_slv_npNotAccepted_ns0", {51, RX_TILE_SCLK}},
    {"CI_PERF_slv_npNotAccepted_s0", {52, RX_TILE_SCLK}},
    {"CI_PERF_slv_num128BRdCpl0", {53, RX_TILE_SCLK}},
    {"CI_PERF_slv_num32BMemRdTx0", {54, RX_TILE_SCLK}},
    {"CI_PERF_slv_num64BMemRdTx0", {55, RX_TILE_SCLK}},
    {"CI_PERF_slv_num16BMemWrTx0", {56, RX_TILE_SCLK}},
    {"CI_PERF_slv_num32BMemWrTx0", {57, RX_TILE_SCLK}},
    {"CI_PERF_slv_num64BMemWrTx0", {58, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemRd_Bandwidth0", {59, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemWr_Bandwidth0", {60, RX_TILE_SCLK}},
    {"TX_PERF_S_RCLK_s_tag_buf_empty", {61, RX_TILE_SCLK}},
    {"P_request_latency_500ns_or_more", {62, RX_TILE_SCLK}},
    {"P_request_latency_250_to_500ns", {63, RX_TILE_SCLK}},
    {"P_request_latency_100_to_250ns", {64, RX_TILE_SCLK}},
    {"P_request_latency_100ns_or_less", {65, RX_TILE_SCLK}},
    {"NP_request_latency_500ns_or_more", {66, RX_TILE_SCLK}},
    {"NP_request_latency_250_to_500ns", {67, RX_TILE_SCLK}},
    {"NP_request_latency_100_to_250ns", {68, RX_TILE_SCLK}},
    {"NP_request_latency_100ns_or_less", {69, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemRd_wait_for_cpl_slot[0]", {70, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemRd_wait_for_tag[0]", {71, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemRd_wait_for_d_credit[0]", {72, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemRd_wait_for_h_credit[0]", {73, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemWr_wait_for_tag[0]", {74, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemWr_wait_for_d_credit[0]", {75, RX_TILE_SCLK}},
    {"CI_PERF_slv_MemWr_wait_for_h_credit[0]", {76, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC1_no_tags_q", {77, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC1_data_credits_q", {78, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC1_req_credits_q", {79, RX_TILE_SCLK}},
    {"CISLV_PERF_no_cpl_slots_q[0]", {80, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC0_no_tags_q", {81, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC0_data_credits_q", {82, RX_TILE_SCLK}},
    {"CISLV_PERF_no_VC0_req_credits_q", {83, RX_TILE_SCLK}}};

}  // namespace PCIE_MI200


#endif