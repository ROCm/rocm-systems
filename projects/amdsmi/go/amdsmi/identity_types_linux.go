// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

const gpuUUIDSize int = C.AMDSMI_GPU_UUID_SIZE

type ASICInfo struct {
	MarketName            string
	VendorID              uint32
	VendorName            string
	SubvendorID           uint32
	DeviceID              uint64
	RevisionID            uint32
	Serial                string
	OAMID                 uint32
	ComputeUnits          uint32
	TargetGraphicsVersion uint64
	SubsystemID           uint32
	Flags                 uint64
	PhysicalAcceleratorID uint32
	ChipRevisionID        uint32
	ExternalRevisionID    uint32
}

type DriverInfo struct {
	Version string
	Date    string
	Name    string
}

type BoardInfo struct {
	ModelNumber      string
	ProductSerial    string
	FRUID            string
	ProductName      string
	ManufacturerName string
}

type FirmwareBlock uint32

type FirmwareInfo struct {
	ID      FirmwareBlock
	Version uint64
}

type VBIOSInfo struct {
	Name         string
	BuildDate    string
	PartNumber   string
	Version      string
	BootFirmware string
}

const maxFirmwareEntries int = C.AMDSMI_FW_ID__MAX

const (
	AMDSMI_FW_ID_SMU                      FirmwareBlock = C.AMDSMI_FW_ID_SMU
	AMDSMI_FW_ID_FIRST                    FirmwareBlock = C.AMDSMI_FW_ID_FIRST
	AMDSMI_FW_ID_CP_CE                    FirmwareBlock = C.AMDSMI_FW_ID_CP_CE
	AMDSMI_FW_ID_CP_PFP                   FirmwareBlock = C.AMDSMI_FW_ID_CP_PFP
	AMDSMI_FW_ID_CP_ME                    FirmwareBlock = C.AMDSMI_FW_ID_CP_ME
	AMDSMI_FW_ID_CP_MEC_JT1               FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC_JT1
	AMDSMI_FW_ID_CP_MEC_JT2               FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC_JT2
	AMDSMI_FW_ID_CP_MEC1                  FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC1
	AMDSMI_FW_ID_CP_MEC2                  FirmwareBlock = C.AMDSMI_FW_ID_CP_MEC2
	AMDSMI_FW_ID_RLC                      FirmwareBlock = C.AMDSMI_FW_ID_RLC
	AMDSMI_FW_ID_SDMA0                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA0
	AMDSMI_FW_ID_SDMA1                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA1
	AMDSMI_FW_ID_SDMA2                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA2
	AMDSMI_FW_ID_SDMA3                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA3
	AMDSMI_FW_ID_SDMA4                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA4
	AMDSMI_FW_ID_SDMA5                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA5
	AMDSMI_FW_ID_SDMA6                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA6
	AMDSMI_FW_ID_SDMA7                    FirmwareBlock = C.AMDSMI_FW_ID_SDMA7
	AMDSMI_FW_ID_VCN                      FirmwareBlock = C.AMDSMI_FW_ID_VCN
	AMDSMI_FW_ID_UVD                      FirmwareBlock = C.AMDSMI_FW_ID_UVD
	AMDSMI_FW_ID_VCE                      FirmwareBlock = C.AMDSMI_FW_ID_VCE
	AMDSMI_FW_ID_ISP                      FirmwareBlock = C.AMDSMI_FW_ID_ISP
	AMDSMI_FW_ID_DMCU_ERAM                FirmwareBlock = C.AMDSMI_FW_ID_DMCU_ERAM
	AMDSMI_FW_ID_DMCU_ISR                 FirmwareBlock = C.AMDSMI_FW_ID_DMCU_ISR
	AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_GPM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_SRM_MEM
	AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL    FirmwareBlock = C.AMDSMI_FW_ID_RLC_RESTORE_LIST_CNTL
	AMDSMI_FW_ID_RLC_V                    FirmwareBlock = C.AMDSMI_FW_ID_RLC_V
	AMDSMI_FW_ID_MMSCH                    FirmwareBlock = C.AMDSMI_FW_ID_MMSCH
	AMDSMI_FW_ID_PSP_SYSDRV               FirmwareBlock = C.AMDSMI_FW_ID_PSP_SYSDRV
	AMDSMI_FW_ID_PSP_SOSDRV               FirmwareBlock = C.AMDSMI_FW_ID_PSP_SOSDRV
	AMDSMI_FW_ID_PSP_TOC                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_TOC
	AMDSMI_FW_ID_PSP_KEYDB                FirmwareBlock = C.AMDSMI_FW_ID_PSP_KEYDB
	AMDSMI_FW_ID_DFC                      FirmwareBlock = C.AMDSMI_FW_ID_DFC
	AMDSMI_FW_ID_PSP_SPL                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_SPL
	AMDSMI_FW_ID_DRV_CAP                  FirmwareBlock = C.AMDSMI_FW_ID_DRV_CAP
	AMDSMI_FW_ID_MC                       FirmwareBlock = C.AMDSMI_FW_ID_MC
	AMDSMI_FW_ID_PSP_BL                   FirmwareBlock = C.AMDSMI_FW_ID_PSP_BL
	AMDSMI_FW_ID_CP_PM4                   FirmwareBlock = C.AMDSMI_FW_ID_CP_PM4
	AMDSMI_FW_ID_RLC_P                    FirmwareBlock = C.AMDSMI_FW_ID_RLC_P
	AMDSMI_FW_ID_SEC_POLICY_STAGE2        FirmwareBlock = C.AMDSMI_FW_ID_SEC_POLICY_STAGE2
	AMDSMI_FW_ID_REG_ACCESS_WHITELIST     FirmwareBlock = C.AMDSMI_FW_ID_REG_ACCESS_WHITELIST
	AMDSMI_FW_ID_IMU_DRAM                 FirmwareBlock = C.AMDSMI_FW_ID_IMU_DRAM
	AMDSMI_FW_ID_IMU_IRAM                 FirmwareBlock = C.AMDSMI_FW_ID_IMU_IRAM
	AMDSMI_FW_ID_SDMA_TH0                 FirmwareBlock = C.AMDSMI_FW_ID_SDMA_TH0
	AMDSMI_FW_ID_SDMA_TH1                 FirmwareBlock = C.AMDSMI_FW_ID_SDMA_TH1
	AMDSMI_FW_ID_CP_MES                   FirmwareBlock = C.AMDSMI_FW_ID_CP_MES
	AMDSMI_FW_ID_MES_KIQ                  FirmwareBlock = C.AMDSMI_FW_ID_MES_KIQ
	AMDSMI_FW_ID_MES_STACK                FirmwareBlock = C.AMDSMI_FW_ID_MES_STACK
	AMDSMI_FW_ID_MES_THREAD1              FirmwareBlock = C.AMDSMI_FW_ID_MES_THREAD1
	AMDSMI_FW_ID_MES_THREAD1_STACK        FirmwareBlock = C.AMDSMI_FW_ID_MES_THREAD1_STACK
	AMDSMI_FW_ID_RLX6                     FirmwareBlock = C.AMDSMI_FW_ID_RLX6
	AMDSMI_FW_ID_RLX6_DRAM_BOOT           FirmwareBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT
	AMDSMI_FW_ID_RS64_ME                  FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME
	AMDSMI_FW_ID_RS64_ME_P0_DATA          FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME_P0_DATA
	AMDSMI_FW_ID_RS64_ME_P1_DATA          FirmwareBlock = C.AMDSMI_FW_ID_RS64_ME_P1_DATA
	AMDSMI_FW_ID_RS64_PFP                 FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP
	AMDSMI_FW_ID_RS64_PFP_P0_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP_P0_DATA
	AMDSMI_FW_ID_RS64_PFP_P1_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_PFP_P1_DATA
	AMDSMI_FW_ID_RS64_MEC                 FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC
	AMDSMI_FW_ID_RS64_MEC_P0_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P0_DATA
	AMDSMI_FW_ID_RS64_MEC_P1_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P1_DATA
	AMDSMI_FW_ID_RS64_MEC_P2_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P2_DATA
	AMDSMI_FW_ID_RS64_MEC_P3_DATA         FirmwareBlock = C.AMDSMI_FW_ID_RS64_MEC_P3_DATA
	AMDSMI_FW_ID_PPTABLE                  FirmwareBlock = C.AMDSMI_FW_ID_PPTABLE
	AMDSMI_FW_ID_PSP_SOC                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_SOC
	AMDSMI_FW_ID_PSP_DBG                  FirmwareBlock = C.AMDSMI_FW_ID_PSP_DBG
	AMDSMI_FW_ID_PSP_INTF                 FirmwareBlock = C.AMDSMI_FW_ID_PSP_INTF
	AMDSMI_FW_ID_RLX6_CORE1               FirmwareBlock = C.AMDSMI_FW_ID_RLX6_CORE1
	AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1     FirmwareBlock = C.AMDSMI_FW_ID_RLX6_DRAM_BOOT_CORE1
	AMDSMI_FW_ID_RLCV_LX7                 FirmwareBlock = C.AMDSMI_FW_ID_RLCV_LX7
	AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST    FirmwareBlock = C.AMDSMI_FW_ID_RLC_SAVE_RESTORE_LIST
	AMDSMI_FW_ID_ASD                      FirmwareBlock = C.AMDSMI_FW_ID_ASD
	AMDSMI_FW_ID_TA_RAS                   FirmwareBlock = C.AMDSMI_FW_ID_TA_RAS
	AMDSMI_FW_ID_TA_XGMI                  FirmwareBlock = C.AMDSMI_FW_ID_TA_XGMI
	AMDSMI_FW_ID_RLC_SRLG                 FirmwareBlock = C.AMDSMI_FW_ID_RLC_SRLG
	AMDSMI_FW_ID_RLC_SRLS                 FirmwareBlock = C.AMDSMI_FW_ID_RLC_SRLS
	AMDSMI_FW_ID_PM                       FirmwareBlock = C.AMDSMI_FW_ID_PM
	AMDSMI_FW_ID_DMCU                     FirmwareBlock = C.AMDSMI_FW_ID_DMCU
	AMDSMI_FW_ID_PLDM_BUNDLE              FirmwareBlock = C.AMDSMI_FW_ID_PLDM_BUNDLE
	AMDSMI_FW_ID__MAX                     FirmwareBlock = C.AMDSMI_FW_ID__MAX
)
