// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

func GetUUID(h ProcessorHandle) (string, error) {
	const op = "amdsmi_get_gpu_device_uuid"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (string, error) {
		var out [C.AMDSMI_GPU_UUID_SIZE]C.char
		size := C.uint(len(out))
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_device_uuid(p, &size, &out[0])); err != nil {
			return "", err
		}
		n, err := checkedCount(op, uint64(size), len(out))
		if err != nil {
			return "", err
		}
		return boundedString(&out[0], n), nil
	})
}

func GetASICInfo(h ProcessorHandle) (ASICInfo, error) {
	const op = "amdsmi_get_gpu_asic_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ASICInfo, error) {
		var out C.amdsmi_asic_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_asic_info(p, &out)); err != nil {
			return ASICInfo{}, err
		}
		return ASICInfo{
			MarketName:  boundedString(&out.market_name[0], len(out.market_name)),
			VendorID:    uint32(out.vendor_id),
			VendorName:  boundedString(&out.vendor_name[0], len(out.vendor_name)),
			SubvendorID: uint32(out.subvendor_id), DeviceID: uint64(out.device_id),
			RevisionID: uint32(out.rev_id),
			Serial:     boundedString(&out.asic_serial[0], len(out.asic_serial)),
			OAMID:      uint32(out.oam_id), ComputeUnits: uint32(out.num_of_compute_units),
			TargetGraphicsVersion: uint64(out.target_graphics_version),
			SubsystemID:           uint32(out.subsystem_id), Flags: uint64(out.flags),
			PhysicalAcceleratorID: uint32(out.physical_acc_id),
			ChipRevisionID:        uint32(out.chip_rev_id), ExternalRevisionID: uint32(out.external_rev_id),
		}, nil
	})
}

func GetDriverInfo(h ProcessorHandle) (DriverInfo, error) {
	const op = "amdsmi_get_gpu_driver_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (DriverInfo, error) {
		var out C.amdsmi_driver_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_driver_info(p, &out)); err != nil {
			return DriverInfo{}, err
		}
		return DriverInfo{
			Version: boundedString(&out.driver_version[0], len(out.driver_version)),
			Date:    boundedString(&out.driver_date[0], len(out.driver_date)),
			Name:    boundedString(&out.driver_name[0], len(out.driver_name)),
		}, nil
	})
}

func GetBoardInfo(h ProcessorHandle) (BoardInfo, error) {
	const op = "amdsmi_get_gpu_board_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (BoardInfo, error) {
		var out C.amdsmi_board_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_board_info(p, &out)); err != nil {
			return BoardInfo{}, err
		}
		return BoardInfo{
			ModelNumber:      boundedString(&out.model_number[0], len(out.model_number)),
			ProductSerial:    boundedString(&out.product_serial[0], len(out.product_serial)),
			FRUID:            boundedString(&out.fru_id[0], len(out.fru_id)),
			ProductName:      boundedString(&out.product_name[0], len(out.product_name)),
			ManufacturerName: boundedString(&out.manufacturer_name[0], len(out.manufacturer_name)),
		}, nil
	})
}

func GetFirmwareInfo(h ProcessorHandle) ([]FirmwareInfo, error) {
	const op = "amdsmi_get_fw_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) ([]FirmwareInfo, error) {
		var out C.amdsmi_fw_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_fw_info(p, &out)); err != nil {
			return nil, err
		}
		n, err := checkedCount(op, uint64(out.num_fw_info), len(out.fw_info_list))
		if err != nil {
			return nil, err
		}
		result := make([]FirmwareInfo, n)
		for i := range result {
			result[i] = FirmwareInfo{ID: FirmwareBlock(out.fw_info_list[i].fw_id),
				Version: uint64(out.fw_info_list[i].fw_version)}
		}
		return result, nil
	})
}

func GetVBIOSInfo(h ProcessorHandle) (VBIOSInfo, error) {
	const op = "amdsmi_get_gpu_vbios_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VBIOSInfo, error) {
		var out C.amdsmi_vbios_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vbios_info(p, &out)); err != nil {
			return VBIOSInfo{}, err
		}
		return VBIOSInfo{
			Name:         boundedString(&out.name[0], len(out.name)),
			BuildDate:    boundedString(&out.build_date[0], len(out.build_date)),
			PartNumber:   boundedString(&out.part_number[0], len(out.part_number)),
			Version:      boundedString(&out.version[0], len(out.version)),
			BootFirmware: boundedString(&out.boot_firmware[0], len(out.boot_firmware)),
		}, nil
	})
}
