// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
use std::fmt;

// Re-export all the alias type
pub use crate::amdsmi_wrapper::{AmdsmiEventHandleT, AmdsmiProcessorHandle, AmdsmiSocketHandle};

// Re-export all the enums type
pub use crate::amdsmi_wrapper::{
    AmdsmiCachePropertyTypeT, AmdsmiCardFormFactorT, AmdsmiClkLimitTypeT, AmdsmiClkTypeT,
    AmdsmiComputePartitionTypeT, AmdsmiCounterCommandT, AmdsmiDevPerfLevelT, AmdsmiEventGroupT,
    AmdsmiEventTypeT, AmdsmiEvtNotificationTypeT, AmdsmiFreqIndT, AmdsmiFwBlockT, AmdsmiGpuBlockT,
    AmdsmiInitFlagsT, AmdsmiLinkTypeT, AmdsmiMemoryPartitionTypeT, AmdsmiMemoryTypeT,
    AmdsmiPowerProfilePresetMasksT, AmdsmiRasErrStateT, AmdsmiStatusT,
    AmdsmiTemperatureMetricT, AmdsmiTemperatureTypeT, AmdsmiUtilizationCounterTypeT,
    AmdsmiVoltageMetricT, AmdsmiVoltageTypeT, AmdsmiXgmiStatusT, ProcessorTypeT, AmdsmiAcceleratorPartitionTypeT
};

// Re-export all the struct type
pub use crate::amdsmi_wrapper::{
    AmdMetricsTableHeaderT, AmdsmiAcceleratorPartitionProfileT, AmdsmiAsicInfoT, AmdsmiBoardInfoT,
    AmdsmiClkInfoT, AmdsmiCounterValueT, AmdsmiDpmPolicyEntryT, AmdsmiDpmPolicyT,
    AmdsmiDriverInfoT, AmdsmiEngineUsageT, AmdsmiErrorCountT, AmdsmiEvtNotificationDataT,
    AmdsmiFreqVoltRegionT, AmdsmiFrequenciesT, AmdsmiFrequencyRangeT, AmdsmiFwInfoT,
    AmdsmiGpuCacheInfoT, AmdsmiGpuCacheInfoTCache, AmdsmiGpuMetricsT, AmdsmiKfdInfoT,
    AmdsmiLinkMetricsT, AmdsmiLinkMetricsTLinks, AmdsmiNameValueT,
    AmdsmiOdVoltFreqDataT, AmdsmiP2pCapabilityT, AmdsmiPcieBandwidthT, AmdsmiPcieInfoT,
    AmdsmiPcieInfoTPcieMetric, AmdsmiPcieInfoTPcieStatic, AmdsmiPowerCapInfoT, AmdsmiPowerInfoT,
    AmdsmiPowerProfileStatusT, AmdsmiProcInfoT, AmdsmiProcInfoTEngineUsage,
    AmdsmiProcInfoTMemoryUsage, AmdsmiProcessInfoT, AmdsmiRangeT, AmdsmiRasFeatureT,
    AmdsmiRegTypeT, AmdsmiRetiredPageRecordT, AmdsmiTopologyNearestT, AmdsmiUtilizationCounterT,
    AmdsmiVbiosInfoT, AmdsmiVersionT, AmdsmiViolationStatusT, AmdsmiVramInfoT, AmdsmiVramUsageT,
    AmdsmiXgmiInfoT, AmdsmiNpsCapsT, AmdsmiNpsCapsTNpsFlags
};

//Re-export all the union type
pub use crate::amdsmi_wrapper::AmdsmiBdfT;

//Re-export the constant type
pub use crate::amdsmi_wrapper::{
    AMDSMI_MAX_AID, AMDSMI_MAX_CACHE_TYPES, AMDSMI_MAX_CONTAINER_TYPE, AMDSMI_MAX_DEVICES,
    AMDSMI_MAX_ENGINES, AMDSMI_MAX_FAN_SPEED, AMDSMI_MAX_MM_IP_COUNT, AMDSMI_MAX_NUM_CLKS,
    AMDSMI_MAX_NUM_FREQUENCIES, AMDSMI_MAX_NUM_GFX_CLKS, AMDSMI_MAX_NUM_JPEG,
    AMDSMI_MAX_NUM_PM_POLICIES, AMDSMI_MAX_NUM_VCN, AMDSMI_MAX_NUM_XGMI_LINKS,
    AMDSMI_MAX_NUM_XGMI_PHYSICAL_LINK, AMDSMI_MAX_UTILIZATION_VALUES, AMDSMI_NUM_HBM_INSTANCES,
    AMDSMI_NUM_VOLTAGE_CURVE_POINTS,
};

pub type AmdsmiResult<T> = Result<T, AmdsmiStatusT>;

//#[macro_export]
macro_rules! call_unsafe {
    ($call:expr) => {{
        let status = unsafe { $call };
        if status != AmdsmiStatusT::AmdsmiStatusSuccess {
            return Err(status);
        }
    }};
}

macro_rules! define_cstr {
    ($len:expr) => {{
        let buf = vec![0 as std::os::raw::c_char; $len as usize];
        let len = $len as usize;
        (buf, len)
    }};
}

macro_rules! cstr_to_string {
    ($vec:expr) => {{
        let c_str = unsafe { std::ffi::CStr::from_ptr($vec.as_ptr()) };
        c_str.to_string_lossy().into_owned()
    }};
}

// Implements getter methods for C string fields in the specified struct.
//
// This macro generates methods that convert C string fields to Rust `String`.
//
// # Arguments
//
// * `$struct_name` - The name of the struct for which the getter methods are being generated.
// * `$($field_name),+` - The names of the fields for which the getter methods are being generated.
//
// # Example
///
// ```rust
//
// use std::ffi::c_char;
//
// pub struct AmdsmiVbiosInfoT {
//     pub name: [c_char; 64usize],
//     pub version: [c_char; 32usize],
// }
//
// impl_cstr_getters!(AmdsmiVbiosInfoT, name, version);
// ```
macro_rules! impl_cstr_getters {
    ($struct_name:ident, $($field_name:ident),+) => {
        impl $struct_name {
            $(
                pub fn $field_name(&self) -> String {
                    unsafe {
                        std::ffi::CStr::from_ptr(self.$field_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    }
                }
            )+
        }
    };
}

impl fmt::Display for AmdsmiStatusT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status_str = crate::amdsmi_status_code_to_string(*self)
            .unwrap_or("An unknown error occurred".to_string());
        write!(f, "{}", status_str)
    }
}

impl fmt::Display for AmdsmiBdfT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_human_readable_string())
    }
}

impl AmdsmiBdfT {
    /// Converts the AmdsmiBdfT structure to a human-readable string.
    pub fn to_human_readable_string(&self) -> String {
        format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            unsafe { self.__bindgen_anon_1.domain_number() },
            unsafe { self.__bindgen_anon_1.bus_number() },
            unsafe { self.__bindgen_anon_1.device_number() },
            unsafe { self.__bindgen_anon_1.function_number() }
        )
    }
}

// Implement the getters for the C string fields in AmdsmiVbiosInfoT
impl_cstr_getters!(AmdsmiVbiosInfoT, name, build_date, part_number, version, boot_firmware);

// Implement the getters for the C string fields in AmdsmiAsicInfoT
impl_cstr_getters!(AmdsmiAsicInfoT, market_name, vendor_name, asic_serial);

// Implement the getters for the C string fields in AmdsmiBoardInfoT
impl_cstr_getters!(
    AmdsmiBoardInfoT,
    model_number,
    product_serial,
    fru_id,
    product_name,
    manufacturer_name
);

// Implement the getters for the C string fields in AmdsmiDriverInfoT
impl_cstr_getters!(AmdsmiDriverInfoT, driver_version, driver_date, driver_name);

// Implement the getters for the C string fields in AmdsmiNameValueT
impl_cstr_getters!(AmdsmiNameValueT, name);

// Implement the getters for the C string fields in AmdsmiProcInfoT
impl_cstr_getters!(AmdsmiProcInfoT, name, container_name);

// Implement the getters for the C string fields in AmdsmiDpmPolicyEntryT
impl_cstr_getters!(AmdsmiDpmPolicyEntryT, policy_description);
